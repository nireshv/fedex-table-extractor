"""
LLM extraction layer using LangChain.

Responsibilities:
- Build a provider-agnostic LangChain chain using .with_structured_output()
- Return a validated PageClassification object (never raises on LLM errors)
- Log classification outcomes for observability
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from src.config import Settings
from src.models import PageClassification

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """\
Classify this FedEx Service Guide page. Return structure metadata only — no rate values.

Set skipped=true if the page has no weight-based rate table (zone charts, fee pages, T&C, cover, checkmark grids).

For rate table pages return:
- table_type: one of us_package, us_express_freight, us_multiweight, sameday,
  intl_package_export, intl_package_import, intl_premium, ground_domestic, ground_ak_hi, ground_canada
- direction: domestic / us_export / us_import
- zone: as printed (e.g. "2", "A", "17"); null only for sameday
- services: service name columns left-to-right, exact names, include delivery_commitment if shown
- has_envelope_row: true if FedEx Envelope is a separate row
- has_pak_row: true if FedEx Pak is a separate row

table_type guide:
  us_package          US domestic package, zones 2–8
  us_express_freight  US domestic freight per-lb, zones 2–16
  us_multiweight      Express Multiweight per-lb bulk (100 lbs+)
  us_one_rate         FedEx One Rate® flat pricing by package type (Envelope/Pak/Box/Tube), zones 2/3-4/5-8
  sameday             FedEx SameDay / NFO, no zone
  intl_package_export Intl package US export, zones A–O
  intl_package_import Intl package US import, zones A–O
  intl_premium        Intl Premium freight
  ground_domestic     FedEx Ground / Home Delivery, zones 2–8
  ground_ak_hi        Ground Alaska / Hawaii
  ground_canada       Intl Ground Canada, zones 51/54

services:
  - For pages where service names are COLUMN headers (e.g., us_package zone tables):
    list service name columns left-to-right.
  - For us_express_freight and us_multiweight pages: zones (2, 3, 4, …) are the column
    headers; each table represents ONE freight service. List services in table order
    (top to bottom), one per table. Service names appear as text above each table
    (e.g., "FedEx First Overnight® Freight", "FedEx 1Day® Freight").
  - For intl_package_export / intl_package_import box-rate pages (10kg Box / 25kg Box):
    zones A–O are column headers; each table represents ONE service + ONE package type.
    List services in table order (top to bottom), one per table.
    Set package_type in each ServiceInfo entry (e.g., "FedEx 10kg Box", "FedEx 25kg Box").
    The service name and package type appear as a header above each table, e.g.
    "FedEx International Priority Express: FedEx 10kg Box".
  - Use exact FedEx names only — no weight or zone column labels.
"""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(settings: Settings) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic"
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            ) from e
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=16000,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from e
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_completion_tokens=16000,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")


def build_classification_chain(settings: Settings) -> Runnable:
    """
    Build the lightweight classification chain:
        prompt messages → LLM → structured PageClassification output

    Used by the hybrid pipeline: LLM identifies table type / services / zone,
    then code parses the actual values from pdfplumber table data.
    """
    llm = get_llm(settings)
    return llm.with_structured_output(PageClassification)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Wraps the LangChain chain with retry logic and error isolation.

    All public methods return a PageClassification — they never raise.
    Failures are logged and result in a skipped classification.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._classification_chain = build_classification_chain(settings)

    async def aclassify_page(self, page_content: str, page_number: int) -> "PageClassification":
        """
        Asynchronously classify a page (table_type, zone, services).
        Returns a skipped PageClassification on any error.
        """
        messages = _build_classification_messages(page_content)
        max_retries = self._settings.max_retries
        wait_base = self._settings.retry_wait_seconds
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                import asyncio
                result = await self._classification_chain.ainvoke(messages)
                logger.info(
                    "Classification complete",
                    extra={"page": page_number, "table_type": result.table_type, "skipped": result.skipped},
                )
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = wait_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Classification call failed, retrying",
                        extra={"page": page_number, "attempt": attempt, "wait": wait, "error": str(exc)},
                    )
                    await asyncio.sleep(wait)

        logger.error("Classification failed after retries", extra={"page": page_number, "error": str(last_exc)})
        return PageClassification(skipped=True, skip_reason=f"Classification failed: {last_exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_classification_messages(page_content: str) -> list[Any]:
    """Build the message list for the classification (hybrid) call.

    Classification only needs table headers and the first few rows — not all data rows.
    Truncating to 2 000 chars covers the page title, zone label, and service column headers
    while cutting ~85 % of a typical rate-table page's tokens.
    """
    _CLASSIFICATION_MAX_CHARS = 2_000
    truncated = page_content[:_CLASSIFICATION_MAX_CHARS]
    return [
        SystemMessage(content=CLASSIFICATION_PROMPT),
        HumanMessage(content=truncated),
    ]
