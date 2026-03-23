"""
LLM extraction layer using LangChain.

Responsibilities:
- Build a provider-agnostic LangChain chain using .with_structured_output()
- Apply tenacity retry with exponential backoff for transient API failures
- Return a validated PageExtraction object (never raises on LLM/validation errors)
- Log extraction outcomes for observability
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings
from src.models import PageClassification, PageExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a deterministic data extraction engine specializing in FedEx shipping rate tables.

Your task: read the FedEx Service Guide page provided and extract EVERY SINGLE weight-based rate entry — no exceptions, no stopping early.

=== WHAT TO EXTRACT ===
Extract rate entries where a shipment weight maps to a dollar price for a specific service.
Examples of what to extract:
- "1 lb | FedEx Priority Overnight | Zone 2 | $34.71"
- "151–499 lbs | FedEx First Overnight Freight | Zone 5 | $11.73/lb" (price_type=per_pound)
- "Minimum charge | FedEx Ground | Zone 3 | $226.00" (price_type=minimum_charge)

=== WHAT TO SKIP ===
Return skipped=true (empty rates list) for pages that contain:
- Zone charts / zone lookup tables (destination country → zone number)
- Fee description pages (address correction, dangerous goods, fuel surcharge, etc.)
- Terms and Conditions text
- Cover pages, table of contents, instructional text
- Service applicability matrices (checkmark grids)

=== CRITICAL: COMPLETENESS REQUIREMENT ===
You MUST extract EVERY row listed in the page — do not stop partway through a weight sequence.

If the page lists weights 1 through 150 lbs, you must produce entries for ALL 150 weights.
If there are 6 services per weight, that is 150 × 6 = 900 RateEntry objects — produce all 900.

**Before finalizing your output**, count the distinct weights visible in the page content.
Then verify your output contains (weight_count × service_count) entries for each package type.
If your count is lower, you have missed rows — go back and extract the missing ones.

Common failure modes to avoid:
- Stopping after the first table section and ignoring the plain text section (or vice versa)
- Extracting only the first few weights (e.g., 1–3 lbs) and skipping the rest (4–150 lbs)
- Skipping package type variants (e.g., extracting "Other packaging" rows but missing "FedEx Envelope" rows)
- Treating the page as done after the first visual table block when more weight rows follow

=== FIELD RULES ===
table_type: Choose the most specific matching type:
  - "us_package"          → US domestic rates by zone (2–8), services like FedEx Priority Overnight
  - "us_express_freight"  → US domestic freight rates (per pound, zones 2–16), services like FedEx First Overnight Freight
  - "us_multiweight"      → Express Multiweight per-pound bulk rates (100 lbs+)
  - "sameday"             → FedEx SameDay / Next Flight Out (no zone)
  - "intl_package_export" → International package rates from US (zones A–O, Puerto Rico, etc.)
  - "intl_package_import" → International package rates to US (zones A–O)
  - "intl_premium"        → FedEx International Premium freight
  - "ground_domestic"     → FedEx Ground / Home Delivery (zones 2–8)
  - "ground_ak_hi"        → FedEx Ground Alaska or Hawaii (zones 17,22,23,25,9,14,92,96)
  - "ground_canada"       → FedEx International Ground Canada (zones 51,54)

zone: REQUIRED for all table_type values except "sameday". Use the zone label as printed.

direction: "domestic" for US-to-US; "us_export" for US-to-international; "us_import" for international-to-US.

price_type:
  - "flat"           → dollar amount is the total per-shipment charge (most common)
  - "per_pound"      → multiply the dollar amount by total shipment weight to get the charge
  - "minimum_charge" → this is the floor charge for this service/zone combination

price_usd: Parse dollar values like "$1,038.28" → 1038.28. Strip $ and commas.
  - If the cell shows "*"  → set price_usd=null, price_note="Based on package weight rate"
  - If the cell shows "**" → set price_usd=null, price_note="One-pound rate applies"

weight_lbs: Exact weight for flat-rate rows. Use weight_range_min_lbs / weight_range_max_lbs for ranges.

package_type: Include when the rate differs by packaging (FedEx Envelope, FedEx Pak, FedEx 10kg Box, etc.)
  For rows that apply to "all other packaging" or unspecified packaging, set package_type="Other packaging".

=== ARTIFACT HANDLING ===
The text "[LABEL: Shipments in all other packaging / maximum weight in lbs.]" is a label artifact
from rotated text in the PDF. Treat it as a column/row label, not as a rate value.

=== OUTPUT GRANULARITY ===
Produce ONE RateEntry per (weight_or_range, service_name, package_type) combination.
If a page has 49 weights × 6 services × 2 package types = 588 combinations, return 588 RateEntry objects.
Do NOT aggregate or summarise — extract every individual price point.

=== CONFIDENCE ===
Set confidence="high" when table structure is clear and values are unambiguous.
Set confidence="medium" for minor alignment issues or inferred headers.
Set confidence="low" when table structure is garbled or values are uncertain.
"""

CLASSIFICATION_PROMPT = """\
You are a FedEx Service Guide analyst. Your task is to classify a single PDF page
and identify the structure of any rate table it contains.

You do NOT need to extract individual rate values — a separate program handles that.
You only need to answer: what kind of table is on this page, and what are the column headers?

=== WHAT TO RETURN ===

skipped=true for pages with NO weight-based rate tables:
  - Zone charts / zone lookup tables (country → zone number)
  - Fee description pages (address correction, fuel surcharge, etc.)
  - Terms and Conditions, cover pages, table of contents, instructional text
  - Service applicability matrices (checkmark grids)

For pages WITH a rate table, return:
  - table_type: the best matching type from the allowed values
  - direction: "domestic" / "us_export" / "us_import"
  - zone: zone identifier as printed (e.g. "2", "A", "17"). NULL only for sameday.
  - services: list of service columns in left-to-right order, with exact service names and delivery commitments
  - has_envelope_row: true if FedEx Envelope appears as a separate row
  - has_pak_row: true if FedEx Pak appears as a separate row

=== TABLE TYPE VALUES ===
  "us_package"          → US domestic package rates by zone (2–8)
  "us_express_freight"  → US domestic freight rates per pound (zones 2–16)
  "us_multiweight"      → Express Multiweight per-pound bulk rates
  "sameday"             → FedEx SameDay / Next Flight Out (no zone)
  "intl_package_export" → International package rates from US (zones A–O, Puerto Rico, etc.)
  "intl_package_import" → International package rates to US (zones A–O)
  "intl_premium"        → FedEx International Premium freight
  "ground_domestic"     → FedEx Ground / Home Delivery (zones 2–8)
  "ground_ak_hi"        → FedEx Ground Alaska or Hawaii
  "ground_canada"       → FedEx International Ground Canada (zones 51, 54)

=== SERVICES ===
List ONLY the actual service name columns — not the weight/zone column.
Use exact FedEx service names (e.g. "FedEx Priority Overnight", "FedEx 2Day A.M.").
Include the delivery_commitment text if it appears above the service name in the table.
Order them left-to-right as they appear in the table.
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


def build_chain(settings: Settings) -> Runnable:
    """
    Build the extraction chain:
        prompt messages → LLM → structured PageExtraction output

    The chain is stateless and can be shared across async tasks.
    """
    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(PageExtraction)
    return structured_llm


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

    All public methods return a PageExtraction — they never raise.
    Failures are logged and result in an empty/skipped extraction.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chain = build_chain(settings)
        self._classification_chain = build_classification_chain(settings)
        self._invoke_with_retry = self._make_retry_wrapper()

    def _make_retry_wrapper(self):  # type: ignore[return]
        """Build a tenacity-wrapped version of the raw chain invoke."""
        max_retries = self._settings.max_retries
        wait_base = self._settings.retry_wait_seconds

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=wait_base, min=wait_base, max=60),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _invoke(messages: list[Any]) -> PageExtraction:
            return self._chain.invoke(messages)

        return _invoke

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def extract(self, page_content: str, page_number: int) -> PageExtraction:
        """
        Synchronously extract rate entries from a formatted page string.
        Returns an empty PageExtraction (skipped=True) on any unrecoverable error.
        """
        messages = _build_messages(page_content)
        try:
            result = self._invoke_with_retry(messages)
            logger.info(
                "Extraction complete",
                extra={
                    "page": page_number,
                    "rows": len(result.rates),
                    "skipped": result.skipped,
                },
            )
            return result
        except ValidationError as exc:
            logger.error(
                "Structured output validation failed",
                extra={"page": page_number, "error": str(exc)},
            )
        except Exception as exc:
            logger.error(
                "LLM extraction failed after retries",
                extra={"page": page_number, "error": str(exc)},
            )
        return PageExtraction(
            rates=[],
            skipped=True,
            skip_reason="Extraction failed — see logs for details",
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aextract(self, page_content: str, page_number: int) -> PageExtraction:
        """
        Asynchronously extract rate entries from a formatted page string.
        Returns an empty PageExtraction (skipped=True) on any unrecoverable error.
        """
        messages = _build_messages(page_content)
        try:
            result = await self._aextract_with_retry(messages, page_number)
            logger.info(
                "Extraction complete",
                extra={
                    "page": page_number,
                    "rows": len(result.rates),
                    "skipped": result.skipped,
                },
            )
            return result
        except ValidationError as exc:
            logger.error(
                "Structured output validation failed",
                extra={"page": page_number, "error": str(exc)},
            )
        except Exception as exc:
            logger.error(
                "LLM extraction failed after retries",
                extra={"page": page_number, "error": str(exc)},
            )
        return PageExtraction(
            rates=[],
            skipped=True,
            skip_reason="Extraction failed — see logs for details",
        )

    async def _aextract_with_retry(
        self, messages: list[Any], page_number: int
    ) -> PageExtraction:
        """Async chain invoke with manual retry (tenacity doesn't wrap async natively)."""
        max_retries = self._settings.max_retries
        wait_base = self._settings.retry_wait_seconds
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return await self._chain.ainvoke(messages)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    import asyncio
                    wait = wait_base * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed, retrying",
                        extra={
                            "page": page_number,
                            "attempt": attempt,
                            "wait": wait,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Hybrid classification interface (lightweight — LLM returns metadata only)
    # ------------------------------------------------------------------

    def classify_page(self, page_content: str, page_number: int) -> "PageClassification":
        """
        Synchronously classify a page (table_type, zone, services).
        Returns a skipped PageClassification on any error.
        """
        messages = _build_classification_messages(page_content)
        try:
            result = self._classification_chain.invoke(messages)
            logger.info(
                "Classification complete",
                extra={"page": page_number, "table_type": result.table_type, "skipped": result.skipped},
            )
            return result
        except Exception as exc:
            logger.error("Classification failed", extra={"page": page_number, "error": str(exc)})
            return PageClassification(skipped=True, skip_reason=f"Classification failed: {exc}")

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

def _build_messages(page_content: str) -> list[Any]:
    """Build the message list for the LLM call."""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=page_content),
    ]


def _build_classification_messages(page_content: str) -> list[Any]:
    """Build the message list for the classification (hybrid) call."""
    return [
        SystemMessage(content=CLASSIFICATION_PROMPT),
        HumanMessage(content=page_content),
    ]
