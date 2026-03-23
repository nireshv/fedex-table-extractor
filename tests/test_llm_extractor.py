"""
Tests for LLMExtractor.

All LLM calls are mocked — no real API keys are required.
Tests verify:
- Correct chain construction
- Retry behavior on transient failures
- Error isolation (always returns PageExtraction, never raises)
- Async interface works
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import PageExtraction, RateEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_extraction() -> PageExtraction:
    return PageExtraction(
        rates=[
            RateEntry(
                source_page=13,
                table_type="us_package",
                service_category="US Package",
                service_name="FedEx Priority Overnight",
                direction="domestic",
                zone="2",
                weight_lbs=1.0,
                price_usd=34.71,
                price_type="flat",
            )
        ]
    )


def _make_settings(provider: str = "anthropic"):
    """Build a Settings object for testing without real API keys."""
    from src.config import Settings
    if provider == "anthropic":
        return Settings(
            llm_provider="anthropic",
            anthropic_api_key="test-key-not-real",
            pdf_path="table_full.pdf",
            db_path=":memory:",
            max_retries=2,
            retry_wait_seconds=0.01,
        )
    return Settings(
        llm_provider="openai",
        openai_api_key="test-key-not-real",
        pdf_path="table_full.pdf",
        db_path=":memory:",
        max_retries=2,
        retry_wait_seconds=0.01,
    )


def _make_extractor_with_mock_chain(mock_chain: MagicMock) -> "LLMExtractor":
    """Build an LLMExtractor whose internal chain is replaced by a mock."""
    from src.llm_extractor import LLMExtractor
    settings = _make_settings()
    with patch("src.llm_extractor.build_chain", return_value=mock_chain):
        extractor = LLMExtractor(settings)
    return extractor


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------

class TestLLMExtractorSync:
    def test_extract_returns_page_extraction(self) -> None:
        expected = _make_valid_extraction()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected
        mock_chain.ainvoke = AsyncMock(return_value=expected)
        mock_chain.with_structured_output = MagicMock(return_value=mock_chain)

        extractor = _make_extractor_with_mock_chain(mock_chain)
        # Bypass tenacity retry wrapper
        extractor._invoke_with_retry = mock_chain.invoke

        result = extractor.extract("page content", page_number=13)
        assert isinstance(result, PageExtraction)
        assert len(result.rates) == 1

    def test_extract_returns_skipped_on_api_error(self) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("API down")
        mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
        mock_chain.with_structured_output = MagicMock(return_value=mock_chain)

        extractor = _make_extractor_with_mock_chain(mock_chain)
        extractor._invoke_with_retry = mock_chain.invoke

        result = extractor.extract("page content", page_number=99)
        assert isinstance(result, PageExtraction)
        assert result.skipped is True
        assert "failed" in (result.skip_reason or "").lower()


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestLLMExtractorAsync:
    @pytest.mark.asyncio
    async def test_aextract_returns_page_extraction(self) -> None:
        expected = _make_valid_extraction()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=expected)
        mock_chain.with_structured_output = MagicMock(return_value=mock_chain)

        extractor = _make_extractor_with_mock_chain(mock_chain)
        result = await extractor.aextract("page content", page_number=13)

        assert isinstance(result, PageExtraction)
        assert len(result.rates) == 1

    @pytest.mark.asyncio
    async def test_aextract_retries_on_transient_error(self) -> None:
        from src.config import Settings

        expected = _make_valid_extraction()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            side_effect=[RuntimeError("timeout"), RuntimeError("timeout"), expected]
        )
        mock_chain.with_structured_output = MagicMock(return_value=mock_chain)

        # Need max_retries=3 to allow: fail, fail, succeed
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="test-key",
            max_retries=3,
            retry_wait_seconds=0.01,
        )
        with patch("src.llm_extractor.build_chain", return_value=mock_chain):
            from src.llm_extractor import LLMExtractor
            extractor = LLMExtractor(settings)

        result = await extractor.aextract("page content", page_number=13)

        assert isinstance(result, PageExtraction)
        assert mock_chain.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_aextract_returns_skipped_after_max_retries(self) -> None:
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=RuntimeError("persistent failure"))
        mock_chain.with_structured_output = MagicMock(return_value=mock_chain)

        extractor = _make_extractor_with_mock_chain(mock_chain)
        result = await extractor.aextract("page content", page_number=99)

        assert result.skipped is True
        assert "failed" in (result.skip_reason or "").lower()
        # max_retries=2 in test settings
        assert mock_chain.ainvoke.call_count == 2


# ---------------------------------------------------------------------------
# get_llm tests
# ---------------------------------------------------------------------------

class TestGetLLM:
    def test_anthropic_provider(self) -> None:
        from src.llm_extractor import get_llm
        settings = _make_settings("anthropic")
        with patch("langchain_anthropic.ChatAnthropic") as MockChat:
            MockChat.return_value = MagicMock()
            # Import inside the function to pick up the patch
            import langchain_anthropic
            with patch.object(langchain_anthropic, "ChatAnthropic", MockChat):
                llm = get_llm(settings)
        # Just verify no exception was raised and we got something back
        assert llm is not None

    def test_openai_provider(self) -> None:
        from src.llm_extractor import get_llm
        settings = _make_settings("openai")
        with patch("langchain_openai.ChatOpenAI") as MockChat:
            MockChat.return_value = MagicMock()
            import langchain_openai
            with patch.object(langchain_openai, "ChatOpenAI", MockChat):
                llm = get_llm(settings)
        assert llm is not None

    def test_missing_anthropic_key_raises(self) -> None:
        from src.llm_extractor import get_llm
        from src.config import Settings
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key=None,
        )
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_llm(settings)

    def test_missing_openai_key_raises(self) -> None:
        from src.llm_extractor import get_llm
        from src.config import Settings
        settings = Settings(
            llm_provider="openai",
            openai_api_key=None,
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm(settings)

    def test_unknown_provider_raises(self) -> None:
        from src.llm_extractor import get_llm
        from src.config import Settings
        settings = Settings(
            llm_provider="anthropic",
            anthropic_api_key="test",
        )
        settings.llm_provider = "unknown"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(settings)
