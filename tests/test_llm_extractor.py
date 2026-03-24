"""
Tests for LLMExtractor.

All LLM calls are mocked — no real API keys are required.
Tests verify:
- Correct LLM construction for each provider
- Error handling for missing keys and unknown providers
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
