"""
Tests for PDFReader and PageFormatter.

These tests use the real PDF file (table_full.pdf) so they serve as
integration tests for the PDF extraction layer. No LLM is called.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.page_formatter import MAX_CHARS, _clean_text, format_page
from src.pdf_reader import PDFReader, _is_nav_table

PDF_PATH = Path(__file__).parent.parent / "table_full.pdf"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reader() -> PDFReader:
    if not PDF_PATH.exists():
        pytest.skip(f"PDF not found at {PDF_PATH}")
    r = PDFReader(PDF_PATH)
    yield r
    r.close()


# ---------------------------------------------------------------------------
# PDFReader tests
# ---------------------------------------------------------------------------

class TestPDFReader:
    def test_opens_pdf(self, reader: PDFReader) -> None:
        assert reader.total_pages == 186

    def test_read_page_returns_content(self, reader: PDFReader) -> None:
        content = reader.read_page(13)
        assert content.page_number == 13
        assert isinstance(content.text, str)
        assert isinstance(content.tables, list)
        assert not content.is_empty

    def test_rate_page_has_text(self, reader: PDFReader) -> None:
        content = reader.read_page(13)
        # Page 13 is "U.S. package rates: Zone 2"
        assert "Zone 2" in content.text

    def test_nav_table_is_filtered(self, reader: PDFReader) -> None:
        """The CONTENTS | RATES | TERMS nav bar should not appear in content.tables."""
        for page_num in [13, 50, 118]:
            content = reader.read_page(page_num)
            for table in content.tables:
                assert not _is_nav_table(table), (
                    f"Nav table leaked through on page {page_num}: {table}"
                )

    def test_rate_page_has_tables(self, reader: PDFReader) -> None:
        content = reader.read_page(13)
        assert len(content.tables) >= 1

    def test_out_of_range_page_raises(self, reader: PDFReader) -> None:
        with pytest.raises(ValueError, match="out of range"):
            reader.read_page(0)
        with pytest.raises(ValueError, match="out of range"):
            reader.read_page(reader.total_pages + 1)

    def test_empty_or_non_rate_page(self, reader: PDFReader) -> None:
        # Page 1 is the cover — pdfplumber may return some text but it's not a rate table
        content = reader.read_page(1)
        assert content.page_number == 1

    def test_context_manager(self) -> None:
        if not PDF_PATH.exists():
            pytest.skip("PDF not found")
        with PDFReader(PDF_PATH) as r:
            assert r.total_pages > 0


class TestNavTableDetection:
    def test_detects_nav_table(self) -> None:
        assert _is_nav_table([["CONTENTS", "RATES", "TERMS"]]) is True

    def test_ignores_real_table(self) -> None:
        assert _is_nav_table([["Zone", "Price", "Service"]]) is False

    def test_ignores_empty(self) -> None:
        assert _is_nav_table([]) is False

    def test_ignores_multi_row(self) -> None:
        assert _is_nav_table([["CONTENTS", "RATES", "TERMS"], ["a", "b", "c"]]) is False


# ---------------------------------------------------------------------------
# PageFormatter tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_nav_bar(self) -> None:
        text = "CONTENTS RATES TERMS\nSome rate content"
        cleaned = _clean_text(text)
        assert "CONTENTS" not in cleaned
        assert "Some rate content" in cleaned

    def test_fixes_rotated_label(self) -> None:
        garbled = ".sbl ni thgiew mumixam / gnigakcap rehto lla ni stnempihS"
        cleaned = _clean_text(garbled)
        assert "[LABEL:" in cleaned
        assert "maximum weight in lbs" in cleaned.lower()

    def test_fixes_rotated_service_name(self) -> None:
        text = "®thginrevO ytiroirP xEdeF"
        cleaned = _clean_text(text)
        assert "FedEx Priority Overnight" in cleaned

    def test_collapses_blank_lines(self) -> None:
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = _clean_text(text)
        assert "\n\n\n" not in cleaned


class TestFormatPage:
    def test_format_real_page(self, reader: PDFReader) -> None:
        content = reader.read_page(13)
        formatted = format_page(content)
        assert f"=== PAGE 13 ===" in formatted
        assert len(formatted) > 100

    def test_format_includes_table_marker(self, reader: PDFReader) -> None:
        content = reader.read_page(13)
        formatted = format_page(content)
        assert "--- TABLE" in formatted

    def test_token_budget_guard(self, reader: PDFReader) -> None:
        """If formatted output is too long, it should be truncated."""
        content = reader.read_page(13)
        # Artificially inflate content
        content.text = content.text * 50
        formatted = format_page(content)
        assert len(formatted) <= MAX_CHARS + 100  # small slack for truncation message

    def test_empty_page(self) -> None:
        from src.pdf_reader import PageContent
        empty = PageContent(page_number=99, text="", tables=[], is_empty=True)
        formatted = format_page(empty)
        assert "=== PAGE 99 ===" in formatted
