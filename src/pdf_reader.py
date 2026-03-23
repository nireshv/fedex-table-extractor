"""
PDF reading layer using pdfplumber.

Responsibilities:
- Open and cache the PDF handle for the process lifetime
- Extract raw text and table data for a given page number
- Return a structured PageContent dataclass for downstream formatting
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)

# Raw table row: list of cell values (str | None)
RawTable = list[list[Any]]


@dataclass
class PageContent:
    """Raw extraction output for a single PDF page."""

    page_number: int           # 1-indexed
    text: str                  # full page text as extracted by pdfplumber
    tables: list[RawTable]     # list of tables; each table is list-of-rows
    is_empty: bool = False     # True when pdfplumber found no content at all


class PDFReader:
    """
    Wraps pdfplumber for the lifetime of the extraction run.

    Usage:
        reader = PDFReader("table_full.pdf")
        content = reader.read_page(13)
        reader.close()

    Or as a context manager:
        with PDFReader("table_full.pdf") as reader:
            content = reader.read_page(13)
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self._path = Path(pdf_path)
        if not self._path.exists():
            raise FileNotFoundError(f"PDF not found: {self._path}")
        self._pdf = pdfplumber.open(self._path)
        logger.info("Opened PDF", extra={"path": str(self._path), "pages": self.total_pages})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_pages(self) -> int:
        return len(self._pdf.pages)

    def read_page(self, page_number: int) -> PageContent:
        """
        Extract content from a 1-indexed page.

        Returns PageContent with text and tables.
        Never raises — returns PageContent(is_empty=True) on any error.
        """
        if page_number < 1 or page_number > self.total_pages:
            raise ValueError(
                f"page_number {page_number} out of range 1–{self.total_pages}"
            )

        page = self._pdf.pages[page_number - 1]

        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Failed to extract text from page",
                extra={"page": page_number, "error": str(exc)},
            )
            text = ""

        try:
            raw_tables = page.extract_tables() or []
        except Exception as exc:
            logger.warning(
                "Failed to extract tables from page",
                extra={"page": page_number, "error": str(exc)},
            )
            raw_tables = []

        # Filter out the persistent navigation bar table that appears on every page.
        # It always has exactly 1 row × 3 cols: ['CONTENTS', 'RATES', 'TERMS']
        tables = [t for t in raw_tables if not _is_nav_table(t)]

        is_empty = not text.strip() and not tables

        if is_empty:
            logger.debug("Page appears empty", extra={"page": page_number})

        return PageContent(
            page_number=page_number,
            text=text,
            tables=tables,
            is_empty=is_empty,
        )

    def close(self) -> None:
        try:
            self._pdf.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PDFReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nav_table(table: RawTable) -> bool:
    """
    Detect the navigation bar that pdfplumber picks up on every page.
    It is always: [['CONTENTS', 'RATES', 'TERMS']]
    """
    if not table or len(table) != 1:
        return False
    row = table[0]
    if len(row) != 3:
        return False
    cells = [str(c).strip().upper() if c else "" for c in row]
    return cells == ["CONTENTS", "RATES", "TERMS"]
