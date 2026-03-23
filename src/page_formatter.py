"""
Converts a PageContent dataclass into a clean, LLM-readable string.

Key responsibilities:
- Remove known PDF artifacts (rotated text, navigation labels)
- Format raw pdfplumber tables as readable ASCII grids
- Apply a token-budget guard to avoid exceeding the LLM context window
- Produce a deterministic, reproducible string for the same page
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.pdf_reader import PageContent, RawTable

logger = logging.getLogger(__name__)

# Approximate character budget before we start truncating table content.
# ~8 000 tokens × 4 chars/token = 32 000 chars. We use 28 000 to leave
# room for the system prompt and structured-output schema overhead.
MAX_CHARS = 28_000

# ---------------------------------------------------------------------------
# Known rotated-text artifacts produced by pdfplumber when the PDF has
# text rendered at 90° or 270°. These appear as garbled reversed strings.
# Each tuple is (garbled_pattern, replacement).
# ---------------------------------------------------------------------------
_ROTATED_TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    # "Shipments in all other packaging / maximum weight in lbs."
    (
        r"\.sbl\s+ni\s+thgiew\s+mumixam\s*/\s*gnigakcap\s+rehto\s+lla\s+ni\s+stnempihS",
        "[LABEL: Shipments in all other packaging / maximum weight in lbs.]",
    ),
    # Reversed service names in the column-header rotation artifacts
    # (e.g. "®thginrevO tsriF xEdeF" → readable name)
    (r"®thginrevO\s+tsriF\s+xEdeF", "FedEx First Overnight®"),
    (r"®thginrevO\s+ytiroirP\s+xEdeF", "FedEx Priority Overnight®"),
    (r"®thginrevO\s+dradnatS\s+xEdeF", "FedEx Standard Overnight®"),
    (r"\.M\.A\s+®yaD2\s+xEdeF", "FedEx 2Day® A.M."),
    (r"®yaD2\s+xEdeF", "FedEx 2Day®"),
    (r"®revaS\s+sserpxE\s+xEdeF", "FedEx Express Saver®"),
    (r"®dnuorG\s+xEdeF", "FedEx Ground®"),
    (r"®yrevileD\s+emoH\s+xEdeF", "FedEx Home Delivery®"),
    (r"®tsriF\s+lanoitanretnI\s+xEdeF", "FedEx International First®"),
    (r"sserpxE\s+®ytiroirP\s+lanoitanretnI\s+xEdeF", "FedEx International Priority® Express"),
    (r"®ytiroirP\s+lanoitanretnI\s+xEdeF", "FedEx International Priority®"),
    (r"®ymonocE\s+lanoitanretnI\s+xEdeF", "FedEx International Economy®"),
    (r"sulP\s+tcennoC\s+lanoitanretnI\s+®xEdeF", "FedEx® International Connect Plus"),
    (r"®dnuorG\s+lanoitanretnI\s+xEdeF", "FedEx International Ground®"),
    # Freight services
    (r"thgierF\s+®thginrevO\s+tsriF\s+xEdeF", "FedEx First Overnight® Freight"),
    (r"thgierF\s+®yaD1\s+xEdeF", "FedEx 1Day® Freight"),
    (r"thgierF\s+®yaD2\s+xEdeF", "FedEx 2Day® Freight"),
    (r"thgierF\s+®yaD3\s+xEdeF", "FedEx 3Day® Freight"),
    (r"thgierF\s+®ytiroirP\s+lanoitanretnI\s+xEdeF", "FedEx International Priority® Freight"),
    (r"thgierF\s+®ymonocE\s+lanoitanretnI\s+xEdeF", "FedEx International Economy® Freight"),
    (r"thgierF\s+derrefeD\s+lanoitanretnI\s+®xEdeF", "FedEx® International Deferred Freight"),
    (r"®muimerP\s+lanoitanretnI\s+xEdeF", "FedEx International Premium®"),
    (r"noitamrofni\s+eroM", "More information"),
]

# Navigation bar text that appears at top of every page — strip it
_NAV_BAR_RE = re.compile(r"^CONTENTS\s+RATES\s+TERMS\s*\n?", re.MULTILINE)


def format_page(content: PageContent) -> str:
    """
    Return a clean, LLM-readable string for the given page content.

    Structure:
        === PAGE {n} ===
        [TEXT CONTENT]
        --- TABLE 1 ---
        [grid rows]
        --- TABLE 2 ---
        [grid rows]
    """
    parts: list[str] = [f"=== PAGE {content.page_number} ===\n"]

    # Clean page text
    cleaned_text = _clean_text(content.text)
    if cleaned_text:
        parts.append(cleaned_text)
        parts.append("\n")

    # Format tables
    for idx, table in enumerate(content.tables, start=1):
        formatted = _format_table(table)
        if formatted:
            parts.append(f"\n--- TABLE {idx} ---\n")
            parts.append(formatted)
            parts.append("\n")

    result = "".join(parts)

    # Token budget guard
    if len(result) > MAX_CHARS:
        logger.warning(
            "Page content exceeds character budget, truncating",
            extra={
                "page": content.page_number,
                "chars": len(result),
                "budget": MAX_CHARS,
            },
        )
        result = result[:MAX_CHARS] + "\n[... content truncated due to length ...]"

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Apply artifact removal and whitespace normalisation to raw page text."""
    if not text:
        return ""

    # Remove navigation bar
    text = _NAV_BAR_RE.sub("", text)

    # Apply known rotated-text replacements
    for pattern, replacement in _ROTATED_TEXT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Collapse excessive blank lines (>2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _format_table(table: RawTable) -> str:
    """
    Render a pdfplumber table (list of rows) as a plain-text grid.

    Cells are separated by ' | ' and rows by newlines.
    None cells are rendered as empty strings.
    Multi-line cell values have their newlines replaced with ' / '.
    """
    if not table:
        return ""

    rows: list[list[str]] = []
    for raw_row in table:
        row: list[str] = []
        for cell in raw_row:
            if cell is None:
                row.append("")
            else:
                # Replace intra-cell newlines with ' / ' for readability
                cell_str = str(cell).replace("\n", " / ")
                # Apply rotated-text fixes inside cell values too
                for pattern, replacement in _ROTATED_TEXT_REPLACEMENTS:
                    cell_str = re.sub(pattern, replacement, cell_str, flags=re.IGNORECASE)
                row.append(cell_str.strip())
        rows.append(row)

    if not rows:
        return ""

    # Compute column widths
    num_cols = max(len(r) for r in rows)
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    # Render
    lines: list[str] = []
    for row in rows:
        padded = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            padded.append(cell.ljust(col_widths[i]))
        lines.append(" | ".join(padded).rstrip())

    return "\n".join(lines)
