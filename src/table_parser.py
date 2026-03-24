"""
Programmatic parser for pdfplumber table data.

Used by the hybrid extraction pipeline:
  1. LLM classifies the page (table_type, zone, services)
  2. This module parses the actual rate values from pdfplumber's table matrix

Handles the multi-value cell format used in FedEx Service Guide tables, where
a single pdfplumber cell contains multiple values separated by newlines, e.g.:
  weights: "1lb.\n2 lbs.\n3\n4\n5"
  prices:  "$ 73.31\n73.87\n77.77\n82.02\n82.45"
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.models import PageClassification, RateEntry, TableType

logger = logging.getLogger(__name__)

def _clean_cell(value: object) -> str:
    """Return a stripped string from a cell value, empty string if None."""
    if value is None:
        return ""
    return str(value).strip()


def _parse_price(s: str) -> Optional[float]:
    """Parse a price string like '$ 73.31' or '73.87' into float. Returns None for * / **."""
    s = s.strip().lstrip("$").replace(",", "").strip()
    if s in ("", "*", "**", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_price_note(s: str) -> Optional[str]:
    """Return a price_note if the price cell is a footnote marker."""
    s = s.strip()
    if s == "*":
        return "Based on package weight rate"
    if s == "**":
        return "One-pound rate applies"
    return None


def _parse_weight(s: str) -> Optional[float]:
    """
    Parse a weight string into a float in lbs.
    Handles: '1lb.', '2 lbs.', '3', '8 oz.', 'up to 8 oz.'
    Returns None for non-numeric strings.
    """
    s = s.strip().lower()
    # ounces
    if "oz" in s:
        nums = [n for n in re.findall(r"[\d.]+", s) if re.search(r"\d", n)]
        if nums:
            try:
                return float(nums[-1]) / 16.0  # convert oz to lbs for storage
            except ValueError:
                pass
    # pounds
    s = re.sub(r"lbs?\.?", "", s).strip()
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_oz_weight(s: str) -> bool:
    return "oz" in s.lower()


def _normalize(text: str) -> str:
    """Normalize for matching: lowercase, strip ®/™, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[®™]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_service_columns(
    table: list[list],
    classification: PageClassification,
) -> dict[int, str]:
    """
    Map column index → service_name using the LLM classification as a guide.

    Scans the first few rows of the table for service name text and matches
    them (case-insensitive, whitespace-normalized) to the names from the classification.

    Returns {col_index: service_name}.
    """
    service_names = [s.service_name for s in classification.services]
    if not service_names:
        return {}

    col_to_service: dict[int, str] = {}
    n_cols = max(len(row) for row in table[:4]) if table else 0

    # Build per-column text by concatenating header rows
    col_texts: dict[int, str] = {i: "" for i in range(n_cols)}
    for row in table[:4]:
        for col_idx, cell in enumerate(row):
            col_texts[col_idx] = col_texts.get(col_idx, "") + " " + _normalize(_clean_cell(cell))

    # Sort service names by descending length so more specific names (e.g. "FedEx 2Day A.M.")
    # are tried before shorter prefixes (e.g. "FedEx 2Day") that are substrings of them.
    service_names_by_specificity = sorted(service_names, key=len, reverse=True)

    for col_idx, col_text in col_texts.items():
        # Strip "fedex" prefix to focus on the distinctive part of the name
        col_key = col_text.replace("fedex", "").strip()
        for svc in service_names_by_specificity:
            if col_idx in col_to_service:
                break
            svc_norm = _normalize(svc).replace("fedex", "").strip()
            # Match if the column header contains the normalized service key
            if svc_norm and svc_norm in col_key:
                col_to_service[col_idx] = svc
            elif svc_norm and col_key and col_key in svc_norm and len(col_key) > 4:
                col_to_service[col_idx] = svc

    if len(col_to_service) < len(service_names):
        logger.debug(
            "Column matching incomplete: found %d of %d services",
            len(col_to_service),
            len(service_names),
        )

    return col_to_service


def _find_weight_column(table: list[list]) -> Optional[int]:
    """
    Find the column index that contains the weight values for data rows.

    Two patterns exist in FedEx rate tables:
    - US package tables: col 0 = package-type label, col 1 = weights (multi-value cells)
    - Multiweight/freight tables: col 0 = weight ranges ("100–499 lbs."), col 1 = empty

    Returns None if no weight column is found (e.g. a secondary table that shares
    weights with a sibling table on the same page).
    """
    # Check col 0 first: if it has weight ranges (contains "–" or "-" with digits), use it
    for row in table[1:4]:
        col0 = _clean_cell(row[0] if row else None)
        if re.search(r"\d+\s*[–\-]\s*\d+\s*(lbs?|oz)", col0, re.IGNORECASE):
            return 0

    # Look for a column with "lbs" or "oz" keyword in data rows (strong signal).
    # Exclude "up to X oz." cells — those are package-type labels (FedEx Envelope),
    # not weight data rows.
    for row in table[3:7]:
        for col_idx in range(min(3, len(row))):
            text = _clean_cell(row[col_idx])
            if re.search(r"lbs?\.?\b|oz\.?\b", text, re.IGNORECASE) and "up to" not in text.lower():
                return col_idx

    # Look for col with multi-value weight-like cells (not price-like)
    for row in table[3:7]:
        for col_idx, cell in enumerate(row):
            text = _clean_cell(cell)
            if not text:
                continue
            parts = [p.strip() for p in text.split("\n") if p.strip()]
            numeric_parts = [p for p in parts if re.search(r"\d", p)]
            if len(numeric_parts) >= 2:
                # Distinguish weights from prices:
                # - Prices have "$", comma-formatted decimals, or cent decimals (123.45)
                # - Weights are integers: "101", "102", "1lb.", "2 lbs."
                has_dollar = any("$" in p for p in parts)
                has_price_format = any(re.search(r"\d,\d{3}", p) for p in parts)
                has_decimal_values = any(re.search(r"\d+\.\d{2}", p) for p in numeric_parts)
                if not has_dollar and not has_price_format and not has_decimal_values:
                    return col_idx

    # No weight column found
    return None


def _extract_table_zone(table: list[list]) -> Optional[str]:
    """
    Extract zone identifier from table header rows.

    Looks for a standalone zone number/letter in the first few rows,
    contextually near a "Zone" keyword or region header.
    """
    for row_idx, row in enumerate(table[:3]):
        for col_idx, cell in enumerate(row):
            text = _clean_cell(cell)
            # Match standalone zone identifiers: 1-2 digit numbers or single letters A-O
            if re.match(r"^\d{1,2}$", text) or re.match(r"^[A-O]$", text):
                # Verify context: "zone" or region header in same/adjacent rows
                for r in table[max(0, row_idx - 1) : row_idx + 1]:
                    row_text = " ".join(_clean_cell(c) for c in r).lower()
                    if "zone" in row_text or "to " in row_text or "from " in row_text:
                        return text
    return None


def _extract_table_region(table: list[list]) -> Optional[str]:
    """Extract destination region description from table header rows."""
    for row in table[:2]:
        for cell in row:
            text = _clean_cell(cell)
            if not text or len(text) < 5:
                continue
            text_lower = text.lower()
            if any(
                kw in text_lower
                for kw in ("to ", "from ", "hawaii", "alaska", "puerto", "canada", "mexico")
            ):
                return text
    return None


def _collect_weight_strings(table: list[list], weight_col: int) -> list[list[str]]:
    """
    Collect weight strings from data rows of a table, grouped by row.

    Returns a list of lists: each inner list contains the weight strings
    from one data row's multi-value cell (e.g. ["101 lbs.", "102", "103", "104", "105"]).
    """
    data_rows = _find_data_rows(table, weight_col)
    result = []
    for row_idx in data_rows:
        weight_cell = _clean_cell(table[row_idx][weight_col] if weight_col < len(table[row_idx]) else None)
        if not weight_cell:
            continue
        weight_strings = [w.strip() for w in weight_cell.split("\n") if w.strip()]
        if weight_strings:
            result.append(weight_strings)
    return result


def _find_zone_columns(table: list[list]) -> dict[int, str]:
    """
    Detect and return zone-column mapping when table columns represent zones.

    Used for us_express_freight / us_multiweight tables (numeric zones 2, 3, 4, 9–10, …)
    and intl_package_export box-rate tables (letter zones A–O).

    Returns {col_idx: zone_label} or empty dict if not a zone-column table.
    """
    col_to_zone: dict[int, str] = {}
    n_cols = max(len(row) for row in table[:3]) if table else 0
    for col_idx in range(n_cols):
        for row in table[:3]:
            if col_idx >= len(row):
                continue
            cell = _clean_cell(row[col_idx])
            # Numeric zone: 1-2 digit number or range like "9–10" / "13–16"
            # Letter zone: single uppercase letter A–O (international zones)
            if (
                re.match(r"^\d{1,2}([\u2013\-]\d{1,3})?$", cell)
                or re.match(r"^[A-O]$", cell)
            ):
                col_to_zone[col_idx] = cell
                break
    return col_to_zone


def _parse_box_rate_zone_col_table(
    table: list[list],
    zone_col_map: dict[int, str],
    zone_cols: list[int],
    svc_info: "ServiceInfo",
    classification: "PageClassification",
    table_region: Optional[str],
    source_page: int,
) -> list["RateEntry"]:
    """
    Parse a box-rate table where letter zones (A–O) are column headers.

    Each table covers one service + one package type with two rate rows:
      Row N:   ['Base rate\\n1–22 lbs.', None, None, ...]  ← weight label; prices in next row
      Row N+1: [None, '$206.13', '$251.07', ...]            ← base rate prices per zone
      Row N+2: ['Additional\\nper-pound rate', '9.37', ...] ← per-pound rate + prices

    Returns 2 × len(zone_cols) RateEntry objects (flat base + per-pound additional).
    """
    entries: list[RateEntry] = []
    pkg_type = svc_info.package_type if svc_info else None

    base_weight_min: Optional[float] = None
    base_weight_max: Optional[float] = None
    base_prices_row: Optional[list] = None
    perpound_row: Optional[list] = None

    for row_idx, row in enumerate(table):
        col0 = _clean_cell(row[0] if row else None)
        col0_lower = col0.lower()

        if "base rate" in col0_lower:
            # Extract weight range embedded in cell (e.g. "Base rate\n1–22 lbs.")
            for part in col0.split("\n"):
                part = part.strip()
                if _is_weight_range(part):
                    base_weight_min, base_weight_max = _parse_weight_range(part)

            # Prices may be in this row or the immediately following row
            has_prices = any(
                _clean_cell(row[ci] if ci < len(row) else None)
                for ci in zone_cols
            )
            if has_prices:
                base_prices_row = row
            elif row_idx + 1 < len(table):
                next_row = table[row_idx + 1]
                next_col0 = _clean_cell(next_row[0] if next_row else None)
                if not next_col0:  # prices row has no label in col 0
                    base_prices_row = next_row

        elif "additional" in col0_lower or ("per" in col0_lower and "pound" in col0_lower):
            perpound_row = row

    def _emit(
        prices_row: list,
        price_type: str,
        weight_min: Optional[float],
        weight_max: Optional[float],
    ) -> None:
        for col_idx in zone_cols:
            price_raw = _clean_cell(prices_row[col_idx] if col_idx < len(prices_row) else None)
            price = _parse_price(price_raw)
            note = _parse_price_note(price_raw) or (None if price is not None else "Rate not specified")
            try:
                e = RateEntry(
                    source_page=source_page,
                    table_type=classification.table_type,
                    service_category=_table_type_to_category(classification.table_type),
                    service_name=svc_info.service_name,
                    delivery_commitment=svc_info.delivery_commitment,
                    direction=classification.direction or "us_export",
                    zone=zone_col_map[col_idx],
                    destination_region=table_region,
                    package_type=pkg_type,
                    weight_lbs=None,
                    weight_range_min_lbs=weight_min,
                    weight_range_max_lbs=weight_max,
                    weight_unit="lbs",
                    price_usd=price,
                    price_type=price_type,
                    price_note=note,
                )
                entries.append(e)
            except Exception as exc:
                logger.debug(
                    "Skipping box-rate entry (svc=%s zone=%s): %s",
                    svc_info.service_name, zone_col_map[col_idx], exc,
                )

    if base_prices_row is not None:
        _emit(base_prices_row, "flat", base_weight_min, base_weight_max)
    if perpound_row is not None:
        # "Additional per-pound rate" applies to weights beyond the base range.
        # If base covers 1–22 lbs, the per-pound rate starts at 23 with no upper limit.
        perpound_min = (base_weight_max + 1) if base_weight_max is not None else base_weight_min
        _emit(perpound_row, "per_pound", perpound_min, None)

    return entries


def _find_data_rows_no_weight(table: list[list]) -> list[int]:
    """Find data rows in a table that has no weight column (all cols are service/price cols)."""
    data_rows = []
    for row_idx, row in enumerate(table):
        if row_idx < 4:  # Skip header rows
            continue
        # Check if any cell has numeric/price content
        for cell in row:
            text = _clean_cell(cell)
            if text and re.search(r"\d", text) and not any(
                kw in text.lower()
                for kw in ("fedex", "zone", "delivery", "commitment", "service")
            ):
                data_rows.append(row_idx)
                break
    return data_rows


def _expand_with_external_weights(
    row: list,
    service_cols: list[int],
    weight_strings: list[str],
) -> list[dict]:
    """
    Expand a data row using externally provided weight strings
    (from a sibling table that has the weight column).

    Same output format as _expand_multivalue_cells.
    """
    # Parse prices for each service column
    price_lists: list[list[str]] = []
    for col_idx in service_cols:
        cell = _clean_cell(row[col_idx] if col_idx < len(row) else None)
        prices = [p.strip() for p in cell.split("\n") if p.strip()] if cell else []
        price_lists.append(prices)

    result = []
    for i, w_str in enumerate(weight_strings):
        if _is_weight_range(w_str):
            min_w, max_w = _parse_weight_range(w_str)
            entry = {
                "weight_lbs": None,
                "weight_range_min_lbs": min_w,
                "weight_range_max_lbs": max_w,
                "weight_unit": "lbs",
            }
        else:
            w = _parse_weight(w_str)
            if w is None:
                continue
            entry = {
                "weight_lbs": w if not _is_oz_weight(w_str) else None,
                "weight_range_min_lbs": None,
                "weight_range_max_lbs": None,
                "weight_unit": "oz" if _is_oz_weight(w_str) else "lbs",
            }

        prices = []
        for price_list in price_lists:
            p_str = price_list[i] if i < len(price_list) else ""
            prices.append((_parse_price(p_str), _parse_price_note(p_str)))
        entry["prices"] = prices
        result.append(entry)
    return result


def _find_data_rows(
    table: list[list],
    weight_col: int,
) -> list[int]:
    """Return row indices that contain main weight data (not headers, not package-type rows)."""
    data_rows = []
    for row_idx, row in enumerate(table):
        cell = _clean_cell(row[weight_col] if weight_col < len(row) else None)
        if not cell:
            continue
        # Skip known header / special-row strings
        if any(
            kw in cell.lower()
            for kw in ("fedex", "delivery", "commitment", "pak", "envelope", "up to", "zone")
        ):
            continue
        # Include "Minimum charge" rows — no digit, but a valid rate row in freight tables
        if "minimum" in cell.lower():
            data_rows.append(row_idx)
            continue
        # Must contain at least one digit
        if re.search(r"\d", cell):
            data_rows.append(row_idx)
    return data_rows


def _get_min_charge_weight_range(table: list[list], weight_col: int) -> tuple[float, float]:
    """
    Return (0.0, first_weight_min - 1) for the minimum charge weight range.

    Freight tables start at a minimum weight (usually 151 lbs); the minimum
    charge applies to shipments below that threshold, so the range is 0 to
    first_weight_min - 1.  Defaults to (0.0, 150.0) if not determinable.
    """
    for row in table:
        cell = _clean_cell(row[weight_col] if weight_col < len(row) else None)
        if not cell:
            continue
        for part in cell.split("\n"):
            part = part.strip()
            if _is_weight_range(part):
                min_w, _ = _parse_weight_range(part)
                if min_w is not None and min_w > 0:
                    return 0.0, min_w - 1.0
    return 0.0, 150.0  # FedEx freight default


def _parse_weight_range(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse a weight range string like "100–499 lbs." into (min, max) in lbs.
    "2,000+ lbs." → (2000.0, None).  Returns (None, None) if not a range.
    """
    s = re.sub(r"lbs?\.?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace(",", "")
    # Pattern: "100–499" or "100-499"
    m = re.match(r"^([\d.]+)\s*[–\-]\s*([\d.]+)$", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Pattern: "2000+" (open-ended upper)
    m = re.match(r"^([\d.]+)\s*\+$", s)
    if m:
        return float(m.group(1)), None
    return None, None


def _is_weight_range(s: str) -> bool:
    """Return True if the string looks like a weight range (contains – or + after digits)."""
    # Strip unit text first so "2,000+ lbs." matches the same as "2,000+"
    normalized = re.sub(r"lbs?\.?", "", s, flags=re.IGNORECASE).strip()
    return bool(re.search(r"\d+\s*[–\-+]\s*(\d+|$)", normalized))


def _expand_multivalue_cells(
    row: list,
    weight_col: int,
    service_cols: list[int],
) -> list[dict]:
    """
    Expand a row where each cell may contain N values separated by newlines.

    Returns a list of dicts with keys:
      weight_lbs, weight_range_min_lbs, weight_range_max_lbs, weight_unit,
      prices: [(price_usd, price_note), ...]  — parallel to service_cols
    """
    weight_cell = _clean_cell(row[weight_col] if weight_col < len(row) else None)
    if not weight_cell:
        return []

    weight_strings = [w.strip() for w in weight_cell.split("\n") if w.strip()]
    if not weight_strings:
        return []

    # Parse prices for each service column
    price_lists: list[list[str]] = []
    for col_idx in service_cols:
        cell = _clean_cell(row[col_idx] if col_idx < len(row) else None)
        prices = [p.strip() for p in cell.split("\n") if p.strip()] if cell else []
        price_lists.append(prices)

    result = []
    for i, w_str in enumerate(weight_strings):
        if _is_weight_range(w_str):
            min_w, max_w = _parse_weight_range(w_str)
            entry = {
                "weight_lbs": None,
                "weight_range_min_lbs": min_w,
                "weight_range_max_lbs": max_w,
                "weight_unit": "lbs",
            }
        else:
            w = _parse_weight(w_str)
            if w is None:
                continue
            entry = {
                "weight_lbs": w if not _is_oz_weight(w_str) else None,
                "weight_range_min_lbs": None,
                "weight_range_max_lbs": None,
                "weight_unit": "oz" if _is_oz_weight(w_str) else "lbs",
            }

        prices = []
        for price_list in price_lists:
            p_str = price_list[i] if i < len(price_list) else ""
            prices.append((_parse_price(p_str), _parse_price_note(p_str)))
        entry["prices"] = prices
        result.append(entry)
    return result


def _extract_zones_from_text(text: str) -> list[str]:
    """
    Extract zone labels in order from page text.

    Matches patterns like "Zone 2", "Zone A", "Zone 17" as they appear
    sequentially in the page text. Used when tables lack in-table zone headers.
    """
    return re.findall(r"Zone\s+(\d{1,2}|[A-O])\b", text)


def _extract_one_rate_zone_labels(page_text: str) -> list[str]:
    """
    Extract zone range labels for FedEx One Rate tables from page text.

    Handles patterns like "zone 2", "zones 3–4", "zones 5–8" (case-insensitive).
    Returns labels like ["2", "3-4", "5-8"].
    """
    raw = re.findall(r"[Zz]ones?\s+([\d–\-]+)", page_text)
    return [z.replace("–", "-") for z in raw]


def _normalize_package_type(text: str) -> str:
    """Normalize a One Rate package type label: strip footnote digits, ®/™, extra whitespace."""
    text = re.sub(r"[®™]", "", text)
    # Strip trailing footnote numbers (e.g. "Envelope6" → "Envelope", "Pak6" → "Pak")
    text = re.sub(r"(\w)\d+\s*$", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_one_rate_data_row(text: str) -> bool:
    """Return True if a cell looks like a FedEx One Rate package-type column (not weights or prices)."""
    text_lower = text.lower()
    return "envelope" in text_lower or ("pak" in text_lower and "fedex" in text_lower)


def _parse_one_rate_table(
    table: list[list],
    service_col_map: dict[int, str],
    service_cols: list[int],
    classification: "PageClassification",
    zone: Optional[str],
    region: Optional[str],
    source_page: int,
) -> list["RateEntry"]:
    """
    Parse a FedEx One Rate table where rows represent package types, not weights.

    The data row has all package type names in col 0 (newline-separated) and
    corresponding prices in the service columns (also newline-separated).
    Returns one RateEntry per (package_type, service) combination.
    """
    entries: list[RateEntry] = []

    for row in table:
        col0 = _clean_cell(row[0] if row else None)
        if not col0 or not _is_one_rate_data_row(col0):
            continue

        pkg_types = [_normalize_package_type(p) for p in col0.split("\n") if p.strip()]
        if not pkg_types:
            continue

        # Collect price lists for each service column (parallel to pkg_types)
        price_lists: list[list[str]] = []
        for col_idx in service_cols:
            cell = _clean_cell(row[col_idx] if col_idx < len(row) else None)
            prices = [p.strip() for p in cell.split("\n") if p.strip()] if cell else []
            price_lists.append(prices)

        for i, pkg_type in enumerate(pkg_types):
            for col_pos, col_idx in enumerate(service_cols):
                if col_pos >= len(price_lists):
                    break
                p_str = price_lists[col_pos][i] if i < len(price_lists[col_pos]) else ""
                price = _parse_price(p_str)
                note = _parse_price_note(p_str) or (None if price is not None else "Rate not specified")
                svc_name = service_col_map[col_idx]
                svc_info = next(
                    (s for s in classification.services if s.service_name == svc_name), None
                )
                try:
                    entry = RateEntry(
                        source_page=source_page,
                        table_type=classification.table_type,
                        service_category=_table_type_to_category(classification.table_type),
                        service_name=svc_name,
                        delivery_commitment=svc_info.delivery_commitment if svc_info else None,
                        direction=classification.direction or "domestic",
                        zone=zone,
                        destination_region=region,
                        package_type=pkg_type,
                        weight_lbs=None,
                        weight_unit="lbs",
                        price_usd=price,
                        price_type="flat",
                        price_note=note,
                    )
                    entries.append(entry)
                except Exception as exc:
                    logger.debug(
                        "Skipping one_rate entry (pkg=%s, svc=%s): %s", pkg_type, svc_name, exc
                    )

    return entries


def parse_table(
    raw_tables: list[list[list]],
    classification: PageClassification,
    source_page: int,
    page_text: str = "",
) -> list[RateEntry]:
    """
    Parse pdfplumber tables into RateEntry objects using the LLM classification.

    Args:
        raw_tables: list of tables from pdfplumber (each table is list of rows,
                    each row is list of cell values).
        classification: LLM-provided page metadata.
        source_page: 1-indexed page number.
        page_text: raw page text used to extract zone labels when tables lack
                   in-table zone headers (e.g. multi-zone pages like page 43).

    Returns:
        List of RateEntry objects. Empty list if no rate data found.
    """
    if classification.skipped or not classification.table_type:
        return []

    # Skip nav table (CONTENTS | RATES | TERMS) and empty tables
    rate_tables = [
        t for t in raw_tables
        if t and not (
            len(t) == 1
            and len(t[0]) >= 2
            and "CONTENTS" in str(t[0][0]).upper()
        )
    ]
    if not rate_tables:
        logger.warning("No rate tables found on page %d after filtering nav table", source_page)
        return []

    entries: list[RateEntry] = []
    # Determine price_type: per_pound for multiweight/freight, flat otherwise
    is_per_pound = classification.table_type in ("us_multiweight", "us_express_freight", "intl_premium")

    # First pass: find a reference weight column from the first table that has one.
    # Some pages have multiple side-by-side tables where only the first has a weight column.
    reference_weight_strings: Optional[list[list[str]]] = None
    reference_weight_col: Optional[int] = None
    for table in rate_tables:
        wc = _find_weight_column(table)
        if wc is not None:
            reference_weight_strings = _collect_weight_strings(table, wc)
            reference_weight_col = wc
            break

    # For FedEx One Rate tables, zones are described as ranges in the page text
    # (e.g. "zone 2", "zones 3–4", "zones 5–8"). Extract these before the main loop.
    is_one_rate = classification.table_type == "us_one_rate"
    one_rate_zones: list[str] = []
    use_one_rate_zones = False
    if is_one_rate and page_text:
        one_rate_zones = _extract_one_rate_zone_labels(page_text)
        use_one_rate_zones = len(one_rate_zones) == len(rate_tables)

    # Extract zone labels from page text for multi-zone pages (e.g. page 43 has Zone 2-7).
    # Zones in FedEx PDFs are labelled as "Zone 2", "Zone 3", … in the text between tables,
    # not inside the pdfplumber table cells, so _extract_table_zone() returns None for them.
    text_zones = _extract_zones_from_text(page_text) if page_text else []
    # Only use text-derived zones when their count matches the number of rate tables,
    # ensuring a 1-to-1 positional mapping.
    use_text_zones = len(text_zones) == len(rate_tables)

    # Process ALL rate tables on the page (some pages have multiple sub-tables per zone group)
    for table_idx, table in enumerate(rate_tables):
        # --- Detect per-table zone and region from headers ---
        table_zone = (
            _extract_table_zone(table)
            or (one_rate_zones[table_idx] if use_one_rate_zones else None)
            or (text_zones[table_idx] if use_text_zones else None)
            or classification.zone
        )
        table_region = _extract_table_region(table) or classification.destination_region

        service_col_map = _find_service_columns(table, classification)
        weight_col = _find_weight_column(table)
        has_own_weights = weight_col is not None

        # Freight tables (us_express_freight / us_multiweight) often have zone numbers as
        # column headers, with one service per table rather than one service per column.
        # Detect this layout FIRST — before the service-column fallback — so we don't
        # accidentally map zone price columns to service names.
        zone_col_map: dict[int, str] = {}
        use_zone_columns = False
        if not service_col_map and has_own_weights:
            _candidate = _find_zone_columns(table)
            if _candidate and table_idx < len(classification.services):
                zone_col_map = _candidate
                use_zone_columns = True

        if not service_col_map and not use_zone_columns:
            if has_own_weights:
                # Fall back: use columns 2..N for services
                n_services = len(classification.services)
                service_col_map = {
                    (i + 2): classification.services[i].service_name
                    for i in range(n_services)
                    if i + 2 < (len(table[0]) if table else 0)
                }
            else:
                # No weight column — all columns may be service columns.
                # _find_service_columns should have already matched them;
                # if not, try mapping columns 0..N to services.
                n_cols = len(table[0]) if table else 0
                n_services = len(classification.services)
                service_col_map = {
                    i: classification.services[i].service_name
                    for i in range(min(n_cols, n_services))
                }

        if not service_col_map and not use_zone_columns:
            continue  # skip this sub-table if no services matched

        service_cols = sorted(service_col_map.keys())

        # --- FedEx One Rate tables: rows are package types, not weights ---
        # Dispatch to the One Rate parser when explicitly classified, OR as a
        # fallback when no weight column exists and col 0 contains package type names.
        if is_one_rate or (
            weight_col is None
            and reference_weight_strings is None
            and any(
                _is_one_rate_data_row(_clean_cell(row[0] if row else None))
                for row in table
            )
        ):
            one_rate_entries = _parse_one_rate_table(
                table, service_col_map, service_cols,
                classification, table_zone, table_region, source_page,
            )
            entries.extend(one_rate_entries)
            continue

        # Helper to build a RateEntry from shared fields
        def _make_entry(
            _zone: str = table_zone,
            _region: Optional[str] = table_region,
            **kwargs,
        ) -> Optional[RateEntry]:  # type: ignore[return]
            svc_name = kwargs.pop("service_name")
            svc_info = next((s for s in classification.services if s.service_name == svc_name), None)
            try:
                return RateEntry(
                    source_page=source_page,
                    table_type=classification.table_type,
                    service_category=_table_type_to_category(classification.table_type),
                    service_name=svc_name,
                    delivery_commitment=svc_info.delivery_commitment if svc_info else None,
                    direction=classification.direction or "domestic",
                    zone=_zone,
                    destination_region=_region,
                    **kwargs,
                )
            except Exception as exc:
                logger.debug("Skipping entry (svc=%s): %s", svc_name, exc)
                return None

        # --- Zone-column table (freight or box-rate) ---
        # Layout: col 0 = weight/label, zone cols = prices, each table = one service
        if use_zone_columns:
            svc_info = classification.services[table_idx]
            zone_cols = sorted(zone_col_map.keys())

            # Box-rate tables (intl 10kg/25kg Box): all zone labels are single letters A–O.
            # They have a different row structure — dispatch to the dedicated parser.
            if all(re.match(r"^[A-O]$", z) for z in zone_col_map.values()):
                box_entries = _parse_box_rate_zone_col_table(
                    table, zone_col_map, zone_cols, svc_info,
                    classification, table_region, source_page,
                )
                entries.extend(box_entries)
                continue

            mc_min, mc_max = _get_min_charge_weight_range(table, weight_col)

            for row_idx in _find_data_rows(table, weight_col):
                row_data = table[row_idx]
                weight_cell = _clean_cell(
                    row_data[weight_col] if weight_col < len(row_data) else None
                )
                is_standalone_min_charge = (
                    "minimum" in weight_cell.lower() and not re.search(r"\d", weight_cell)
                )

                if is_standalone_min_charge:
                    # Standalone "Minimum charge" row (one price per zone column)
                    for col_idx in zone_cols:
                        price_raw = _clean_cell(row_data[col_idx] if col_idx < len(row_data) else None)
                        price = _parse_price(price_raw)
                        note = _parse_price_note(price_raw) or ("Rate not specified" if price is None else None)
                        e = _make_entry(
                            _zone=zone_col_map[col_idx],
                            service_name=svc_info.service_name,
                            package_type="Other packaging",
                            weight_lbs=None,
                            weight_range_min_lbs=mc_min,
                            weight_range_max_lbs=mc_max,
                            weight_unit="lbs",
                            price_usd=price,
                            price_type="minimum_charge",
                            price_note=note,
                        )
                        if e is not None:
                            entries.append(e)
                else:
                    # Regular weight rows (possibly multivalue cells)
                    for wdata in _expand_multivalue_cells(row_data, weight_col, zone_cols):
                        price_pairs = wdata["prices"]
                        for col_pos, col_idx in enumerate(zone_cols):
                            if col_pos >= len(price_pairs):
                                break
                            price, note = price_pairs[col_pos]
                            e = _make_entry(
                                _zone=zone_col_map[col_idx],
                                service_name=svc_info.service_name,
                                package_type="Other packaging",
                                weight_lbs=wdata["weight_lbs"],
                                weight_range_min_lbs=wdata["weight_range_min_lbs"],
                                weight_range_max_lbs=wdata["weight_range_max_lbs"],
                                weight_unit=wdata["weight_unit"],
                                price_usd=price,
                                price_type="per_pound" if is_per_pound else "flat",
                                price_note=note,
                            )
                            if e is not None:
                                entries.append(e)

                    # Also extract "Minimum charge" embedded in multivalue weight cells
                    # (e.g. Table 4: "151–499 lbs.\n...\nMinimum charge" in one cell)
                    weight_parts = [w.strip() for w in weight_cell.split("\n") if w.strip()]
                    for mc_idx, w_str in enumerate(weight_parts):
                        if "minimum" not in w_str.lower():
                            continue
                        for col_idx in zone_cols:
                            price_cell = _clean_cell(
                                row_data[col_idx] if col_idx < len(row_data) else None
                            )
                            price_parts = [p.strip() for p in price_cell.split("\n") if p.strip()]
                            p_str = price_parts[mc_idx] if mc_idx < len(price_parts) else ""
                            price = _parse_price(p_str)
                            note = _parse_price_note(p_str) or ("Rate not specified" if price is None else None)
                            e = _make_entry(
                                _zone=zone_col_map[col_idx],
                                service_name=svc_info.service_name,
                                package_type="Other packaging",
                                weight_lbs=None,
                                weight_range_min_lbs=mc_min,
                                weight_range_max_lbs=mc_max,
                                weight_unit="lbs",
                                price_usd=price,
                                price_type="minimum_charge",
                                price_note=note,
                            )
                            if e is not None:
                                entries.append(e)

            continue  # zone-column table fully handled; skip normal paths

        # --- Parse special package-type rows (FedEx Envelope, FedEx Pak) ---
        for row in table:
            col0 = _clean_cell(row[0] if row else None).lower()
            col0_norm = re.sub(r"\s+", " ", col0)

            if "up to" in col0_norm and "oz" in col0_norm:
                pkg_type = "FedEx Envelope"
                # "up to 8 oz" → range [0, 0.5 lbs] with unit converted to lbs
                oz_in_lbs = _parse_weight(col0)  # _parse_weight already returns oz/16
                weight_kwargs: dict = {
                    "weight_lbs": None,
                    "weight_range_min_lbs": 0.0,
                    "weight_range_max_lbs": oz_in_lbs,
                    "weight_unit": "lbs",
                }
            elif "pak" in col0_norm and "fedex" in col0_norm:
                pkg_type = "FedEx Pak"
                weight_kwargs = {"weight_lbs": None, "weight_unit": "lbs"}
            elif "envelope" in col0_norm:
                continue  # header row; prices follow in the next "up to 8 oz." row
            else:
                continue

            for col_idx in service_cols:
                price_raw = _clean_cell(row[col_idx] if col_idx < len(row) else None)
                price = _parse_price(price_raw)
                note = _parse_price_note(price_raw) or ("Rate not specified" if price is None else None)
                e = _make_entry(
                    service_name=service_col_map[col_idx],
                    package_type=pkg_type,
                    price_usd=price,
                    price_type="flat",
                    price_note=note,
                    **weight_kwargs,
                )
                if e is not None:
                    entries.append(e)

        # --- Parse main weight rows ---
        if has_own_weights:
            # Normal path: table has its own weight column
            for row_idx in _find_data_rows(table, weight_col):
                row = table[row_idx]
                for wdata in _expand_multivalue_cells(row, weight_col, service_cols):
                    price_pairs = wdata["prices"]
                    for col_pos, col_idx in enumerate(service_cols):
                        if col_pos >= len(price_pairs):
                            break
                        price, note = price_pairs[col_pos]
                        e = _make_entry(
                            service_name=service_col_map[col_idx],
                            package_type="Other packaging",
                            weight_lbs=wdata["weight_lbs"],
                            weight_range_min_lbs=wdata["weight_range_min_lbs"],
                            weight_range_max_lbs=wdata["weight_range_max_lbs"],
                            weight_unit=wdata["weight_unit"],
                            price_usd=price,
                            price_type="per_pound" if is_per_pound else "flat",
                            price_note=note,
                        )
                        if e is not None:
                            entries.append(e)
        elif reference_weight_strings:
            # No weight column — borrow weights from the reference table
            data_rows = _find_data_rows_no_weight(table)
            for data_row_pos, row_idx in enumerate(data_rows):
                if data_row_pos >= len(reference_weight_strings):
                    logger.debug(
                        "Page %d: more data rows than reference weight groups (row %d)",
                        source_page, row_idx,
                    )
                    break
                row = table[row_idx]
                weight_strs = reference_weight_strings[data_row_pos]
                for wdata in _expand_with_external_weights(row, service_cols, weight_strs):
                    price_pairs = wdata["prices"]
                    for col_pos, col_idx in enumerate(service_cols):
                        if col_pos >= len(price_pairs):
                            break
                        price, note = price_pairs[col_pos]
                        e = _make_entry(
                            service_name=service_col_map[col_idx],
                            package_type="Other packaging",
                            weight_lbs=wdata["weight_lbs"],
                            weight_range_min_lbs=wdata["weight_range_min_lbs"],
                            weight_range_max_lbs=wdata["weight_range_max_lbs"],
                            weight_unit=wdata["weight_unit"],
                            price_usd=price,
                            price_type="per_pound" if is_per_pound else "flat",
                            price_note=note,
                        )
                        if e is not None:
                            entries.append(e)
        else:
            logger.warning(
                "Page %d: table has no weight column and no reference weights available",
                source_page,
            )

    logger.info(
        "table_parser: page=%d type=%s zone=%s entries=%d",
        source_page,
        classification.table_type,
        classification.zone,
        len(entries),
    )
    return entries


def _table_type_to_category(table_type: Optional[TableType]) -> str:
    """Map table_type to a human-readable service_category."""
    mapping = {
        "us_package": "US Package",
        "us_express_freight": "US Express Freight",
        "us_multiweight": "US Express Multiweight",
        "us_one_rate": "FedEx One Rate",
        "sameday": "FedEx SameDay",
        "intl_package_export": "International Package US Export",
        "intl_package_import": "International Package US Import",
        "intl_premium": "International Premium",
        "ground_domestic": "FedEx Ground",
        "ground_ak_hi": "FedEx Ground Alaska/Hawaii",
        "ground_canada": "FedEx International Ground Canada",
    }
    return mapping.get(table_type or "", "Unknown")
