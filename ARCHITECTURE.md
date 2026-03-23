# FedEx Table Extractor — Architecture Plan

## Context

The PDF is the **FedEx Service Guide (effective Jan 6, 2025, updated Sep 22, 2025)** — 186 pages covering U.S. domestic, international export/import, and ground shipping rates, plus fee schedules and Terms & Conditions.

**Goal**: Extract all weight-based rate tables into a flat, denormalized SQLite database for easy querying. Each DB row = one (weight, service, zone, price) combination.

**Design decisions**:
- Scope: rate tables only (weight → price), skip zone charts, fee tables, T&C
- LLM: LangChain-agnostic (provider-swappable via `LLM_PROVIDER` env var)
- Page handling: each page processed independently (LLM infers context from page header)
- Schema: flat/denormalized rows (one row per rate point)

---

## PDF Structure Summary

| Pages | Content | In Scope |
|-------|---------|----------|
| 1–12 | Cover, TOC, rate instructions, zone charts | ❌ |
| 13–43 | US package rates by zone (2–8), all weights | ✅ |
| 44 | FedEx Express Multiweight per-pound rates | ✅ |
| 45 | FedEx SameDay rates | ✅ |
| 48 | US express freight per-pound rates by zone | ✅ |
| 49–116 | International package rates (US export + import, zones A–O) | ✅ |
| 117 | FedEx International Premium rates | ✅ |
| 118–120 | FedEx Ground/Home Delivery rates (Zones 2–8) | ✅ |
| 121–122 | FedEx Ground Alaska & Hawaii rates | ✅ |
| 124 | FedEx International Ground Canada rates | ✅ |
| 125–186 | Fee schedules, service matrices, Terms & Conditions | ❌ |

---

## Table Types Observed

1. **US Package Rate Table** — columns = 6 services, rows = weight in lbs, grouped by zone
2. **Freight Per-Pound Rate Table** — weight ranges (151–499 lbs, etc.) × multiple zones × services
3. **Multiweight Rate Table** — weight ranges × services, rate is per-lb multiplier
4. **SameDay Flat Rate Table** — weight range → flat dollar amount (no zone)
5. **International Package Rate Table** — weight × 3–5 services, per destination zone (A–O)
6. **FedEx Ground Rate Table** — weight × zones 2–8, plus Alaska/Hawaii zone variants
7. **Canada Ground Rate Table** — weight × 2 zones (51, 54)

All share a common structure: **(weight or weight range) × (service) → price ($)** plus context metadata (zone, direction, region).

---

## SQLite Schema (Flat/Denormalized)

```sql
CREATE TABLE rates (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Source
    source_page          INTEGER NOT NULL,
    effective_date       TEXT    NOT NULL DEFAULT '2025-01-06',
    -- Table classification
    table_type           TEXT    NOT NULL,
    -- Allowed values:
    --   "us_package"          US domestic package rates (zones 2–8)
    --   "us_express_freight"  US domestic freight per-pound rates (zones 2–16)
    --   "us_multiweight"      US express multiweight per-lb bulk rates
    --   "sameday"             FedEx SameDay® — no zone, weight-only pricing
    --   "intl_package_export" International package US export (zones A–O)
    --   "intl_package_import" International package US import (zones A–O)
    --   "intl_premium"        FedEx International Premium® freight
    --   "ground_domestic"     FedEx Ground / Home Delivery (zones 2–8)
    --   "ground_ak_hi"        FedEx Ground Alaska / Hawaii (zones 17,22,23,25,9,14,92,96)
    --   "ground_canada"       FedEx International Ground Canada (zones 51,54)
    service_category     TEXT    NOT NULL,
    service_name         TEXT    NOT NULL,
    delivery_commitment  TEXT,
    -- Geography
    direction            TEXT    NOT NULL,  -- "domestic" | "us_export" | "us_import"
    zone                 TEXT,
    -- zone is NOT NULL for all table_type values EXCEPT "sameday".
    -- Application-level validation in models.py enforces this constraint.
    destination_region   TEXT,
    -- Package
    package_type         TEXT,
    weight_lbs           REAL,
    weight_range_min_lbs REAL,
    weight_range_max_lbs REAL,
    weight_unit          TEXT    NOT NULL DEFAULT 'lbs',
    -- Pricing
    price_usd            REAL,              -- NULL for '*' / '**' footnote entries
    price_type           TEXT    NOT NULL,  -- "flat" | "per_pound" | "minimum_charge"
    price_note           TEXT,
    -- Quality
    confidence           TEXT    NOT NULL DEFAULT 'high',
    extraction_notes     TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_page, table_type, service_name, zone, package_type, weight_lbs, weight_range_min_lbs)
);

CREATE INDEX idx_rates_lookup ON rates(service_name, zone, weight_lbs);
CREATE INDEX idx_rates_type   ON rates(table_type, direction);
CREATE INDEX idx_rates_page   ON rates(source_page);
```

A `processing_log` table also exists to track per-page extraction status for partial-run recovery.

---

## Common Structured Output (Pydantic)

One unified `RateEntry` Pydantic model maps to all 7 table types and directly to the SQLite schema — zero post-processing transformation needed.

Key validation: `zone` is Optional at the type level but a `@model_validator` raises `ValueError` if `zone is None` and `table_type != "sameday"`. FedEx SameDay is the only service with no zone structure.

---

## File Structure

```
fedex-table-extractor/
├── ARCHITECTURE.md         # this file
├── pyproject.toml          # dependencies
├── .env.example            # LLM_PROVIDER, API keys, paths
├── src/
│   ├── config.py           # pydantic-settings Settings class
│   ├── models.py           # RateEntry, PageExtraction Pydantic models
│   ├── pdf_reader.py       # pdfplumber extraction → PageContent dataclass
│   ├── page_formatter.py   # PageContent → LLM-readable string
│   ├── llm_extractor.py    # LangChain chain with structured output + tenacity retry
│   ├── db_writer.py        # SQLite writer (serial, thread-safe, batch insert)
│   ├── pipeline.py         # Async orchestrator (sequential + parallel modes)
│   └── main.py             # Click CLI: extract / stats / query commands
├── tests/
│   ├── test_pdf_reader.py
│   ├── test_llm_extractor.py
│   └── test_db_writer.py
└── table_full.pdf
```

---

## Pipeline Flow

```
PDF file
  │
  ▼
PDFPageReader.read_page(n)          [pdfplumber: text + tables]
  │
  ▼
PageFormatter.format(page_content)  [clean text for LLM, artifact removal]
  │
  ├─── Empty page? ──► skip (no LLM call)
  │
  ▼
LLMExtractor.extract(formatted)     [LangChain + structured output]
  │
  ├─── skipped=True? ──► log and continue
  │
  ▼  (for each RateEntry in result.rates)
DBWriter.insert_batch(entries)      [sqlite3 executemany, INSERT OR IGNORE]
  │
  ▼
ProcessingLog.mark_done(page_n)
```

Parallel mode wraps each page in an async task with `asyncio.Semaphore(concurrency)`.
The DB writer runs in a **single serial asyncio queue** — all page workers enqueue rows, one writer drains the queue — eliminating SQLite write contention.

---

## Edge Cases & Production Hardening

| # | Edge Case | Handling |
|---|-----------|----------|
| 1 | Rotated text artifacts (`.sbl ni thgiew...`) | Pre-processed in `page_formatter.py` before LLM sees the content |
| 2 | `*` / `**` footnote prices | Explicit LLM prompt rules → `price_usd=null`, `price_note` set |
| 3 | Tables spanning multiple pages | LLM infers zone/context from repeated page header |
| 4 | LLM API rate limit / timeout | `tenacity` exponential backoff (max_retries, retry_wait_seconds) |
| 5 | LLM structured output validation failure | `ValidationError` caught → log page + error, insert 0 rows, continue |
| 6 | Garbled pdfplumber table extraction | Raw text also sent; LLM uses the cleaner representation |
| 7 | Duplicate rows on re-run | `INSERT OR IGNORE` + UNIQUE constraint on key columns |
| 8 | Partial run recovery | `--pages` CLI flag + `processing_log` table tracks completed pages |
| 9 | Token overflow (very long pages) | Content length guard in formatter; truncate with logged warning |
| 10 | Non-rate pages (T&C, zone charts) | LLM returns `skipped=true`; 0 DB writes |
| 11 | Prices with commas (`$1,038.28`) | LLM strips formatting; Pydantic coerces to float |
| 12 | NULL zone for non-SameDay rows | Pydantic `@model_validator` raises before DB write |
| 13 | Parallel SQLite write contention | Serial asyncio writer queue; readers are concurrent, writer is serial |
| 14 | Empty / scanned-image pages | pdfplumber returns empty text → skip without LLM call |

---

## CLI Usage

```bash
# Install dependencies
pip install -e ".[dev]"
cp .env.example .env   # fill in your API key

# Single page smoke test
python -m src.main extract --pages 13

# Page range, parallel
python -m src.main extract --pages 13-124 --mode parallel

# Full PDF
python -m src.main extract --mode parallel

# Show DB statistics
python -m src.main stats

# Query rates
python -m src.main query --service "FedEx Priority Overnight" --zone 5 --weight 10
```

---

## Verification Plan

1. `extract --pages 13` → DB has ~50–100 rows, zone=2, correct service names and prices
2. `stats` → breakdown by service_category and table_type
3. Spot-check: Zone 2, FedEx Priority Overnight, 1 lb FedEx Envelope → expect `$34.71`
4. `extract --pages 150` (T&C page) → 0 rows inserted, `skipped=true` in log
5. `extract --pages 13-124 --mode parallel` → ~10,000–15,000 total rows
6. Re-run same range → row count unchanged (idempotent)
7. SQLite query: `SELECT price_usd FROM rates WHERE service_name='FedEx Ground' AND zone='5' AND weight_lbs=10`
