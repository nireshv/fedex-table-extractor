"""
SQLite writer.

Responsibilities:
- Create the database schema on first run (idempotent)
- Provide a thread-safe serial write interface
- Insert rate entries in batches using executemany
- Track per-page processing status for partial-run recovery
- Expose stats queries for the CLI

Design notes:
- The writer is intentionally SYNCHRONOUS. In the async pipeline, all page
  workers enqueue results through an asyncio.Queue and a single writer
  coroutine drains it. This avoids SQLite's "cannot share connection across
  threads" limitation without needing WAL mode or connection pools.
- INSERT OR IGNORE enforces the UNIQUE constraint for idempotent re-runs.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import RateEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_RATES_TABLE = """
CREATE TABLE IF NOT EXISTS rates (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_page          INTEGER NOT NULL,
    effective_date       TEXT    NOT NULL DEFAULT '2025-01-06',
    -- Stable SHA-256 of all natural key fields (NULL → '') for deduplication.
    -- SQLite treats NULL != NULL in UNIQUE constraints, so multi-column UNIQUE
    -- on nullable fields allows duplicates. This column avoids that.
    row_fingerprint      TEXT    NOT NULL UNIQUE,
    table_type           TEXT    NOT NULL,
    service_category     TEXT    NOT NULL,
    service_name         TEXT    NOT NULL,
    delivery_commitment  TEXT,
    direction            TEXT    NOT NULL,
    zone                 TEXT,
    destination_region   TEXT,
    package_type         TEXT,
    weight_lbs           REAL,
    weight_range_min_lbs REAL,
    weight_range_max_lbs REAL,
    weight_unit          TEXT    NOT NULL DEFAULT 'lbs',
    price_usd            REAL,
    price_type           TEXT    NOT NULL,
    price_note           TEXT,
    confidence           TEXT    NOT NULL DEFAULT 'high',
    extraction_notes     TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_RATES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_rates_lookup ON rates(service_name, zone, weight_lbs);",
    "CREATE INDEX IF NOT EXISTS idx_rates_type   ON rates(table_type, direction);",
    "CREATE INDEX IF NOT EXISTS idx_rates_page   ON rates(source_page);",
]

_CREATE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS processing_log (
    page_number  INTEGER PRIMARY KEY,
    status       TEXT    NOT NULL,  -- 'success' | 'skipped' | 'failed'
    rows_inserted INTEGER NOT NULL DEFAULT 0,
    skip_reason  TEXT,
    error_msg    TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# INSERT statement — ignores on unique-constraint violation (idempotent re-runs).
# row_fingerprint is a SHA-256 computed in Python with NULL → '' substitution,
# which makes duplicates detectable even when nullable key fields are NULL.
_INSERT_RATE = """
INSERT OR IGNORE INTO rates (
    source_page, effective_date, row_fingerprint,
    table_type, service_category, service_name,
    delivery_commitment, direction, zone, destination_region, package_type,
    weight_lbs, weight_range_min_lbs, weight_range_max_lbs, weight_unit,
    price_usd, price_type, price_note, confidence, extraction_notes
) VALUES (
    :source_page, :effective_date, :row_fingerprint,
    :table_type, :service_category, :service_name,
    :delivery_commitment, :direction, :zone, :destination_region, :package_type,
    :weight_lbs, :weight_range_min_lbs, :weight_range_max_lbs, :weight_unit,
    :price_usd, :price_type, :price_note, :confidence, :extraction_notes
);
"""

_UPSERT_LOG = """
INSERT INTO processing_log (page_number, status, rows_inserted, skip_reason, error_msg)
VALUES (:page_number, :status, :rows_inserted, :skip_reason, :error_msg)
ON CONFLICT(page_number) DO UPDATE SET
    status        = excluded.status,
    rows_inserted = excluded.rows_inserted,
    skip_reason   = excluded.skip_reason,
    error_msg     = excluded.error_msg,
    processed_at  = CURRENT_TIMESTAMP;
"""


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class DBStats:
    total_rows: int
    pages_processed: int
    pages_skipped: int
    pages_failed: int
    rows_by_table_type: dict[str, int]
    rows_by_service_category: dict[str, int]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class DBWriter:
    """
    Synchronous SQLite writer. Not thread-safe — use from a single thread/coroutine.

    Usage:
        writer = DBWriter("fedex_rates.db")
        writer.insert_batch(rate_entries)
        writer.log_page(page_num, status="success", rows_inserted=42)
        writer.close()

    Or as a context manager:
        with DBWriter("fedex_rates.db") as writer:
            writer.insert_batch(entries)
    """

    def __init__(self, db_path: str | Path, batch_size: int = 100) -> None:
        self._path = Path(db_path)
        self._batch_size = batch_size
        self._conn = self._connect()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_batch(self, entries: list[RateEntry]) -> int:
        """
        Insert a list of RateEntry objects.

        Uses executemany in chunks of batch_size for efficiency.
        INSERT OR IGNORE silently skips duplicates.

        Returns the number of rows actually inserted (duplicates not counted).
        """
        if not entries:
            return 0

        rows = [_entry_to_row(e) for e in entries]
        inserted = 0

        for chunk_start in range(0, len(rows), self._batch_size):
            chunk = rows[chunk_start : chunk_start + self._batch_size]
            try:
                cursor = self._conn.executemany(_INSERT_RATE, chunk)
                self._conn.commit()
                inserted += cursor.rowcount
            except sqlite3.Error as exc:
                logger.error(
                    "Batch insert failed, rolling back chunk",
                    extra={"error": str(exc), "chunk_size": len(chunk)},
                )
                self._conn.rollback()
                raise

        return inserted

    def log_page(
        self,
        page_number: int,
        status: str,
        rows_inserted: int = 0,
        skip_reason: str | None = None,
        error_msg: str | None = None,
    ) -> None:
        """Record or update the processing status for a page."""
        try:
            self._conn.execute(
                _UPSERT_LOG,
                {
                    "page_number": page_number,
                    "status": status,
                    "rows_inserted": rows_inserted,
                    "skip_reason": skip_reason,
                    "error_msg": error_msg,
                },
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error(
                "Failed to write processing log",
                extra={"page": page_number, "error": str(exc)},
            )

    def get_processed_pages(self) -> set[int]:
        """Return the set of page numbers that completed successfully."""
        cursor = self._conn.execute(
            "SELECT page_number FROM processing_log WHERE status = 'success'"
        )
        return {row[0] for row in cursor.fetchall()}

    def get_stats(self) -> DBStats:
        """Return summary statistics about the database contents."""
        total = self._conn.execute("SELECT COUNT(*) FROM rates").fetchone()[0]

        log_counts = self._conn.execute(
            "SELECT status, COUNT(*) FROM processing_log GROUP BY status"
        ).fetchall()
        status_map: dict[str, int] = {row[0]: row[1] for row in log_counts}

        by_type = self._conn.execute(
            "SELECT table_type, COUNT(*) FROM rates GROUP BY table_type ORDER BY COUNT(*) DESC"
        ).fetchall()

        by_category = self._conn.execute(
            "SELECT service_category, COUNT(*) FROM rates GROUP BY service_category ORDER BY COUNT(*) DESC"
        ).fetchall()

        return DBStats(
            total_rows=total,
            pages_processed=status_map.get("success", 0),
            pages_skipped=status_map.get("skipped", 0),
            pages_failed=status_map.get("failed", 0),
            rows_by_table_type=dict(by_type),
            rows_by_service_category=dict(by_category),
        )

    def query_rates(
        self,
        service_name: str | None = None,
        zone: str | None = None,
        weight_lbs: float | None = None,
        table_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Simple parameterised rate query for the CLI 'query' command."""
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if service_name:
            conditions.append("service_name LIKE :service_name")
            params["service_name"] = f"%{service_name}%"
        if zone is not None:
            conditions.append("zone = :zone")
            params["zone"] = str(zone)
        if weight_lbs is not None:
            conditions.append("weight_lbs = :weight_lbs")
            params["weight_lbs"] = weight_lbs
        if table_type:
            conditions.append("table_type = :table_type")
            params["table_type"] = table_type

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT source_page, table_type, service_name, zone, package_type,
                   weight_lbs, weight_range_min_lbs, weight_range_max_lbs,
                   price_usd, price_type, delivery_commitment, destination_region
            FROM rates
            {where}
            ORDER BY source_page, weight_lbs
            LIMIT :limit
        """
        params["limit"] = limit

        cursor = self._conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DBWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent readers
        conn.execute("PRAGMA synchronous=NORMAL;")  # balance durability/speed
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_schema(self) -> None:
        self._conn.execute(_CREATE_RATES_TABLE)
        for idx_sql in _CREATE_RATES_INDEXES:
            self._conn.execute(idx_sql)
        self._conn.execute(_CREATE_LOG_TABLE)
        self._conn.commit()
        logger.debug("Schema ensured", extra={"db": str(self._path)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_fingerprint(entry: RateEntry) -> str:
    """
    Compute a stable SHA-256 fingerprint for deduplication.

    All nullable key fields are coerced to empty string so that two rows
    with identical data but NULL fields produce the same fingerprint.
    SQLite's native UNIQUE constraint cannot do this (NULL != NULL).
    """
    key = "|".join(str(v) if v is not None else "" for v in [
        entry.source_page,
        entry.table_type,
        entry.service_name,
        entry.zone,
        entry.package_type,
        entry.weight_lbs,
        entry.weight_range_min_lbs,
        entry.weight_unit,
    ])
    return hashlib.sha256(key.encode()).hexdigest()


def _entry_to_row(entry: RateEntry) -> dict[str, Any]:
    """Convert a RateEntry Pydantic model to a plain dict for sqlite3 executemany."""
    return {
        "source_page": entry.source_page,
        "effective_date": entry.effective_date,
        "row_fingerprint": _row_fingerprint(entry),
        "table_type": entry.table_type,
        "service_category": entry.service_category,
        "service_name": entry.service_name,
        "delivery_commitment": entry.delivery_commitment,
        "direction": entry.direction,
        "zone": entry.zone,
        "destination_region": entry.destination_region,
        "package_type": entry.package_type,
        "weight_lbs": entry.weight_lbs,
        "weight_range_min_lbs": entry.weight_range_min_lbs,
        "weight_range_max_lbs": entry.weight_range_max_lbs,
        "weight_unit": entry.weight_unit,
        "price_usd": entry.price_usd,
        "price_type": entry.price_type,
        "price_note": entry.price_note,
        "confidence": entry.confidence,
        "extraction_notes": entry.extraction_notes,
    }
