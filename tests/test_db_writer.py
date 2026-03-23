"""
Tests for DBWriter.

Uses an in-memory SQLite database so tests are fast and have no side-effects.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.db_writer import DBWriter
from src.models import RateEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_rates.db"


@pytest.fixture
def writer(db_path: Path) -> DBWriter:
    w = DBWriter(db_path)
    yield w
    w.close()


def _make_entry(**overrides) -> RateEntry:
    defaults = {
        "source_page": 13,
        "table_type": "us_package",
        "service_category": "US Package",
        "service_name": "FedEx Priority Overnight",
        "direction": "domestic",
        "zone": "2",
        "weight_lbs": 1.0,
        "price_usd": 34.71,
        "price_type": "flat",
    }
    return RateEntry(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, db_path: Path) -> None:
        with DBWriter(db_path) as w:
            conn = sqlite3.connect(str(db_path))
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.close()
        assert "rates" in tables
        assert "processing_log" in tables

    def test_idempotent_schema(self, db_path: Path) -> None:
        """Opening DBWriter twice should not raise."""
        with DBWriter(db_path):
            pass
        with DBWriter(db_path):
            pass


# ---------------------------------------------------------------------------
# Insert tests
# ---------------------------------------------------------------------------

class TestInsertBatch:
    def test_insert_single_entry(self, writer: DBWriter) -> None:
        entry = _make_entry()
        inserted = writer.insert_batch([entry])
        assert inserted == 1

    def test_insert_multiple_entries(self, writer: DBWriter) -> None:
        entries = [_make_entry(weight_lbs=float(i)) for i in range(1, 11)]
        inserted = writer.insert_batch(entries)
        assert inserted == 10

    def test_insert_empty_list(self, writer: DBWriter) -> None:
        inserted = writer.insert_batch([])
        assert inserted == 0

    def test_duplicate_ignored(self, writer: DBWriter) -> None:
        entry = _make_entry()
        writer.insert_batch([entry])
        inserted = writer.insert_batch([entry])
        assert inserted == 0  # duplicate ignored

    def test_idempotent_rerun(self, writer: DBWriter) -> None:
        entries = [_make_entry(weight_lbs=float(i)) for i in range(1, 6)]
        writer.insert_batch(entries)
        writer.insert_batch(entries)  # second run, same data
        stats = writer.get_stats()
        assert stats.total_rows == 5  # not 10

    def test_null_price_stored(self, writer: DBWriter) -> None:
        entry = _make_entry(price_usd=None, price_note="Based on package weight rate")
        writer.insert_batch([entry])
        stats = writer.get_stats()
        assert stats.total_rows == 1

    def test_sameday_entry_null_zone(self, writer: DBWriter) -> None:
        entry = RateEntry(
            source_page=45,
            table_type="sameday",
            service_category="FedEx SameDay",
            service_name="FedEx SameDay",
            direction="domestic",
            zone=None,
            weight_range_min_lbs=0.0,
            weight_range_max_lbs=25.0,
            price_usd=270.0,
            price_type="flat",
        )
        inserted = writer.insert_batch([entry])
        assert inserted == 1

    def test_large_batch(self, writer: DBWriter) -> None:
        entries = [_make_entry(weight_lbs=float(i), zone="5") for i in range(1, 201)]
        inserted = writer.insert_batch(entries)
        assert inserted == 200


# ---------------------------------------------------------------------------
# Processing log tests
# ---------------------------------------------------------------------------

class TestProcessingLog:
    def test_log_success(self, writer: DBWriter) -> None:
        writer.log_page(13, status="success", rows_inserted=50)
        processed = writer.get_processed_pages()
        assert 13 in processed

    def test_log_skipped(self, writer: DBWriter) -> None:
        writer.log_page(150, status="skipped", skip_reason="T&C page")
        processed = writer.get_processed_pages()
        assert 150 not in processed  # skipped ≠ success

    def test_log_upsert(self, writer: DBWriter) -> None:
        writer.log_page(13, status="failed", error_msg="API error")
        writer.log_page(13, status="success", rows_inserted=50)
        processed = writer.get_processed_pages()
        assert 13 in processed


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_db_stats(self, writer: DBWriter) -> None:
        stats = writer.get_stats()
        assert stats.total_rows == 0
        assert stats.pages_processed == 0

    def test_stats_after_insert(self, writer: DBWriter) -> None:
        entries = [_make_entry(weight_lbs=float(i)) for i in range(1, 6)]
        writer.insert_batch(entries)
        writer.log_page(13, status="success", rows_inserted=5)

        stats = writer.get_stats()
        assert stats.total_rows == 5
        assert stats.pages_processed == 1
        assert "us_package" in stats.rows_by_table_type
        assert stats.rows_by_table_type["us_package"] == 5

    def test_stats_multiple_types(self, writer: DBWriter) -> None:
        entries = [
            _make_entry(weight_lbs=1.0),
            _make_entry(table_type="ground_domestic", service_name="FedEx Ground", weight_lbs=2.0),
        ]
        writer.insert_batch(entries)
        stats = writer.get_stats()
        assert "us_package" in stats.rows_by_table_type
        assert "ground_domestic" in stats.rows_by_table_type


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_by_service(self, writer: DBWriter) -> None:
        entries = [
            _make_entry(weight_lbs=1.0, price_usd=34.71),
            _make_entry(service_name="FedEx 2Day", weight_lbs=1.0, price_usd=25.23),
        ]
        writer.insert_batch(entries)
        rows = writer.query_rates(service_name="Priority Overnight")
        assert len(rows) == 1
        assert rows[0]["price_usd"] == pytest.approx(34.71)

    def test_query_by_zone_and_weight(self, writer: DBWriter) -> None:
        entries = [
            _make_entry(weight_lbs=5.0, zone="2", price_usd=40.0),
            _make_entry(weight_lbs=5.0, zone="5", price_usd=60.0),
        ]
        writer.insert_batch(entries)
        rows = writer.query_rates(zone="5", weight_lbs=5.0)
        assert len(rows) == 1
        assert rows[0]["price_usd"] == pytest.approx(60.0)

    def test_query_no_results(self, writer: DBWriter) -> None:
        rows = writer.query_rates(service_name="NonExistentService")
        assert rows == []
