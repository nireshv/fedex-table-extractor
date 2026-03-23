"""
Tests for Pydantic models (RateEntry, PageExtraction).

Focus areas:
- Field validators (price coercion, weight coercion)
- Model validators (zone required, weight consistency, null price needs note)
- Valid construction across all table_type values
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import PageExtraction, RateEntry


# ---------------------------------------------------------------------------
# Minimal valid RateEntry factories
# ---------------------------------------------------------------------------

def _us_package(**overrides) -> dict:
    base = {
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
    return {**base, **overrides}


def _sameday(**overrides) -> dict:
    base = {
        "source_page": 45,
        "table_type": "sameday",
        "service_category": "FedEx SameDay",
        "service_name": "FedEx SameDay",
        "direction": "domestic",
        "zone": None,  # intentionally no zone
        "weight_range_min_lbs": 0.0,
        "weight_range_max_lbs": 25.0,
        "price_usd": 270.0,
        "price_type": "flat",
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestRateEntryConstruction:
    def test_us_package_valid(self) -> None:
        entry = RateEntry(**_us_package())
        assert entry.zone == "2"
        assert entry.price_usd == 34.71

    def test_sameday_no_zone(self) -> None:
        entry = RateEntry(**_sameday())
        assert entry.zone is None
        assert entry.table_type == "sameday"

    def test_all_table_types_valid(self) -> None:
        table_types = [
            "us_package", "us_express_freight", "us_multiweight",
            "intl_package_export", "intl_package_import", "intl_premium",
            "ground_domestic", "ground_ak_hi", "ground_canada",
        ]
        for tt in table_types:
            data = _us_package(table_type=tt, zone="5")
            entry = RateEntry(**data)
            assert entry.table_type == tt

    def test_per_pound_price_type(self) -> None:
        data = _us_package(
            table_type="us_express_freight",
            weight_lbs=None,
            weight_range_min_lbs=151.0,
            weight_range_max_lbs=499.0,
            price_usd=2.69,
            price_type="per_pound",
        )
        entry = RateEntry(**data)
        assert entry.price_type == "per_pound"
        assert entry.weight_lbs is None
        assert entry.weight_range_min_lbs == 151.0

    def test_minimum_charge(self) -> None:
        data = _us_package(
            price_usd=185.0,
            price_type="minimum_charge",
        )
        entry = RateEntry(**data)
        assert entry.price_type == "minimum_charge"

    def test_null_price_with_note(self) -> None:
        data = _us_package(price_usd=None, price_note="Based on package weight rate")
        entry = RateEntry(**data)
        assert entry.price_usd is None
        assert entry.price_note is not None


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestRateEntryValidators:
    def test_zone_required_for_non_sameday(self) -> None:
        data = _us_package(zone=None)
        with pytest.raises(ValidationError, match="zone must be set"):
            RateEntry(**data)

    def test_zone_not_required_for_sameday(self) -> None:
        entry = RateEntry(**_sameday())
        assert entry.zone is None  # no error

    def test_null_price_auto_sets_note(self) -> None:
        # Validator auto-fills price_note when price_usd is None and no note given
        data = _us_package(price_usd=None, price_note=None)
        entry = RateEntry(**data)
        assert entry.price_usd is None
        assert entry.price_note is not None  # auto-set to default

    def test_weight_consistency_both_missing_adds_note(self) -> None:
        # Validator annotates extraction_notes instead of raising
        data = _us_package(weight_lbs=None, weight_range_min_lbs=None)
        entry = RateEntry(**data)
        assert entry.extraction_notes is not None
        assert "weight missing" in (entry.extraction_notes or "").lower()

    def test_weight_consistency_range_ok(self) -> None:
        data = _us_package(weight_lbs=None, weight_range_min_lbs=100.0)
        entry = RateEntry(**data)
        assert entry.weight_range_min_lbs == 100.0

    def test_price_coercion_dollar_string(self) -> None:
        data = _us_package(price_usd="$1,038.28")
        entry = RateEntry(**data)
        assert entry.price_usd == pytest.approx(1038.28)

    def test_price_coercion_star(self) -> None:
        data = _us_package(price_usd="*", price_note="Based on package weight rate")
        entry = RateEntry(**data)
        assert entry.price_usd is None

    def test_price_coercion_double_star(self) -> None:
        data = _us_package(price_usd="**", price_note="One-pound rate applies")
        entry = RateEntry(**data)
        assert entry.price_usd is None

    def test_weight_coercion_with_suffix(self) -> None:
        data = _us_package(weight_lbs="10 lbs.")
        entry = RateEntry(**data)
        assert entry.weight_lbs == 10.0

    def test_weight_coercion_comma(self) -> None:
        data = _us_package(weight_range_min_lbs="1,000")
        entry = RateEntry(**data)
        assert entry.weight_range_min_lbs == 1000.0


# ---------------------------------------------------------------------------
# PageExtraction tests
# ---------------------------------------------------------------------------

class TestPageExtraction:
    def test_empty_skipped(self) -> None:
        pe = PageExtraction(skipped=True, skip_reason="Terms and Conditions page")
        assert pe.skipped is True
        assert pe.rates == []

    def test_skipped_with_rates_raises(self) -> None:
        entry = RateEntry(**_us_package())
        with pytest.raises(ValidationError, match="skipped=True but rates list is non-empty"):
            PageExtraction(skipped=True, rates=[entry])

    def test_with_rates(self) -> None:
        entries = [RateEntry(**_us_package(weight_lbs=float(i))) for i in range(1, 6)]
        pe = PageExtraction(rates=entries)
        assert len(pe.rates) == 5
        assert pe.skipped is False
