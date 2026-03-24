"""
Pydantic models for the LLM's structured output.

RateEntry is the single unified schema that covers every rate table type
found in the FedEx Service Guide. It maps 1:1 to the SQLite 'rates' table
with no additional transformation.

PageExtraction is the top-level object the LLM returns per page.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

TableType = Literal[
    "us_package",           # US domestic package rates (zones 2–8)
    "us_express_freight",   # US domestic freight per-pound rates (zones 2–16)
    "us_multiweight",       # US express multiweight per-lb bulk rates
    "us_one_rate",          # FedEx One Rate® flat pricing by package type (no weight scale)
    "sameday",              # FedEx SameDay® — no zone, weight-only pricing
    "intl_package_export",  # International package US export (zones A–O)
    "intl_package_import",  # International package US import (zones A–O)
    "intl_premium",         # FedEx International Premium® freight
    "ground_domestic",      # FedEx Ground / Home Delivery (zones 2–8)
    "ground_ak_hi",         # FedEx Ground Alaska / Hawaii
    "ground_canada",        # FedEx International Ground Canada (zones 51, 54)
]

PriceType = Literal[
    "flat",           # dollar amount is the total per-shipment charge
    "per_pound",      # dollar amount must be multiplied by shipment weight
    "minimum_charge", # minimum floor charge for a service/zone
]

Direction = Literal["domestic", "us_export", "us_import"]

Confidence = Literal["high", "medium", "low"]


class RateEntry(BaseModel):
    """
    One rate data point: (weight × service × zone) → price.

    This model is intentionally flat/denormalized so every row is
    self-contained and can be inserted directly into SQLite without
    any joins or secondary lookups.
    """

    # --- Source metadata ---
    source_page: int = Field(description="1-indexed PDF page number this row was extracted from")
    effective_date: str = Field(
        default="2025-01-06",
        description="Rate effective date in YYYY-MM-DD format",
    )

    # --- Table classification ---
    table_type: TableType = Field(
        description=(
            "Category of rate table. Must be one of the defined literals. "
            "Use 'sameday' only for FedEx SameDay NFO/Counter rates which have no zone."
        )
    )

    # --- Service ---
    service_category: str = Field(
        description=(
            "High-level service group, e.g. 'US Package', 'US Express Freight', "
            "'International Package US Export', 'FedEx Ground', 'FedEx SameDay'"
        )
    )
    service_name: str = Field(
        description=(
            "Specific FedEx service name exactly as printed, e.g. "
            "'FedEx Priority Overnight', 'FedEx International Priority', "
            "'FedEx Ground', 'FedEx 2Day A.M.'"
        )
    )
    delivery_commitment: Optional[str] = Field(
        default=None,
        description=(
            "Delivery time commitment as printed, e.g. "
            "'Next day by 10:30 a.m. or 11 a.m.', '2nd day by 5 p.m.', "
            "'1–5 days based on distance to destination'"
        ),
    )

    # --- Geography ---
    direction: Direction = Field(
        description="'domestic' for US domestic, 'us_export' for outbound international, 'us_import' for inbound"
    )
    zone: Optional[str] = Field(
        default=None,
        description=(
            "Zone identifier as printed. Required for all table_type values EXCEPT 'sameday'. "
            "Examples: '2'–'8' for US domestic; 'A'–'O' for international; "
            "'17', '22', '23' for AK/HI ground; '51', '54' for Canada ground."
        ),
    )
    destination_region: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the geographic region for this rate, "
            "e.g. 'Puerto Rico', 'Canada', 'Mexico', "
            "'Zone J countries: Australia, Guam, Indonesia, Macau SAR China, Malaysia'"
        ),
    )

    # --- Package ---
    package_type: Optional[str] = Field(
        default=None,
        description=(
            "FedEx packaging type when rate differs by package, e.g. "
            "'FedEx Envelope', 'FedEx Pak', 'FedEx 10kg Box', 'FedEx 25kg Box', "
            "'Other packaging'. NULL when rate applies uniformly."
        ),
    )

    # --- Weight ---
    weight_lbs: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Exact shipment weight in pounds (or ounces when weight_unit='oz'). "
            "NULL for multiweight/freight rows that use a weight range instead."
        ),
    )
    weight_range_min_lbs: Optional[float] = Field(
        default=None,
        ge=0,
        description="Lower bound of weight range (inclusive) for per-pound / multiweight rows, in lbs.",
    )
    weight_range_max_lbs: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Upper bound of weight range (inclusive) for per-pound / multiweight rows, in lbs. "
            "NULL means open-ended (no upper limit)."
        ),
    )
    weight_unit: Literal["oz", "lbs"] = Field(
        default="lbs",
        description="Unit for weight_lbs. 'oz' only for FedEx Envelope (up to 8 oz) entries.",
    )

    # --- Pricing ---
    price_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Rate in USD. NULL when the table shows '*' or '**' (footnote-based rates). "
            "See price_note for the explanation."
        ),
    )
    price_type: PriceType = Field(
        default="flat",
        description=(
            "'flat' = total per-shipment charge; "
            "'per_pound' = multiply this amount by shipment weight to get total; "
            "'minimum_charge' = floor charge for this service/zone"
        ),
    )
    price_note: Optional[str] = Field(
        default=None,
        description=(
            "Explains a NULL price_usd. "
            "E.g. 'Based on package weight rate' (for *) or 'One-pound rate applies' (for **)."
        ),
    )

    # --- Extraction quality ---
    confidence: Confidence = Field(
        default="high",
        description=(
            "'high' = table structure was clear and values unambiguous; "
            "'medium' = minor ambiguity in alignment or header; "
            "'low' = table structure was garbled or inferred with uncertainty"
        ),
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Free-text notes about any extraction caveats or assumptions for this row.",
    )

    # --- Validators ---

    @field_validator("price_usd", mode="before")
    @classmethod
    def coerce_price(cls, v: object) -> Optional[float]:
        """Strip $ signs, commas, and whitespace; convert to float."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lstrip("$").replace(",", "").strip()
        if s in ("", "*", "**", "N/A", "n/a"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @field_validator("weight_lbs", "weight_range_min_lbs", "weight_range_max_lbs", mode="before")
    @classmethod
    def coerce_weight(cls, v: object) -> Optional[float]:
        """Strip 'lbs.' suffixes and convert to float."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower().rstrip(".").replace("lbs", "").replace("lb", "").replace(",", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @model_validator(mode="after")
    def zone_required_unless_sameday(self) -> "RateEntry":
        """
        Zone is required for every rate type except FedEx SameDay,
        which has no zone-based pricing structure.
        """
        if self.table_type != "sameday" and not self.zone:
            raise ValueError(
                f"zone must be set for table_type='{self.table_type}'. "
                "Only 'sameday' rates are zone-free."
            )
        return self

    @model_validator(mode="after")
    def weight_consistency(self) -> "RateEntry":
        """
        Either weight_lbs OR weight_range_min_lbs must be present.
        If neither is set, log a note rather than raising — allows partial
        data through instead of failing the entire page extraction.
        """
        has_exact = self.weight_lbs is not None
        has_range = self.weight_range_min_lbs is not None
        if not has_exact and not has_range:
            self.extraction_notes = (
                (self.extraction_notes or "") +
                " [weight missing — neither weight_lbs nor weight_range_min_lbs was set]"
            ).strip()
        return self

    @model_validator(mode="after")
    def null_price_needs_note(self) -> "RateEntry":
        """When price_usd is NULL, auto-set a default price_note if absent."""
        if self.price_usd is None and not self.price_note:
            self.price_note = "Rate not specified"
        return self


class PageExtraction(BaseModel):
    """
    Top-level object the LLM returns for every page.
    If the page contains no rate tables, skipped=True and rates is empty.
    """

    rates: list[RateEntry] = Field(
        default_factory=list,
        description=(
            "All rate entries extracted from this page. "
            "Empty list if page has no rate tables."
        ),
    )
    skipped: bool = Field(
        default=False,
        description=(
            "True if this page contains no weight-based rate tables "
            "(e.g. zone charts, fee descriptions, Terms and Conditions, cover pages)."
        ),
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of why the page was skipped, if skipped=True.",
    )

    @model_validator(mode="after")
    def skipped_means_empty(self) -> "PageExtraction":
        if self.skipped and self.rates:
            raise ValueError(
                "skipped=True but rates list is non-empty. "
                "Either set skipped=False or clear the rates list."
            )
        return self


# ---------------------------------------------------------------------------
# Lightweight classification model (used by hybrid extraction approach)
# ---------------------------------------------------------------------------

class ServiceInfo(BaseModel):
    """Describes one service column in the rate table."""
    service_name: str = Field(description="Exact FedEx service name as printed, e.g. 'FedEx Priority Overnight'")
    delivery_commitment: Optional[str] = Field(
        default=None,
        description="Delivery time as printed, e.g. 'Next day by 10:30 a.m. or 11 a.m.'",
    )
    package_type: Optional[str] = Field(
        default=None,
        description=(
            "Package type for tables where each sub-table covers a single packaging variant, "
            "e.g. 'FedEx 10kg Box', 'FedEx 25kg Box'. NULL for most tables."
        ),
    )


class PageClassification(BaseModel):
    """
    Lightweight LLM output used by the hybrid extraction pipeline.

    The LLM classifies the page and identifies table structure metadata.
    Actual rate values are parsed from pdfplumber table data by code.
    """
    skipped: bool = Field(
        default=False,
        description=(
            "True if this page contains no weight-based rate tables "
            "(e.g. zone charts, fee descriptions, Terms and Conditions, cover pages)."
        ),
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation if skipped=True.",
    )
    table_type: Optional[TableType] = Field(
        default=None,
        description="Category of rate table on this page. NULL if skipped=True.",
    )
    direction: Optional[Direction] = Field(
        default=None,
        description="'domestic', 'us_export', or 'us_import'. NULL if skipped=True.",
    )
    zone: Optional[str] = Field(
        default=None,
        description=(
            "Zone for the rates on this page, as printed. "
            "Examples: '2' for Zone 2, 'A' for Zone A. "
            "NULL for sameday table_type or skipped pages."
        ),
    )
    destination_region: Optional[str] = Field(
        default=None,
        description="Geographic region description, e.g. 'Puerto Rico', 'Canada'.",
    )
    services: list[ServiceInfo] = Field(
        default_factory=list,
        description=(
            "Service columns in the rate table, ordered left-to-right as they appear. "
            "Exclude header/weight columns — only include actual service name columns."
        ),
    )
    has_envelope_row: bool = Field(
        default=False,
        description="True if the table has a FedEx Envelope row (separate from main weight rows).",
    )
    has_pak_row: bool = Field(
        default=False,
        description="True if the table has a FedEx Pak row.",
    )
