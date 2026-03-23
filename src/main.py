"""
CLI entry point for the FedEx table extractor.

Commands:
  extract   Extract rate tables from the PDF into SQLite
  stats     Show database statistics
  query     Query rates from the database
  inspect   Preview what pdfplumber sees on a specific page (debugging aid)

Usage examples:
  python -m src.main extract --pages 13
  python -m src.main extract --pages 13-124 --mode parallel
  python -m src.main extract --mode parallel --skip-completed
  python -m src.main stats
  python -m src.main query --service "FedEx Priority Overnight" --zone 5 --weight 10
  python -m src.main inspect --page 13
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

import click

# Configure structured logging before importing anything from src
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.option("--debug", is_flag=True, default=False, help="Enable DEBUG logging")
def cli(debug: bool) -> None:
    """FedEx rate table extractor — PDF → SQLite via LLM."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# extract command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--pages",
    default=None,
    help=(
        "Page range to process. Examples: '13' (single), '13-124' (range). "
        "Defaults to full PDF."
    ),
)
@click.option(
    "--mode",
    type=click.Choice(["sequential", "parallel"]),
    default="sequential",
    show_default=True,
    help="Processing mode. Use 'parallel' for faster runs (requires stable API access).",
)
@click.option(
    "--skip-completed",
    is_flag=True,
    default=False,
    help="Skip pages already recorded as 'success' in processing_log (partial-run recovery).",
)
@click.option("--pdf", default=None, help="Path to PDF file (overrides PDF_PATH env var)")
@click.option("--db", default=None, help="Path to SQLite DB (overrides DB_PATH env var)")
def extract(
    pages: Optional[str],
    mode: str,
    skip_completed: bool,
    pdf: Optional[str],
    db: Optional[str],
) -> None:
    """Extract rate tables from the FedEx Service Guide PDF into SQLite."""
    from src.config import Settings

    # Build settings, applying CLI overrides
    overrides: dict = {}
    if pdf:
        overrides["pdf_path"] = pdf
    if db:
        overrides["db_path"] = db
    if pages:
        start, end = _parse_page_range(pages)
        overrides["page_start"] = start
        if end is not None:
            overrides["page_end"] = end

    settings = Settings(**overrides)

    click.echo(
        f"Extracting pages {settings.page_start}–{settings.page_end or 'end'} "
        f"| mode={mode} | provider={settings.llm_provider}/{settings.llm_model}"
    )

    from src.pipeline import run_pipeline

    result = asyncio.run(
        run_pipeline(settings, mode=mode, skip_completed=skip_completed)  # type: ignore[arg-type]
    )

    click.echo("\n=== Extraction complete ===")
    click.echo(f"  Pages attempted : {result.pages_attempted}")
    click.echo(f"  Succeeded       : {result.pages_succeeded}")
    click.echo(f"  Skipped         : {result.pages_skipped}")
    click.echo(f"  Failed          : {result.pages_failed}")
    click.echo(f"  Rows inserted   : {result.total_rows_inserted}")

    if result.errors:
        click.echo("\nFailed pages:")
        for page_num, err in result.errors:
            click.echo(f"  Page {page_num}: {err}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default=None, help="Path to SQLite DB (overrides DB_PATH env var)")
@click.option("--json-output", is_flag=True, default=False, help="Output as JSON")
def stats(db: Optional[str], json_output: bool) -> None:
    """Show extraction statistics from the SQLite database."""
    from src.config import settings as default_settings
    from src.db_writer import DBWriter

    db_path = db or default_settings.db_path

    with DBWriter(db_path) as writer:
        s = writer.get_stats()

    if json_output:
        click.echo(json.dumps({
            "total_rows": s.total_rows,
            "pages_processed": s.pages_processed,
            "pages_skipped": s.pages_skipped,
            "pages_failed": s.pages_failed,
            "rows_by_table_type": s.rows_by_table_type,
            "rows_by_service_category": s.rows_by_service_category,
        }, indent=2))
        return

    click.echo("\n=== Database Statistics ===")
    click.echo(f"  Total rows      : {s.total_rows:,}")
    click.echo(f"  Pages succeeded : {s.pages_processed}")
    click.echo(f"  Pages skipped   : {s.pages_skipped}")
    click.echo(f"  Pages failed    : {s.pages_failed}")

    if s.rows_by_table_type:
        click.echo("\n  Rows by table type:")
        for ttype, count in sorted(s.rows_by_table_type.items(), key=lambda x: -x[1]):
            click.echo(f"    {ttype:<25} {count:>8,}")

    if s.rows_by_service_category:
        click.echo("\n  Rows by service category:")
        for cat, count in sorted(s.rows_by_service_category.items(), key=lambda x: -x[1]):
            click.echo(f"    {cat:<35} {count:>8,}")


# ---------------------------------------------------------------------------
# query command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--service", default=None, help="Filter by service name (partial match)")
@click.option("--zone", default=None, help="Filter by zone (exact match, e.g. '5' or 'A')")
@click.option("--weight", default=None, type=float, help="Filter by exact weight in lbs")
@click.option("--table-type", default=None, help="Filter by table type (exact match)")
@click.option("--limit", default=50, show_default=True, help="Maximum rows to return")
@click.option("--db", default=None, help="Path to SQLite DB (overrides DB_PATH env var)")
@click.option("--json-output", is_flag=True, default=False, help="Output as JSON")
def query(
    service: Optional[str],
    zone: Optional[str],
    weight: Optional[float],
    table_type: Optional[str],
    limit: int,
    db: Optional[str],
    json_output: bool,
) -> None:
    """Query extracted rates from the SQLite database."""
    from src.config import settings as default_settings
    from src.db_writer import DBWriter

    db_path = db or default_settings.db_path

    with DBWriter(db_path) as writer:
        rows = writer.query_rates(
            service_name=service,
            zone=zone,
            weight_lbs=weight,
            table_type=table_type,
            limit=limit,
        )

    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    if not rows:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(rows)} result(s):\n")
    for row in rows:
        price = f"${row['price_usd']:.2f}" if row["price_usd"] is not None else "N/A"
        weight_str = (
            f"{row['weight_lbs']} lbs"
            if row["weight_lbs"] is not None
            else f"{row['weight_range_min_lbs']}–{row['weight_range_max_lbs'] or '∞'} lbs"
        )
        click.echo(
            f"  p{row['source_page']:>3} | {row['table_type']:<22} | "
            f"{row['service_name']:<35} | zone={row['zone'] or 'N/A':<4} | "
            f"{weight_str:<15} | {price} ({row['price_type']})"
        )


# ---------------------------------------------------------------------------
# inspect command (debugging aid)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--page", required=True, type=int, help="1-indexed page number to inspect")
@click.option("--pdf", default=None, help="Path to PDF file (overrides PDF_PATH env var)")
def inspect(page: int, pdf: Optional[str]) -> None:
    """Preview the raw text and table content pdfplumber extracts from a page."""
    from src.config import settings as default_settings
    from src.page_formatter import format_page
    from src.pdf_reader import PDFReader

    pdf_path = pdf or default_settings.pdf_path

    with PDFReader(pdf_path) as reader:
        content = reader.read_page(page)

    click.echo(f"\n=== RAW TEXT (page {page}) ===\n{content.text}\n")
    click.echo(f"=== TABLES FOUND: {len(content.tables)} ===")
    for i, table in enumerate(content.tables, 1):
        click.echo(f"\n--- Table {i} ({len(table)} rows × {len(table[0]) if table else 0} cols) ---")
        for row in table[:10]:
            click.echo(f"  {row}")
        if len(table) > 10:
            click.echo(f"  ... ({len(table) - 10} more rows)")

    click.echo(f"\n=== FORMATTED FOR LLM ===\n")
    click.echo(format_page(content))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_page_range(pages_str: str) -> tuple[int, Optional[int]]:
    """
    Parse a page range string.
    '13'       → (13, None)  [single page: end=None means use page 13 only]
    '13-124'   → (13, 124)
    """
    pages_str = pages_str.strip()
    if "-" in pages_str:
        parts = pages_str.split("-", 1)
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if end < start:
                raise click.BadParameter(f"End page {end} is less than start page {start}")
            return start, end
        except ValueError:
            raise click.BadParameter(f"Invalid page range: '{pages_str}'. Use '13' or '13-124'.")
    else:
        try:
            page = int(pages_str)
            return page, page
        except ValueError:
            raise click.BadParameter(f"Invalid page number: '{pages_str}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
