"""
Pipeline orchestrator.

Supports two execution modes:
  - sequential: pages processed one at a time (lowest API concurrency, easiest to debug)
  - parallel:   pages processed concurrently up to settings.concurrency limit

Architecture for parallel mode:
  - N async worker coroutines read + extract pages concurrently (LLM I/O bound)
  - 1 serial writer coroutine drains an asyncio.Queue into SQLite
    (avoids SQLite write contention; the queue decouples producers from the writer)

Every page that completes (success, skip, or failure) is recorded in the
processing_log table, enabling partial-run recovery via --pages CLI flag
or the skip_completed option.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from src.config import Settings
from src.db_writer import DBWriter
from src.llm_extractor import LLMExtractor
from src.models import PageExtraction
from src.page_formatter import format_page
from src.pdf_reader import PDFReader
from src.table_parser import parse_table

logger = logging.getLogger(__name__)

# Sentinel pushed to the write queue to signal "no more work"
_QUEUE_DONE = object()


@dataclass
class PageResult:
    page_number: int
    rows_inserted: int
    skipped: bool
    failed: bool
    error: str | None = None


@dataclass
class PipelineResult:
    pages_attempted: int = 0
    pages_succeeded: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    total_rows_inserted: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)

    def record(self, result: PageResult) -> None:
        self.pages_attempted += 1
        self.total_rows_inserted += result.rows_inserted
        if result.failed:
            self.pages_failed += 1
            if result.error:
                self.errors.append((result.page_number, result.error))
        elif result.skipped:
            self.pages_skipped += 1
        else:
            self.pages_succeeded += 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    settings: Settings,
    mode: Literal["sequential", "parallel"] = "sequential",
    skip_completed: bool = False,
) -> PipelineResult:
    """
    Extract rate tables from the PDF and write them to SQLite.

    Args:
        settings:        Application settings (paths, LLM config, etc.)
        mode:            'sequential' or 'parallel'
        skip_completed:  If True, pages already in processing_log with status='success'
                         are skipped without re-processing (partial-run recovery).

    Returns:
        PipelineResult with per-page outcome counts and any errors.
    """
    reader = PDFReader(settings.pdf_path)
    writer = DBWriter(settings.db_path, batch_size=settings.batch_size)
    extractor = LLMExtractor(settings)

    # Determine page range
    total = reader.total_pages
    start = settings.page_start
    end = settings.page_end if settings.page_end is not None else total
    end = min(end, total)
    page_numbers = list(range(start, end + 1))

    # Optionally skip already-completed pages
    if skip_completed:
        completed = writer.get_processed_pages()
        before = len(page_numbers)
        page_numbers = [p for p in page_numbers if p not in completed]
        skipped_count = before - len(page_numbers)
        if skipped_count:
            logger.info(
                "Skipping already-completed pages",
                extra={"skipped": skipped_count, "remaining": len(page_numbers)},
            )

    logger.info(
        "Pipeline starting",
        extra={
            "mode": mode,
            "pages": len(page_numbers),
            "range": f"{start}–{end}",
            "pdf": settings.pdf_path,
            "db": settings.db_path,
        },
    )

    pipeline_result = PipelineResult()

    try:
        if mode == "sequential":
            await _run_sequential(page_numbers, reader, extractor, writer, pipeline_result)
        else:
            await _run_parallel(
                page_numbers,
                reader,
                extractor,
                writer,
                pipeline_result,
                concurrency=settings.concurrency,
            )
    finally:
        reader.close()
        writer.close()

    logger.info(
        "Pipeline complete",
        extra={
            "succeeded": pipeline_result.pages_succeeded,
            "skipped": pipeline_result.pages_skipped,
            "failed": pipeline_result.pages_failed,
            "total_rows": pipeline_result.total_rows_inserted,
        },
    )
    return pipeline_result


# ---------------------------------------------------------------------------
# Sequential mode
# ---------------------------------------------------------------------------

async def _run_sequential(
    page_numbers: list[int],
    reader: PDFReader,
    extractor: LLMExtractor,
    writer: DBWriter,
    result: PipelineResult,
) -> None:
    for page_num in page_numbers:
        page_result = await _process_page(page_num, reader, extractor, writer)
        result.record(page_result)
        _log_progress(page_result, result)


# ---------------------------------------------------------------------------
# Parallel mode
# ---------------------------------------------------------------------------

async def _run_parallel(
    page_numbers: list[int],
    reader: PDFReader,
    extractor: LLMExtractor,
    writer: DBWriter,
    result: PipelineResult,
    concurrency: int,
) -> None:
    """
    Fan-out: N workers process pages concurrently (LLM calls are I/O-bound).
    Fan-in:  A single serial writer coroutine writes to SQLite (avoids contention).

    Flow:
        workers ──(PageExtraction)──► write_queue ──► db_writer_coroutine ──► SQLite
    """
    write_queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(page_num: int) -> None:
        async with semaphore:
            content = _read_page(page_num, reader)
            if content is None:
                writer.log_page(page_num, status="skipped", skip_reason="Empty page")
                await write_queue.put((page_num, PageExtraction(skipped=True, skip_reason="Empty page")))
                return

            page_formatted = format_page(content)
            classification = await extractor.aclassify_page(page_formatted, page_num)

            if classification.skipped:
                await write_queue.put((
                    page_num,
                    PageExtraction(
                        skipped=True,
                        skip_reason=classification.skip_reason or "LLM indicated no rate table",
                    ),
                ))
                return

            try:
                rates = parse_table(content.tables, classification, page_num, page_text=content.text)
                extraction = PageExtraction(rates=rates)
            except Exception as exc:
                msg = f"Table parsing failed: {exc}"
                logger.error("Table parsing error", extra={"page": page_num, "error": msg})
                extraction = PageExtraction(skipped=True, skip_reason=msg)

            await write_queue.put((page_num, extraction))

    async def db_writer_coroutine() -> dict[int, PageResult]:
        """Drain the queue and write to SQLite serially."""
        page_results: dict[int, PageResult] = {}
        while True:
            item = await write_queue.get()
            if item is _QUEUE_DONE:
                break
            page_num, extraction = item
            pr = _write_extraction(page_num, extraction, writer)
            page_results[page_num] = pr
            write_queue.task_done()
        return page_results

    # Start the DB writer coroutine
    writer_task = asyncio.create_task(db_writer_coroutine())

    # Run all page workers concurrently
    worker_tasks = [asyncio.create_task(worker(p)) for p in page_numbers]
    await asyncio.gather(*worker_tasks, return_exceptions=True)

    # Signal the writer that no more work is coming
    await write_queue.put(_QUEUE_DONE)
    page_results = await writer_task

    # Aggregate results
    for pr in page_results.values():
        result.record(pr)
        _log_progress(pr, result)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_page(page_num: int, reader: PDFReader):  # type: ignore[return]
    """Read a page. Returns PageContent or None for empty pages."""
    try:
        from src.pdf_reader import PageContent
        content = reader.read_page(page_num)
        if content.is_empty:
            return None
        return content
    except Exception as exc:
        logger.error(
            "Failed to read page",
            extra={"page": page_num, "error": str(exc)},
        )
        return None


async def _process_page(
    page_num: int,
    reader: PDFReader,
    extractor: LLMExtractor,
    writer: DBWriter,
) -> PageResult:
    """
    Hybrid extraction pipeline for a single page:
      1. Read pdfplumber data (text + raw tables)
      2. LLM classifies the page (table_type, zone, services) — small, fast response
      3. Code parses rate values from pdfplumber table matrix using classification
      4. Write to SQLite
    """
    content = _read_page(page_num, reader)

    if content is None:
        writer.log_page(page_num, status="skipped", skip_reason="Empty page")
        return PageResult(page_number=page_num, rows_inserted=0, skipped=True, failed=False)

    page_formatted = format_page(content)
    classification = await extractor.aclassify_page(page_formatted, page_num)

    if classification.skipped:
        writer.log_page(
            page_num,
            status="skipped",
            skip_reason=classification.skip_reason or "LLM indicated no rate table",
        )
        return PageResult(page_number=page_num, rows_inserted=0, skipped=True, failed=False)

    try:
        rates = parse_table(content.tables, classification, page_num, page_text=content.text)
    except Exception as exc:
        msg = f"Table parsing failed: {exc}"
        logger.error("Table parsing error", extra={"page": page_num, "error": msg})
        writer.log_page(page_num, status="failed", error_msg=msg)
        return PageResult(page_number=page_num, rows_inserted=0, skipped=False, failed=True, error=msg)

    extraction = PageExtraction(rates=rates)
    return _write_extraction(page_num, extraction, writer)


def _write_extraction(
    page_num: int, extraction: PageExtraction, writer: DBWriter
) -> PageResult:
    """Write a PageExtraction to the DB and log the result."""
    if extraction.skipped:
        writer.log_page(
            page_num,
            status="skipped",
            skip_reason=extraction.skip_reason or "LLM indicated no rate table",
        )
        return PageResult(page_number=page_num, rows_inserted=0, skipped=True, failed=False)

    if extraction.skip_reason and "failed" in (extraction.skip_reason or "").lower():
        writer.log_page(page_num, status="failed", error_msg=extraction.skip_reason)
        return PageResult(
            page_number=page_num,
            rows_inserted=0,
            skipped=False,
            failed=True,
            error=extraction.skip_reason,
        )

    try:
        inserted = writer.insert_batch(extraction.rates)
        writer.log_page(page_num, status="success", rows_inserted=inserted)
        return PageResult(
            page_number=page_num,
            rows_inserted=inserted,
            skipped=False,
            failed=False,
        )
    except Exception as exc:
        msg = str(exc)
        logger.error("DB write failed", extra={"page": page_num, "error": msg})
        writer.log_page(page_num, status="failed", error_msg=msg)
        return PageResult(
            page_number=page_num,
            rows_inserted=0,
            skipped=False,
            failed=True,
            error=msg,
        )


def _log_progress(page_result: PageResult, pipeline_result: PipelineResult) -> None:
    status = "failed" if page_result.failed else ("skipped" if page_result.skipped else "ok")
    logger.info(
        "Page processed",
        extra={
            "page": page_result.page_number,
            "status": status,
            "rows": page_result.rows_inserted,
            "total_attempted": pipeline_result.pages_attempted,
        },
    )
