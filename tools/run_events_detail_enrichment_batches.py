"""
Run resumable MOPS event detail enrichment in repeated bounded batches.

This is a thin control layer over downloader.events.enrich_existing_details().
It keeps each source request path, CSV schema, and failure-log behavior in the
canonical downloader while adding batch-level progress logging for long runs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downloader import events  # noqa: E402


DETAIL_FIELDS = [
    "FactDate",
    "Clause",
    "Description",
    "Spokesperson",
    "SpokespersonTitle",
    "SpokespersonPhone",
]

STATUS_COLUMNS = [
    "Timestamp",
    "Batch",
    "Phase",
    "CsvFiles",
    "UpdatedCsvFiles",
    "FullyUpdatedCsvFiles",
    "PartiallyUpdatedCsvFiles",
    "WaitingCsvFiles",
    "DetailKeyRows",
    "EnrichedRows",
    "RemainingRows",
    "FailureLogRows",
    "ReportPath",
]


@dataclass(frozen=True)
class EnrichmentStatus:
    csv_files: int
    updated_csv_files: int
    fully_updated_csv_files: int
    partially_updated_csv_files: int
    waiting_csv_files: int
    detail_key_rows: int
    enriched_rows: int
    failure_log_rows: int

    @property
    def remaining_rows(self) -> int:
        return self.detail_key_rows - self.enriched_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated resumable MOPS event detail-enrichment batches."
    )
    parser.add_argument("--metadata", default=str(events.METADATA_PATH))
    parser.add_argument("--output-dir", default=str(events.OUTPUT_DIR))
    parser.add_argument("--log-dir", default=str(events.LOG_DIR))
    parser.add_argument(
        "--universe",
        choices=["listed-stocks", "all-metadata"],
        default="all-metadata",
        help="Universe passed through to downloader.events. Default: all-metadata.",
    )
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument(
        "--end-date",
        default=events.DEFAULT_END_DATE.isoformat(),
        help="End date YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=12,
        help="Maximum number of enrichment batches to run unless --run-until-complete is set.",
    )
    parser.add_argument(
        "--run-until-complete",
        action="store_true",
        help="Keep running batches until no remaining detail rows or zero-progress stopping is reached.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Optional wall-clock runtime limit. Checked between batches.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print current enrichment status and exit without making network requests or edits.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Rows to request per batch.",
    )
    parser.add_argument("--detail-sleep-min", type=float, default=3.0)
    parser.add_argument("--detail-sleep-max", type=float, default=5.0)
    parser.add_argument("--detail-save-every", type=int, default=10)
    parser.add_argument("--detail-retries", type=int, default=1)
    parser.add_argument("--detail-timeout", type=float, default=12.0)
    parser.add_argument("--max-consecutive-detail-failures", type=int, default=8)
    parser.add_argument("--detail-failure-log", default=None)
    parser.add_argument("--retry-known-detail-failures", action="store_true")
    parser.add_argument(
        "--stop-after-zero-progress",
        type=int,
        default=2,
        help="Stop after this many consecutive batches add no enriched rows.",
    )
    parser.add_argument(
        "--status-log",
        default=None,
        help="Batch progress CSV. Default: logs/events/events_detail_batch_runner_<timestamp>.csv.",
    )
    return parser.parse_args()


def resolve_dates(args: argparse.Namespace, instruments: list[events.Instrument]) -> tuple[date, date]:
    end_date = events.parse_iso_date(args.end_date)
    metadata_start_dates = [
        instrument.start_date for instrument in instruments if instrument.start_date
    ]
    default_start = max(
        min(metadata_start_dates or [events.MOPS_HISTORY_START_DATE]),
        events.MOPS_HISTORY_START_DATE,
    )
    start_date = events.parse_iso_date(args.start_date) if args.start_date else default_start
    start_date = max(start_date, events.MOPS_HISTORY_START_DATE)
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}.")
    return start_date, end_date


def count_failure_log_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def count_status(output_dir: Path, detail_failure_log: Path) -> EnrichmentStatus:
    csv_files = 0
    updated_csv_files = 0
    fully_updated_csv_files = 0
    partially_updated_csv_files = 0
    waiting_csv_files = 0
    detail_key_rows = 0
    enriched_rows = 0
    for path in output_dir.glob("*.csv"):
        csv_files += 1
        file_detail_key_rows = 0
        file_enriched_rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("DetailSeqNo"):
                    file_detail_key_rows += 1
                    detail_key_rows += 1
                    if any((row.get(field) or "").strip() for field in DETAIL_FIELDS):
                        file_enriched_rows += 1
                        enriched_rows += 1
        if file_detail_key_rows <= 0:
            continue
        if file_enriched_rows <= 0:
            waiting_csv_files += 1
        else:
            updated_csv_files += 1
            if file_enriched_rows >= file_detail_key_rows:
                fully_updated_csv_files += 1
            else:
                partially_updated_csv_files += 1
    return EnrichmentStatus(
        csv_files=csv_files,
        updated_csv_files=updated_csv_files,
        fully_updated_csv_files=fully_updated_csv_files,
        partially_updated_csv_files=partially_updated_csv_files,
        waiting_csv_files=waiting_csv_files,
        detail_key_rows=detail_key_rows,
        enriched_rows=enriched_rows,
        failure_log_rows=count_failure_log_rows(detail_failure_log),
    )


def append_status(
    path: Path,
    batch: int,
    phase: str,
    status: EnrichmentStatus,
    report_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "Batch": batch,
                "Phase": phase,
                "CsvFiles": status.csv_files,
                "UpdatedCsvFiles": status.updated_csv_files,
                "FullyUpdatedCsvFiles": status.fully_updated_csv_files,
                "PartiallyUpdatedCsvFiles": status.partially_updated_csv_files,
                "WaitingCsvFiles": status.waiting_csv_files,
                "DetailKeyRows": status.detail_key_rows,
                "EnrichedRows": status.enriched_rows,
                "RemainingRows": status.remaining_rows,
                "FailureLogRows": status.failure_log_rows,
                "ReportPath": str(report_path or ""),
            }
        )


def print_status(status: EnrichmentStatus, prefix: str = "Status") -> None:
    percent = (
        status.enriched_rows / status.detail_key_rows * 100
        if status.detail_key_rows
        else 100.0
    )
    print(
        f"{prefix}: csv updated={status.updated_csv_files} "
        f"(full={status.fully_updated_csv_files}, partial={status.partially_updated_csv_files}) "
        f"waiting={status.waiting_csv_files} total_csv={status.csv_files}; "
        f"rows={status.enriched_rows}/{status.detail_key_rows} "
        f"remaining={status.remaining_rows} ({percent:.2f}%); "
        f"failures={status.failure_log_rows}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    detail_failure_log = (
        Path(args.detail_failure_log)
        if args.detail_failure_log
        else log_dir / "events_detail_failures.csv"
    )
    status_log = (
        Path(args.status_log)
        if args.status_log
        else log_dir
        / f"events_detail_batch_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    instruments = events.load_metadata(metadata_path, args.universe, args.codes)
    if not instruments:
        raise ValueError("No metadata instruments selected.")
    start_date, end_date = resolve_dates(args, instruments)

    if args.status_only:
        print_status(count_status(output_dir, detail_failure_log))
        return

    zero_progress_batches = 0
    started_at = time.monotonic()
    batch_limit = float("inf") if args.run_until_complete else args.batches
    print(
        "Running event detail-enrichment batches: "
        f"instruments={len(instruments)} range={start_date}..{end_date} "
        f"batches={'until-complete' if args.run_until_complete else args.batches} "
        f"chunk_size={args.chunk_size} status_log={status_log}",
        flush=True,
    )

    batch = 1
    while batch <= batch_limit:
        if args.max_runtime_minutes is not None:
            elapsed_minutes = (time.monotonic() - started_at) / 60
            if elapsed_minutes >= args.max_runtime_minutes:
                print(
                    f"Stopping after runtime limit: {elapsed_minutes:.1f} minutes.",
                    flush=True,
                )
                break

        before = count_status(output_dir, detail_failure_log)
        append_status(status_log, batch, "before", before)
        print_status(before, prefix=f"Before batch {batch}")
        if before.remaining_rows <= 0:
            print("No remaining detail rows; stopping.", flush=True)
            break

        report_path = events.enrich_existing_details(
            output_dir=output_dir,
            log_dir=log_dir,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            detail_sleep_min=args.detail_sleep_min,
            detail_sleep_max=args.detail_sleep_max,
            max_detail_rows=args.chunk_size,
            detail_save_every=args.detail_save_every,
            detail_retries=args.detail_retries,
            detail_timeout=args.detail_timeout,
            max_consecutive_detail_failures=args.max_consecutive_detail_failures,
            detail_failure_log=detail_failure_log,
            retry_known_detail_failures=args.retry_known_detail_failures,
        )

        after = count_status(output_dir, detail_failure_log)
        append_status(status_log, batch, "after", after, report_path)
        gained = after.enriched_rows - before.enriched_rows
        print(
            f"Batch {batch}: enriched +{gained}; "
            f"total={after.enriched_rows}/{after.detail_key_rows}; "
            f"remaining={after.remaining_rows}; failures={after.failure_log_rows}",
            flush=True,
        )
        print_status(after, prefix=f"After batch {batch}")

        if gained <= 0:
            zero_progress_batches += 1
            if zero_progress_batches >= args.stop_after_zero_progress:
                print(
                    f"Stopping after {zero_progress_batches} zero-progress batches.",
                    flush=True,
                )
                break
        else:
            zero_progress_batches = 0
        batch += 1


if __name__ == "__main__":
    main()
