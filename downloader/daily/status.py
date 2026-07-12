from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO


FAILED_ACTIONS = {"failed", "unexpected_source_payload"}


@dataclass(frozen=True)
class UpdateResult:
    dataset: str
    action: str
    rows: int = 0
    path: str | None = None
    note: str = ""

    @property
    def failed(self) -> bool:
        return self.action in FAILED_ACTIONS

    def format_line(self) -> str:
        parts = [f"{self.dataset}: {self.action}", f"rows={self.rows}"]
        if self.path:
            parts.append(f"path={self.path}")
        if self.note:
            parts.append(self.note)
        return " | ".join(parts)


class StatusCollector:
    """Collect structured updater results while preserving the old stdout format."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream
        self.records: list[UpdateResult] = []

    def emit(
        self,
        dataset: str,
        action: str,
        rows: int = 0,
        path: str | None = None,
        note: str = "",
    ) -> UpdateResult:
        result = UpdateResult(dataset, action, rows, path, note)
        self.records.append(result)
        print(result.format_line(), file=self.stream, flush=True)
        return result

    def has_failures(self) -> bool:
        return any(record.failed for record in self.records)

    def clear(self) -> None:
        self.records.clear()
