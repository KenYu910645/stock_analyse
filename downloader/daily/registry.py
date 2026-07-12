from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


StatusFn = Callable[[str, str, int, str | None, str], object]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    run: Callable[[], object]
    path: str | None = None
    fail_fast: bool = False


def run_task_specs(tasks: Iterable[TaskSpec], status: StatusFn) -> list[object]:
    """Run independent updater tasks and convert uncaught exceptions into statuses."""

    results: list[object] = []
    for task in tasks:
        try:
            results.append(task.run())
        except Exception as exc:
            status(task.name, "failed", 0, task.path, str(exc))
            if task.fail_fast:
                raise
    return results
