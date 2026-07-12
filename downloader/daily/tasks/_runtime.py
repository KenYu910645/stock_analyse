from __future__ import annotations

from typing import Any


class _CoreProxy:
    """Resolve compatibility dependencies from the updater module on demand.

    The lazy lookup avoids an import cycle while preserving the public
    `downloader.update_all_data` monkeypatch surface used by existing callers and
    tests. Task modules reference dependencies explicitly as ``core.<name>``.
    """

    def __getattr__(self, name: str) -> Any:
        from downloader import update_all_data

        return getattr(update_all_data, name)


core = _CoreProxy()
