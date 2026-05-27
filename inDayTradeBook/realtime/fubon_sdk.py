"""Version-aware Fubon SDK construction helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version_tuple(raw_version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in raw_version.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def get_fubon_neo_version() -> str:
    try:
        return version("fubon_neo")
    except PackageNotFoundError:
        return "0"


def create_fubon_sdk(ws_url: str = ""):
    """Create FubonSDK with the constructor style expected by the SDK version.

    SDK 2.2.1+ accepts ping/pong parameters before the optional URL. Older 2.x
    versions, including 2.0.1, use the bare constructor or url= only.
    """

    from fubon_neo.sdk import FubonSDK

    sdk_version = _version_tuple(get_fubon_neo_version())
    if sdk_version >= (2, 2, 1):
        if ws_url:
            return FubonSDK(30, 2, url=ws_url)
        return FubonSDK(30, 2)

    if ws_url:
        return FubonSDK(url=ws_url)
    return FubonSDK()
