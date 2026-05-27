"""Environment-driven configuration for the realtime collector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a runtime dependency.
    load_dotenv = None


DEFAULT_SYMBOLS = ["2330", "2308", "3105"]
DEFAULT_CHANNELS = ["trades", "books"]


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)

    return [item.strip() for item in value.split(",") if item.strip()]


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class RealtimeConfig:
    fubon_id: str
    fubon_password: str
    fubon_cert_path: str
    fubon_cert_password: str
    fubon_mode: str
    fubon_ws_url: str

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    symbols: list[str]
    channels: list[str]
    reconnect_delay_seconds: float = 3.0

    @property
    def postgres_dsn(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        host = self.postgres_host
        port = self.postgres_port
        database = quote_plus(self.postgres_db)
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    @property
    def fubon_cert_password_for_login(self) -> str | None:
        return self.fubon_cert_password or None

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "RealtimeConfig":
        if load_dotenv is not None:
            load_dotenv(env_file)

        return cls(
            fubon_id=_getenv("FUBON_ID"),
            fubon_password=_getenv("FUBON_PASSWORD"),
            fubon_cert_path=_getenv("FUBON_CERT_PATH"),
            fubon_cert_password=_getenv("FUBON_CERT_PASSWORD"),
            fubon_mode=_getenv("FUBON_MODE", "Speed"),
            fubon_ws_url=_getenv("FUBON_WS_URL"),
            postgres_host=_getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(_getenv("POSTGRES_PORT", "5432")),
            postgres_db=_getenv("POSTGRES_DB", "market_data"),
            postgres_user=_getenv("POSTGRES_USER", "postgres"),
            postgres_password=_getenv("POSTGRES_PASSWORD", "postgres"),
            symbols=_split_csv(os.getenv("SYMBOLS"), DEFAULT_SYMBOLS),
            channels=_split_csv(os.getenv("CHANNELS"), DEFAULT_CHANNELS),
            reconnect_delay_seconds=float(_getenv("RECONNECT_DELAY_SECONDS", "3")),
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("FUBON_ID", self.fubon_id),
                ("FUBON_PASSWORD", self.fubon_password),
                ("FUBON_CERT_PATH", self.fubon_cert_path),
            )
            if not value
        ]

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        unsupported_channels = sorted(set(self.channels) - {"trades", "books"})
        if unsupported_channels:
            raise ValueError(
                "Unsupported realtime channels for this collector: "
                + ", ".join(unsupported_channels)
            )
