"""Configuration loading for alpha model runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "data": {
        "price_dir": "data/adj_price",
        "metadata_path": "data/metadata.csv",
        "date_col": "Date",
        "open_col": "Open_adj",
        "high_col": "High_adj",
        "low_col": "Low_adj",
        "close_col": "Close_adj",
        "volume_col": "Capacity",
        "turnover_col": "Turnover",
    },
    "universe": {
        "min_price": 5,
        "min_avg_turnover_20d": 5_000_000,
        "min_history_days": 120,
    },
    "factor": {
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "zscore": True,
        "names": [
            "momentum_20d",
            "momentum_60d",
            "momentum_120d",
            "momentum_60_5",
            "momentum_120_20",
            "vol_adj_momentum_60d",
        ],
    },
    "labels": {"horizons": [1, 5, 10, 20]},
    "evaluation": {
        "quantiles": 5,
        "primary_horizon": 5,
        "transaction_cost": {
            "enabled": False,
            "round_trip_cost": 0.006,
        },
    },
    "output": {"dir": "output/alpha_model/tables", "plot_dir": "output/alpha_model/plots"},
}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override values into base config."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and merge it with defaults."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    return merge_config(DEFAULT_CONFIG, loaded)
