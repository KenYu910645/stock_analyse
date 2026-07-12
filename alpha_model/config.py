"""Configuration loading for alpha model runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on local optional config stack.
    yaml = None


DEFAULT_CONFIG = {
    "data": {
        "price_dir": "data/price",
        "institutional_dir": "data/institutional",
        "metadata_path": "data/metadata.csv",
        "allow_unfiltered_universe": False,
        "date_col": "Date",
        "open_col": "open_adj",
        "high_col": "high_adj",
        "low_col": "low_adj",
        "close_col": "close_adj",
        "volume_col": "Capacity",
        "turnover_col": "Turnover",
    },
    "universe": {
        "min_price": 5,
        "min_avg_turnover_20d": 5_000_000,
        "min_history_days": 120,
    },
    "factor": {
        "kind": "momentum",
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
    "labels": {"method": "close_to_close", "horizons": [1, 5, 10, 20]},
    "evaluation": {
        "quantiles": 5,
        "primary_horizon": 5,
        "transaction_cost": {
            "enabled": False,
            "round_trip_cost": 0.006,
        },
    },
    "output": {"dir": "output/alpha_model/tables", "plot_dir": "data_viz/alpha_model/plots"},
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
        if yaml is not None:
            loaded = yaml.safe_load(config_file) or {}
        else:
            loaded = parse_simple_yaml(config_file.read())
    return merge_config(DEFAULT_CONFIG, loaded)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple YAML subset used by alpha model config files."""
    lines = [
        (len(line) - len(line.lstrip(" ")), line.strip())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][0] < indent:
            return {}, index
        if lines[index][1].startswith("- "):
            values = []
            while index < len(lines):
                current_indent, stripped = lines[index]
                if current_indent != indent or not stripped.startswith("- "):
                    break
                values.append(parse_scalar(stripped[2:].strip()))
                index += 1
            return values, index

        values: dict[str, Any] = {}
        while index < len(lines):
            current_indent, stripped = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in config line: {stripped}")
            if ":" not in stripped:
                raise ValueError(f"Invalid config line: {stripped}")
            key, raw_value = stripped.split(":", 1)
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                values[key] = parse_scalar(raw_value)
            else:
                values[key], index = parse_block(index, indent + 2)
        return values, index

    parsed, final_index = parse_block(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        raise ValueError("Could not parse complete alpha model config.")
    return parsed


def parse_scalar(value: str) -> Any:
    """Parse a scalar value from the config fallback parser."""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
