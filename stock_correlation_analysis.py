"""Raw daily return correlation analysis for Taiwan listed stocks.

This first version intentionally uses raw log-return correlations.  It does
not remove market beta or estimate residual correlation.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from column_schema import read_csv_canonical


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "price"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_TAIEX_PATH = PROJECT_ROOT / "data" / "price" / "TAIEX_發行量加權股價指數.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_viz" / "price_correlation" / "raw"
DEFAULT_RESIDUAL_OUTPUT_DIR = PROJECT_ROOT / "data_viz" / "price_correlation" / "residual_market"
DEFAULT_REPRESENTATIVES_PER_INDUSTRY = 3
DEFAULT_MAX_INDUSTRY_HEATMAP_STOCKS = 120
DEFAULT_PER_STOCK_TOP_N = 15

COMMON_STOCK_TYPE = "\u80a1\u7968"
TWSE_MARKET = "\u4e0a\u5e02"

REPRESENTATIVE_STOCKS = {
    "Semiconductor": ["2330", "2454", "2303", "3034", "2379", "3711"],
    "AI Server / Electronics": ["2317", "2382", "3231", "6669", "2356", "2376"],
    "Financials": ["2881", "2882", "2891", "2886", "2884", "2892"],
    "Shipping": ["2603", "2609", "2615", "2618"],
    "Telecom": ["2412", "3045", "4904"],
    "Traditional / Defensive": ["1216", "2912", "1301", "1303", "2002", "1101", "1102"],
}

ROLLING_COMPARE_STOCKS = ["2454", "2303", "2317", "2881", "2412", "1216"]


@lru_cache(maxsize=1)
def _get_pyplot():
    """Load and configure Matplotlib only when a plot is requested."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as pyplot
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "SimHei",
    ]:
        if font_name in available_fonts:
            pyplot.rcParams["font.family"] = font_name
            break
    pyplot.rcParams["axes.unicode_minus"] = False
    return pyplot


def stock_id_from_path(csv_path: Path) -> str:
    """Extract stock id from filenames like 2330_台積電.csv."""
    return csv_path.stem.split("_", 1)[0]


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(value)).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def load_metadata(metadata_path: str | Path | None = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Load stock metadata indexed by stock id."""
    if metadata_path is None:
        return pd.DataFrame()

    path = Path(metadata_path)
    if not path.exists():
        return pd.DataFrame()

    metadata = read_csv_canonical(path, dtype={"Code": str})
    if "Code" not in metadata.columns:
        return pd.DataFrame()

    metadata["Code"] = metadata["Code"].astype(str)
    return metadata.drop_duplicates("Code").set_index("Code", drop=False)


def get_listed_common_codes(metadata: pd.DataFrame) -> set[str] | None:
    """Return listed common stock codes when metadata can support the filter."""
    required = {"Code", "Type", "Market"}
    if metadata.empty or not required.issubset(metadata.columns):
        return None

    filtered = metadata[
        metadata["Type"].eq(COMMON_STOCK_TYPE)
        & metadata["Market"].eq(TWSE_MARKET)
    ]
    return set(filtered["Code"].astype(str))


def load_price_data(path: str | Path) -> pd.DataFrame:
    """Load a long price dataset or a directory of per-stock adjusted CSVs."""
    price_path = Path(path)
    if price_path.is_dir():
        frames = []
        for csv_path in sorted(price_path.glob("*.csv")):
            if csv_path.name.startswith("twse_price_"):
                continue
            df = read_csv_canonical(csv_path)
            df["stock_id"] = stock_id_from_path(csv_path)
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No price CSV files found in {price_path}.")
        return pd.concat(frames, ignore_index=True)

    if not price_path.exists():
        raise FileNotFoundError(f"Price data path does not exist: {price_path}")
    return read_csv_canonical(price_path, dtype={"stock_id": str, "Code": str})


def load_market_returns(path: str | Path) -> pd.Series:
    """Load market index daily log returns from a TAIEX-style CSV."""
    market_path = Path(path)
    if not market_path.exists():
        raise FileNotFoundError(f"Market data path does not exist: {market_path}")

    df = read_csv_canonical(market_path)
    required = {"Date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Market CSV missing required columns: {sorted(missing)}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = (
        df.dropna(subset=["Date", "Close"])
        .loc[lambda data: data["Close"].gt(0)]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    market_returns = np.log(df.set_index("Date")["Close"]).diff()
    market_returns.name = "market_return"
    return market_returns.dropna()


def clean_price_data(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    common_only: bool = True,
) -> pd.DataFrame:
    """Clean long price data into date, stock_id, stock_name, industry, adj_close."""
    working = df.copy()
    column_map = {}
    if "Date" in working.columns:
        column_map["Date"] = "date"
    if "Code" in working.columns and "stock_id" not in working.columns:
        column_map["Code"] = "stock_id"
    if "close_adj" in working.columns and "adj_close" not in working.columns:
        column_map["close_adj"] = "adj_close"
    elif "Close_adj" in working.columns and "adj_close" not in working.columns:
        column_map["Close_adj"] = "adj_close"
    elif "Close" in working.columns and "adj_close" not in working.columns:
        column_map["Close"] = "adj_close"
    if "Turnover" in working.columns and "turnover" not in working.columns:
        column_map["Turnover"] = "turnover"
    if "Name" in working.columns and "stock_name" not in working.columns:
        column_map["Name"] = "stock_name"
    if "Group" in working.columns and "industry" not in working.columns:
        column_map["Group"] = "industry"

    working = working.rename(columns=column_map)
    required = {"date", "stock_id", "adj_close"}
    missing = required - set(working.columns)
    if missing:
        raise ValueError(f"Price data missing required columns: {sorted(missing)}")

    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["stock_id"] = working["stock_id"].astype(str).str.strip()
    working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")

    keep_columns = ["date", "stock_id", "adj_close"]
    if "turnover" in working.columns:
        keep_columns.append("turnover")
    for optional in ["stock_name", "industry"]:
        if optional in working.columns:
            keep_columns.append(optional)

    working = working[keep_columns]
    working = working.dropna(subset=["date", "stock_id", "adj_close"])
    working = working[working["adj_close"].gt(0)]
    if "turnover" in working.columns:
        working["turnover"] = pd.to_numeric(working["turnover"], errors="coerce")

    if metadata is not None and not metadata.empty:
        if common_only:
            allowed_codes = get_listed_common_codes(metadata)
            if allowed_codes is not None:
                working = working[working["stock_id"].isin(allowed_codes)]

        metadata_cols = [
            col for col in ["Name", "Group"] if col in metadata.columns
        ]
        if metadata_cols:
            meta = metadata[metadata_cols].rename(
                columns={"Name": "stock_name_meta", "Group": "industry_meta"}
            )
            working = working.merge(
                meta,
                left_on="stock_id",
                right_index=True,
                how="left",
            )
            if "stock_name" not in working.columns:
                working["stock_name"] = working["stock_name_meta"]
            else:
                working["stock_name"] = working["stock_name"].fillna(
                    working["stock_name_meta"]
                )
            if "industry" not in working.columns:
                working["industry"] = working["industry_meta"]
            else:
                working["industry"] = working["industry"].fillna(
                    working["industry_meta"]
                )
            working = working.drop(
                columns=[
                    col
                    for col in ["stock_name_meta", "industry_meta"]
                    if col in working.columns
                ]
            )

    for optional in ["stock_name", "industry"]:
        if optional not in working.columns:
            working[optional] = ""
        working[optional] = working[optional].fillna("").astype(str)

    return (
        working.sort_values(["date", "stock_id"])
        .drop_duplicates(["date", "stock_id"], keep="last")
        .reset_index(drop=True)
    )


def make_price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned long price data into a wide adjusted-close matrix."""
    return (
        df.pivot(index="date", columns="stock_id", values="adj_close")
        .sort_index()
        .sort_index(axis=1)
    )


def calculate_log_returns(
    price_wide: pd.DataFrame,
    min_valid_ratio: float = 0.8,
) -> pd.DataFrame:
    """Calculate daily log returns and drop sparse stock columns."""
    if not 0 < min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio must be in the interval (0, 1].")

    ret = np.log(price_wide).diff()
    min_valid_count = int(len(ret) * min_valid_ratio)
    return ret.dropna(axis=1, thresh=min_valid_count)


def make_turnover_matrix(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned long price data into a wide turnover matrix."""
    if "turnover" not in clean_df.columns:
        raise ValueError("Turnover data is required for weighted group returns.")

    return (
        clean_df.pivot(index="date", columns="stock_id", values="turnover")
        .sort_index()
        .sort_index(axis=1)
    )


def calculate_turnover_weighted_group_returns(
    ret: pd.DataFrame,
    turnover_wide: pd.DataFrame,
    stock_info: pd.DataFrame,
    min_valid_members: int = 2,
) -> pd.DataFrame:
    """Calculate daily industry returns weighted by stock trading turnover."""
    if min_valid_members <= 0:
        raise ValueError("min_valid_members must be positive.")
    if stock_info.empty or "industry" not in stock_info.columns:
        raise ValueError("Stock industry metadata is required.")

    aligned_turnover = turnover_wide.reindex(index=ret.index, columns=ret.columns)
    available_stocks = [
        stock
        for stock in ret.columns.astype(str)
        if stock in stock_info.index
    ]
    industries = (
        stock_info.loc[available_stocks, "industry"]
        .replace("", "Unknown")
        .fillna("Unknown")
    )

    group_returns = {}
    for industry in sorted(industries.unique()):
        stocks = industries[industries.eq(industry)].index.tolist()
        stock_returns = ret[stocks]
        weights = aligned_turnover[stocks].where(stock_returns.notna())
        weights = weights.where(weights.gt(0))

        weighted_sum = (stock_returns * weights).sum(axis=1, min_count=1)
        weight_sum = weights.sum(axis=1, min_count=1)
        member_count = stock_returns.notna().sum(axis=1)
        group_return = weighted_sum / weight_sum
        group_return = group_return.where(member_count.ge(min_valid_members))
        group_returns[str(industry)] = group_return

    return pd.DataFrame(group_returns).sort_index(axis=1)


def calculate_market_residual_returns(
    ret: pd.DataFrame,
    market_ret: pd.Series,
    stock_info: pd.DataFrame | None = None,
    min_observations: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regress each stock return on market return and return residuals/betas."""
    if min_observations <= 1:
        raise ValueError("min_observations must be greater than 1.")

    market = market_ret.rename("market_return")
    residual_series_by_stock = {}
    beta_rows = []

    for stock_id in ret.columns.astype(str):
        aligned = pd.concat([ret[stock_id], market], axis=1, join="inner")
        aligned.columns = ["stock_return", "market_return"]
        aligned = aligned.dropna()
        observations = int(len(aligned))
        if observations < min_observations:
            continue

        x = aligned["market_return"].astype(float)
        y = aligned["stock_return"].astype(float)
        x_var = float(x.var(ddof=0))
        if np.isclose(x_var, 0):
            continue

        beta = float(((x - x.mean()) * (y - y.mean())).mean() / x_var)
        alpha = float(y.mean() - beta * x.mean())
        fitted = alpha + beta * x
        residual = y - fitted
        residual_series_by_stock[stock_id] = residual

        ss_res = float((residual**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan
        info = stock_info.loc[stock_id] if stock_info is not None and stock_id in stock_info.index else {}
        beta_rows.append(
            {
                "stock_id": stock_id,
                "stock_name": info.get("stock_name", ""),
                "industry": info.get("industry", ""),
                "alpha": alpha,
                "beta_market": beta,
                "r_squared": r_squared,
                "observations": observations,
            }
        )

    beta_df = pd.DataFrame(beta_rows).sort_values("stock_id").reset_index(drop=True)
    residuals = pd.DataFrame(residual_series_by_stock).reindex(ret.index).sort_index()
    residuals = residuals.dropna(axis=1, how="all")
    return residuals, beta_df


def plot_group_correlation_heatmap(
    group_corr: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot direct industry-to-industry correlation heatmap."""
    plt = _get_pyplot()
    size = max(10, min(22, 0.42 * len(group_corr) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(group_corr.values, cmap="RdYlBu_r", vmin=-0.2, vmax=0.8)
    ax.set_title("Turnover-Weighted Industry Return Correlation Heatmap", pad=14)
    ax.set_xticks(np.arange(len(group_corr.columns)))
    ax.set_xticklabels(group_corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(group_corr.index)))
    ax.set_yticklabels(group_corr.index, fontsize=8)
    ax.tick_params(length=0)

    for row in range(len(group_corr)):
        for col in range(len(group_corr.columns)):
            value = group_corr.iat[row, col]
            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Daily log-return correlation")
    fig.tight_layout()
    _save_or_show(output_path)


def cluster_correlation_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Return a correlation matrix reordered by hierarchical clustering."""
    if len(corr) < 2:
        return corr

    clean_corr = corr.fillna(0).clip(-1, 1)
    distance = np.sqrt(0.5 * (1 - clean_corr))
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0)
    condensed = squareform(distance_values, checks=False)
    clusters = linkage(condensed, method="ward")
    order = leaves_list(clusters)
    return corr.iloc[order, order]


def flatten_stock_groups(stock_groups: dict[str, list[str]]) -> list[str]:
    """Return stocks in group order without duplicates."""
    seen = set()
    ordered = []
    for stocks in stock_groups.values():
        for stock in stocks:
            if stock not in seen:
                ordered.append(stock)
                seen.add(stock)
    return ordered


def get_representative_stock_list(ret: pd.DataFrame) -> list[str]:
    """Return representative stocks that exist in the return matrix."""
    available = set(ret.columns.astype(str))
    return [
        stock
        for stock in flatten_stock_groups(REPRESENTATIVE_STOCKS)
        if stock in available
    ]


def get_industry_representative_stock_groups(
    ret: pd.DataFrame,
    stock_info: pd.DataFrame,
    representatives_per_industry: int = DEFAULT_REPRESENTATIVES_PER_INDUSTRY,
) -> dict[str, list[str]]:
    """Pick representative stocks from every industry available in returns."""
    if representatives_per_industry <= 0:
        raise ValueError("representatives_per_industry must be positive.")
    if stock_info.empty or "industry" not in stock_info.columns:
        return {"Representative": get_representative_stock_list(ret)}

    available = set(ret.columns.astype(str))
    info = stock_info[stock_info["stock_id"].isin(available)].copy().reset_index(drop=True)
    info["industry"] = info["industry"].replace("", "Unknown").fillna("Unknown")
    if "avg_turnover" not in info.columns:
        info["avg_turnover"] = np.nan
    info["return_observations"] = info["stock_id"].map(ret.notna().sum()).fillna(0)
    info = info.sort_values(
        ["industry", "avg_turnover", "return_observations", "stock_id"],
        ascending=[True, False, False, True],
    )

    groups = {}
    for industry, group_df in info.groupby("industry", sort=True):
        selected = group_df.head(representatives_per_industry)["stock_id"].tolist()
        if selected:
            groups[str(industry)] = selected
    return groups


def get_industry_stock_groups(
    ret: pd.DataFrame,
    stock_info: pd.DataFrame,
) -> dict[str, list[str]]:
    """Return every available stock grouped by industry."""
    if stock_info.empty or "industry" not in stock_info.columns:
        return {}

    available = set(ret.columns.astype(str))
    info = stock_info[stock_info["stock_id"].isin(available)].copy().reset_index(drop=True)
    info["industry"] = info["industry"].replace("", "Unknown").fillna("Unknown")
    info["return_observations"] = info["stock_id"].map(ret.notna().sum()).fillna(0)
    if "avg_turnover" not in info.columns:
        info["avg_turnover"] = np.nan
    info = info.sort_values(
        ["industry", "avg_turnover", "return_observations", "stock_id"],
        ascending=[True, False, False, True],
    )
    return {
        str(industry): group_df["stock_id"].tolist()
        for industry, group_df in info.groupby("industry", sort=True)
    }


def calculate_correlation_matrix(ret: pd.DataFrame, stocks: list[str]) -> pd.DataFrame:
    """Calculate pairwise correlations for selected stocks."""
    available = [stock for stock in stocks if stock in ret.columns]
    if len(available) < 2:
        raise ValueError("At least two selected stocks are required for correlation.")
    return ret[available].corr()


def _save_or_show(output_path: str | Path | None) -> None:
    plt = _get_pyplot()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    group_boundaries: Iterable[int] = (),
    annotate_cells: bool | None = None,
) -> None:
    plt = _get_pyplot()
    size = max(8, min(22, 0.38 * len(corr) + 4))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-0.2, vmax=0.8)
    ax.set_title(title, pad=14)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.tick_params(length=0)

    for boundary in group_boundaries:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.0)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.0)

    should_annotate = len(corr) <= 30 if annotate_cells is None else annotate_cells
    if should_annotate:
        font_size = 6 if len(corr) <= 30 else max(3, min(5, int(180 / len(corr))))
        for row in range(len(corr)):
            for col in range(len(corr.columns)):
                value = corr.iat[row, col]
                if pd.notna(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Daily log-return correlation")
    fig.tight_layout()
    _save_or_show(output_path)


def plot_industry_sorted_heatmap(
    corr: pd.DataFrame,
    stock_groups: dict[str, list[str]],
    output_path: str | Path | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    """Plot a heatmap ordered by manual industry groups with boundaries."""
    ordered = [
        stock
        for stock in flatten_stock_groups(stock_groups)
        if stock in corr.index and stock in corr.columns
    ]
    corr_ordered = corr.loc[ordered, ordered]

    boundaries = []
    position = 0
    for stocks in stock_groups.values():
        present = [stock for stock in stocks if stock in corr_ordered.index]
        if present:
            position += len(present)
            boundaries.append(position)
    boundaries = boundaries[:-1]

    if label_map:
        corr_ordered = label_correlation_matrix(corr_ordered, label_map)

    _plot_heatmap(
        corr_ordered,
        "Raw Daily Return Correlation Heatmap - Industry Sorted",
        output_path,
        boundaries,
    )


def plot_clustered_heatmap(
    corr: pd.DataFrame,
    output_path: str | Path | None = None,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Cluster stocks by correlation distance, plot, and return reordered corr."""
    clean_corr = corr.fillna(0).clip(-1, 1)
    distance = np.sqrt(0.5 * (1 - clean_corr))
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0)
    condensed = squareform(distance_values, checks=False)
    clusters = linkage(condensed, method="ward")
    order = leaves_list(clusters)
    corr_ordered = corr.iloc[order, order]
    corr_for_plot = (
        label_correlation_matrix(corr_ordered, label_map)
        if label_map
        else corr_ordered
    )
    _plot_heatmap(
        corr_for_plot,
        "Raw Daily Return Correlation Heatmap - Clustered",
        output_path,
    )
    return corr_ordered


def build_stock_info(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return one metadata row per stock from cleaned price data."""
    columns = ["stock_id", "stock_name", "industry"]
    base = clean_df[columns].drop_duplicates("stock_id").set_index("stock_id", drop=False)
    if "turnover" in clean_df.columns:
        avg_turnover = clean_df.groupby("stock_id")["turnover"].mean()
        base["avg_turnover"] = avg_turnover
    return base


def build_stock_label_map(stock_info: pd.DataFrame) -> dict[str, str]:
    """Return stock-id to Chinese-name labels, falling back to the stock id."""
    if stock_info.empty or "stock_name" not in stock_info.columns:
        return {}

    labels = {}
    for stock_id, row in stock_info.iterrows():
        stock_name = str(row.get("stock_name", "")).strip()
        labels[str(stock_id)] = stock_name or str(stock_id)
    return labels


def label_correlation_matrix(
    corr: pd.DataFrame,
    label_map: dict[str, str],
) -> pd.DataFrame:
    """Rename correlation matrix axes from stock ids to display labels."""
    return corr.rename(index=label_map, columns=label_map)


def rank_correlation_to_target(
    ret: pd.DataFrame,
    target: str = "2330",
    top_n: int = 20,
    stock_info: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank stocks by daily return correlation to the target stock."""
    if target not in ret.columns:
        raise ValueError(f"Target stock {target} is not present in return data.")

    ranking = ret.corrwith(ret[target]).drop(labels=[target], errors="ignore")
    ranking = ranking.dropna().sort_values(ascending=False)

    result = ranking.rename(f"correlation_to_{target}").reset_index()
    result = result.rename(columns={"index": "stock_id"})
    result["stock_id"] = result["stock_id"].astype(str)

    if stock_info is not None and not stock_info.empty:
        info = stock_info[["stock_name", "industry"]].copy()
        result = result.merge(info, left_on="stock_id", right_index=True, how="left")
    else:
        result["stock_name"] = ""
        result["industry"] = ""

    ordered_cols = ["stock_id", "stock_name", "industry", f"correlation_to_{target}"]
    result = result[ordered_cols]
    return result.head(top_n).copy(), result.tail(top_n).sort_values(
        f"correlation_to_{target}"
    ).copy()


def plot_top_correlation_bar(
    corr_df: pd.DataFrame,
    target: str = "2330",
    output_path: str | Path | None = None,
) -> None:
    """Plot top correlations to the target stock as a horizontal bar chart."""
    plt = _get_pyplot()
    corr_col = f"correlation_to_{target}"
    if corr_col not in corr_df.columns:
        raise ValueError(f"Correlation column missing: {corr_col}")

    plot_df = corr_df.sort_values(corr_col, ascending=True).copy()
    labels = plot_df["stock_id"].astype(str)
    if "stock_name" in plot_df.columns:
        names = plot_df["stock_name"].fillna("").astype(str)
        labels = labels.where(names.eq(""), names)

    height = max(6, 0.36 * len(plot_df) + 2)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, plot_df[corr_col], color="#2563eb")
    ax.set_title(f"Top {len(plot_df)} Stocks Correlated with {target} Daily Return")
    ax.set_xlabel(f"Correlation with {target}")
    ax.set_ylabel("Stock ID / Name")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_or_show(output_path)


def calculate_rolling_correlation(
    ret: pd.DataFrame,
    target: str,
    compare_stocks: list[str],
    window: int = 60,
) -> pd.DataFrame:
    """Calculate rolling correlation between target and comparison stocks."""
    if target not in ret.columns:
        raise ValueError(f"Target stock {target} is not present in return data.")

    result = pd.DataFrame(index=ret.index)
    for stock in compare_stocks:
        if stock in ret.columns:
            result[stock] = ret[target].rolling(window).corr(ret[stock])
    return result


def plot_rolling_correlation(
    rolling_corr: pd.DataFrame,
    target: str = "2330",
    output_path: str | Path | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    """Plot rolling correlations as line charts."""
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df = rolling_corr.rename(columns=label_map or {})
    target_label = (label_map or {}).get(target, target)
    plot_df.plot(ax=ax, linewidth=1.6)
    ax.set_title(f"Rolling Correlation with {target_label} Daily Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling correlation")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_or_show(output_path)


def plot_correlation_distribution(
    corr: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Plot the distribution of pairwise correlations from the upper triangle."""
    plt = _get_pyplot()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    values = corr.where(mask).stack().dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=30, color="#0f766e", edgecolor="white")
    ax.set_title("Pairwise Daily Return Correlation Distribution")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Stock pair count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_or_show(output_path)


def get_top_correlated_peers(
    corr: pd.DataFrame,
    target: str,
    top_n: int = DEFAULT_PER_STOCK_TOP_N,
) -> pd.Series:
    """Return the top correlated peer stocks for one target stock."""
    if target not in corr.columns:
        raise ValueError(f"Target stock {target} is not present in correlation matrix.")

    return (
        corr[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )


def plot_top_peer_matrix_heatmap(
    target: str,
    matrix: pd.DataFrame,
    label_map: dict[str, str],
    output_path: str | Path,
) -> None:
    """Plot one target stock and top peers as a square correlation heatmap."""
    plt = _get_pyplot()
    target_label = label_map.get(target, target)
    labels = [label_map.get(str(stock), str(stock)) for stock in matrix.index]

    size = max(7, 0.35 * len(matrix) + 4)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.values, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_title(
        f"{target_label} and Top {len(matrix) - 1} Correlated Stocks",
        pad=12,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)

    for row in range(len(matrix)):
        for col in range(len(matrix.columns)):
            value = matrix.iat[row, col]
            if pd.notna(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Daily log-return correlation")
    fig.tight_layout()
    _save_or_show(output_path)


def write_per_stock_top_correlation_heatmaps(
    corr: pd.DataFrame,
    stock_info: pd.DataFrame,
    label_map: dict[str, str],
    output_dir: Path,
    top_n: int = DEFAULT_PER_STOCK_TOP_N,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Generate one top-peer heatmap per stock and return the manifest table."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    heatmap_dir = output_dir / "per_stock_top_correlation_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in corr.columns.astype(str):
        peers = get_top_correlated_peers(corr, target, top_n=top_n)
        if peers.empty:
            continue

        target_info = stock_info.loc[target] if target in stock_info.index else {}
        png_path = heatmap_dir / f"{target}_{safe_filename(label_map.get(target, target))}.png"
        matrix_stocks = [target] + peers.index.astype(str).tolist()
        peer_matrix = corr.loc[matrix_stocks, matrix_stocks]
        if overwrite or not png_path.exists():
            plot_top_peer_matrix_heatmap(target, peer_matrix, label_map, png_path)

        matrix_path = heatmap_dir / f"{target}_{safe_filename(label_map.get(target, target))}.csv"
        peer_matrix.rename(index=label_map, columns=label_map).to_csv(
            matrix_path,
            encoding="utf-8-sig",
        )

        for rank, (peer, value) in enumerate(peers.items(), start=1):
            peer_info = stock_info.loc[peer] if peer in stock_info.index else {}
            rows.append(
                {
                    "target_stock_id": target,
                    "target_stock_name": target_info.get("stock_name", ""),
                    "target_industry": target_info.get("industry", ""),
                    "rank": rank,
                    "peer_stock_id": peer,
                    "peer_stock_name": peer_info.get("stock_name", ""),
                    "peer_industry": peer_info.get("industry", ""),
                    "correlation": value,
                    "heatmap_path": str(png_path.relative_to(output_dir)),
                    "matrix_path": str(matrix_path.relative_to(output_dir)),
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        output_dir / f"per_stock_top{top_n}_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


def order_by_mean_peer_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """Order stocks by average correlation to same-matrix peers, descending."""
    if corr.empty:
        return corr

    peer_values = corr.to_numpy(copy=True)
    np.fill_diagonal(peer_values, np.nan)
    peer_corr = pd.DataFrame(peer_values, index=corr.index, columns=corr.columns)
    scores = peer_corr.mean(axis=1, skipna=True)
    ordered = scores.sort_values(ascending=False).index.tolist()
    return corr.loc[ordered, ordered]


def write_industry_representatives(
    output_dir: Path,
    representative_groups: dict[str, list[str]],
    stock_info: pd.DataFrame,
) -> None:
    """Persist selected representatives for auditability."""
    rows = []
    for industry, stocks in representative_groups.items():
        for rank, stock_id in enumerate(stocks, start=1):
            info = stock_info.loc[stock_id] if stock_id in stock_info.index else {}
            rows.append(
                {
                    "industry": industry,
                    "rank": rank,
                    "stock_id": stock_id,
                    "stock_name": info.get("stock_name", ""),
                    "avg_turnover": info.get("avg_turnover", np.nan),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "industry_representative_stocks.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_within_industry_heatmaps(
    ret: pd.DataFrame,
    industry_groups: dict[str, list[str]],
    label_map: dict[str, str],
    output_dir: Path,
    max_stocks_per_heatmap: int = DEFAULT_MAX_INDUSTRY_HEATMAP_STOCKS,
) -> pd.DataFrame:
    """Generate all-company correlation heatmaps within each industry."""
    heatmap_dir = output_dir / "industry_heatmaps"
    matrix_dir = output_dir / "industry_matrices_by_name"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for industry, stocks in industry_groups.items():
        available = [stock for stock in stocks if stock in ret.columns]
        if len(available) < 2:
            continue
        if len(available) > max_stocks_per_heatmap:
            available = available[:max_stocks_per_heatmap]

        corr = cluster_correlation_matrix(ret[available].corr())
        corr_by_name = label_correlation_matrix(corr, label_map)
        filename = safe_filename(industry)
        csv_path = matrix_dir / f"{filename}.csv"
        png_path = heatmap_dir / f"{filename}.png"
        _plot_heatmap(
            corr_by_name,
            f"{industry} Daily Return Correlation Heatmap",
            png_path,
            annotate_cells=True,
        )
        try:
            corr_by_name.to_csv(csv_path, encoding="utf-8-sig")
        except PermissionError:
            csv_path = matrix_dir / f"{filename}_peer_sorted.csv"
            corr_by_name.to_csv(csv_path, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "industry": industry,
                "stock_count": len(available),
                "heatmap_path": str(png_path.relative_to(output_dir)),
                "matrix_path": str(csv_path.relative_to(output_dir)),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("industry")
    manifest.to_csv(
        output_dir / "industry_heatmap_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


def write_interpretation(
    output_dir: Path,
    target: str = "2330",
    representatives_per_industry: int = DEFAULT_REPRESENTATIVES_PER_INDUSTRY,
) -> None:
    """Write brief interpretation notes alongside the generated artifacts."""
    text = f"""# Raw Stock Correlation Analysis

This first version uses daily log-return correlations. It does not remove
market beta, TSMC beta, or other common factors.

## Cross-Industry Heatmaps

`industry_sorted_correlation_heatmap.png` selects up to
{representatives_per_industry} liquid representative companies from every
listed-stock industry. It lets you compare how industries move against each
other without plotting all 800+ companies at once.

`clustered_correlation_heatmap.png` uses the same representative companies,
but lets hierarchical clustering reorder them by return similarity.

## Within-Industry Heatmaps

The `industry_heatmaps/` folder contains one heatmap per industry using all
available companies in that industry. These charts are better for checking
which companies inside one group move together.

## TSMC Ranking

Correlation with `{target}` can be read roughly as:

- `corr > 0.6`: very high correlation
- `0.4 ~ 0.6`: high correlation
- `0.2 ~ 0.4`: moderate correlation
- `0.0 ~ 0.2`: low correlation
- `corr < 0.0`: weak negative correlation, needs validation

## Rolling Correlation

Rolling correlation checks whether relationships are stable. If `{target}` and
another stock become more correlated only during semiconductor bull markets,
the relationship is regime-dependent.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate raw daily return correlation analysis outputs."
    )
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--market-path", default=str(DEFAULT_TAIEX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target", default="2330")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-valid-ratio", type=float, default=0.8)
    parser.add_argument(
        "--representatives-per-industry",
        type=int,
        default=DEFAULT_REPRESENTATIVES_PER_INDUSTRY,
        help="Number of representative companies to select from each industry.",
    )
    parser.add_argument(
        "--max-industry-heatmap-stocks",
        type=int,
        default=DEFAULT_MAX_INDUSTRY_HEATMAP_STOCKS,
        help="Safety cap per within-industry heatmap.",
    )
    parser.add_argument(
        "--include-non-common",
        action="store_true",
        help="Do not filter metadata to TWSE listed common stocks.",
    )
    parser.add_argument(
        "--residual-market",
        action="store_true",
        help="Analyze stock residual returns after regressing on TAIEX returns.",
    )
    parser.add_argument(
        "--min-residual-observations",
        type=int,
        default=120,
        help="Minimum aligned stock/market observations for residual regression.",
    )
    return parser.parse_args()


def generate_correlation_outputs(
    returns: pd.DataFrame,
    clean_df: pd.DataFrame,
    stock_info: pd.DataFrame,
    label_map: dict[str, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Generate all correlation outputs for a return-like matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    turnover_wide = make_turnover_matrix(clean_df)
    group_returns = calculate_turnover_weighted_group_returns(
        returns,
        turnover_wide,
        stock_info,
    )
    group_returns.to_csv(
        output_dir / "turnover_weighted_industry_returns.csv",
        encoding="utf-8-sig",
    )
    group_corr = group_returns.corr()
    group_corr.to_csv(
        output_dir / "turnover_weighted_industry_correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    plot_group_correlation_heatmap(
        group_corr,
        output_dir / "turnover_weighted_industry_correlation_heatmap.png",
    )
    clustered_group_corr = cluster_correlation_matrix(group_corr)
    clustered_group_corr.to_csv(
        output_dir / "turnover_weighted_industry_correlation_matrix_clustered.csv",
        encoding="utf-8-sig",
    )
    plot_group_correlation_heatmap(
        clustered_group_corr,
        output_dir / "turnover_weighted_industry_correlation_heatmap_clustered.png",
    )

    representative_groups = get_industry_representative_stock_groups(
        returns,
        stock_info,
        representatives_per_industry=args.representatives_per_industry,
    )
    industry_groups = get_industry_stock_groups(returns, stock_info)
    selected_stocks = flatten_stock_groups(representative_groups)
    corr_selected = calculate_correlation_matrix(returns, selected_stocks)
    corr_selected.to_csv(
        output_dir / "selected_stock_correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    label_correlation_matrix(corr_selected, label_map).to_csv(
        output_dir / "selected_stock_correlation_matrix_by_name.csv",
        encoding="utf-8-sig",
    )

    plot_industry_sorted_heatmap(
        corr_selected,
        representative_groups,
        output_dir / "industry_sorted_correlation_heatmap.png",
        label_map=label_map,
    )
    corr_clustered = plot_clustered_heatmap(
        corr_selected,
        output_dir / "clustered_correlation_heatmap.png",
        label_map=label_map,
    )
    corr_clustered.to_csv(
        output_dir / "clustered_selected_stock_correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    label_correlation_matrix(corr_clustered, label_map).to_csv(
        output_dir / "clustered_selected_stock_correlation_matrix_by_name.csv",
        encoding="utf-8-sig",
    )
    write_industry_representatives(output_dir, representative_groups, stock_info)
    industry_manifest = plot_within_industry_heatmaps(
        returns,
        industry_groups,
        label_map,
        output_dir,
        max_stocks_per_heatmap=args.max_industry_heatmap_stocks,
    )

    top, bottom = rank_correlation_to_target(
        returns,
        target=args.target,
        top_n=args.top_n,
        stock_info=stock_info,
    )
    top.to_csv(
        output_dir / f"top{args.top_n}_corr_with_{args.target}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bottom.to_csv(
        output_dir / f"bottom{args.top_n}_corr_with_{args.target}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_top_correlation_bar(
        top,
        target=args.target,
        output_path=output_dir / f"top{args.top_n}_corr_with_{args.target}_bar.png",
    )
    rolling_corr = calculate_rolling_correlation(
        returns,
        target=args.target,
        compare_stocks=ROLLING_COMPARE_STOCKS,
        window=60,
    )
    rolling_corr.to_csv(output_dir / f"rolling_corr_with_{args.target}.csv", encoding="utf-8-sig")
    rolling_corr.rename(columns=label_map).to_csv(
        output_dir / f"rolling_corr_with_{args.target}_by_name.csv",
        encoding="utf-8-sig",
    )
    plot_rolling_correlation(
        rolling_corr,
        target=args.target,
        output_path=output_dir / f"rolling_corr_with_{args.target}.png",
        label_map=label_map,
    )
    plot_correlation_distribution(
        corr_selected,
        output_path=output_dir / "correlation_distribution.png",
    )
    full_stock_corr = returns.corr()
    per_stock_manifest = write_per_stock_top_correlation_heatmaps(
        full_stock_corr,
        stock_info,
        label_map,
        output_dir,
        top_n=DEFAULT_PER_STOCK_TOP_N,
    )
    write_interpretation(
        output_dir,
        target=args.target,
        representatives_per_industry=args.representatives_per_industry,
    )

    print(f"Correlation analysis outputs written to {output_dir}.")
    print(f"Selected stocks: {len(selected_stocks)}")
    print(f"Industries represented: {len(representative_groups)}")
    print(f"Within-industry heatmaps: {len(industry_manifest)}")
    print(f"Industry correlation groups: {group_corr.shape[0]}")
    print(f"Per-stock top correlation rows: {len(per_stock_manifest)}")
    print(f"Return matrix stocks after filtering: {returns.shape[1]}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.residual_market and str(output_dir) == str(DEFAULT_OUTPUT_DIR):
        output_dir = DEFAULT_RESIDUAL_OUTPUT_DIR

    metadata = load_metadata(args.metadata)
    df = load_price_data(args.price_path)
    clean_df = clean_price_data(
        df,
        metadata=metadata,
        common_only=not args.include_non_common,
    )
    stock_info = build_stock_info(clean_df)
    label_map = build_stock_label_map(stock_info)
    price_wide = make_price_matrix(clean_df)
    returns = calculate_log_returns(price_wide, min_valid_ratio=args.min_valid_ratio)

    if args.residual_market:
        output_dir.mkdir(parents=True, exist_ok=True)
        market_returns = load_market_returns(args.market_path)
        returns, beta_df = calculate_market_residual_returns(
            returns,
            market_returns,
            stock_info=stock_info,
            min_observations=args.min_residual_observations,
        )
        beta_df.to_csv(
            output_dir / "market_residual_beta.csv",
            index=False,
            encoding="utf-8-sig",
        )
        returns.to_csv(
            output_dir / "market_residual_returns.csv",
            encoding="utf-8-sig",
        )
        print(f"Market residual beta values written to {output_dir / 'market_residual_beta.csv'}.")

    generate_correlation_outputs(
        returns=returns,
        clean_df=clean_df,
        stock_info=stock_info,
        label_map=label_map,
        output_dir=output_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
