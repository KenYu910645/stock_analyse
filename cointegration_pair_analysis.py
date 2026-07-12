"""Layer 2/3 pair analysis for high-correlation Taiwan stock pairs.

Layer 2: Engle-Granger style cointegration test on log prices.
Layer 3: Spread mean-reversion quality metrics.

The environment used by this project does not ship statsmodels, so the ADF
test is implemented directly as a first-version approximation:

    delta_spread[t] = c + gamma * spread[t-1] + error[t]

The reported ADF statistic is the t-statistic of gamma.  The p-value is an
approximation interpolated from common no-trend ADF critical values.  Use the
boolean pass/fail and the rank metrics as screening outputs, not final proof.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_correlation_analysis import (
    DEFAULT_METADATA_PATH,
    DEFAULT_PRICE_PATH,
    clean_price_data,
    load_metadata,
    load_price_data,
    make_price_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PAIR_PATH = PROJECT_ROOT / "output" / "price_correlation" / "raw" / "stock_pair_correlations_gt_0_5.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pair_trading" / "cointegration_gt_0_5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cointegration and spread mean-reversion checks for high-correlation stock pairs."
    )
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--pair-path", default=str(DEFAULT_PAIR_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-observations", type=int, default=252)
    parser.add_argument("--adf-pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--hit-lookahead", type=int, default=20)
    return parser.parse_args()


def ols_two_column(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """Fit y = alpha + beta*x and return alpha, beta, residual."""
    data = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    x_values = data["x"].to_numpy(dtype=float)
    y_values = data["y"].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x_values)), x_values])
    alpha, beta = np.linalg.lstsq(x_design, y_values, rcond=None)[0]
    residual = pd.Series(y_values - (alpha + beta * x_values), index=data.index)
    return float(alpha), float(beta), residual


def approximate_adf_pvalue(adf_stat: float) -> float:
    """Approximate ADF p-value from common no-trend critical values."""
    # Critical values are approximate large-sample ADF values with constant only.
    anchors = [
        (-4.00, 0.001),
        (-3.43, 0.010),
        (-2.86, 0.050),
        (-2.57, 0.100),
        (-1.95, 0.500),
        (-1.60, 0.800),
        (0.00, 0.990),
    ]
    if adf_stat <= anchors[0][0]:
        return anchors[0][1]
    if adf_stat >= anchors[-1][0]:
        return anchors[-1][1]

    for (x0, p0), (x1, p1) in zip(anchors, anchors[1:]):
        if x0 <= adf_stat <= x1:
            weight = (adf_stat - x0) / (x1 - x0)
            return float(p0 + weight * (p1 - p0))
    return 0.99


def adf_test_spread(spread: pd.Series) -> dict[str, float]:
    """Run a direct ADF-style regression on the spread."""
    clean = spread.dropna()
    lagged = clean.shift(1)
    delta = clean.diff()
    data = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
    if len(data) < 10:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_gamma": np.nan}

    y = data["delta"].to_numpy(dtype=float)
    x = data["lagged"].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    params = np.linalg.lstsq(x_design, y, rcond=None)[0]
    residual = y - x_design @ params
    dof = len(y) - x_design.shape[1]
    if dof <= 0:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "adf_gamma": float(params[1])}

    sigma2 = float((residual @ residual) / dof)
    cov = sigma2 * np.linalg.inv(x_design.T @ x_design)
    se_gamma = math.sqrt(float(cov[1, 1]))
    adf_stat = float(params[1] / se_gamma) if se_gamma else np.nan
    return {
        "adf_stat": adf_stat,
        "adf_pvalue": approximate_adf_pvalue(adf_stat) if pd.notna(adf_stat) else np.nan,
        "adf_gamma": float(params[1]),
    }


def calculate_half_life(spread: pd.Series) -> float:
    """Estimate spread half-life from delta spread regression."""
    clean = spread.dropna()
    lagged = clean.shift(1)
    delta = clean.diff()
    data = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
    if len(data) < 10:
        return np.nan

    x = data["lagged"].to_numpy(dtype=float)
    y = data["delta"].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])
    lambda_ = float(np.linalg.lstsq(x_design, y, rcond=None)[0][1])
    if lambda_ >= 0:
        return np.inf
    return float(-math.log(2) / lambda_)


def calculate_hurst_exponent(series: pd.Series) -> float:
    """Estimate Hurst exponent using lagged-difference scaling."""
    values = series.dropna().to_numpy(dtype=float)
    lags = np.array([2, 4, 8, 16, 32])
    usable_lags = lags[lags < len(values) / 2]
    if len(usable_lags) < 2:
        return np.nan

    tau = []
    lag_list = []
    for lag in usable_lags:
        diff = values[lag:] - values[:-lag]
        std = np.std(diff)
        if std > 0:
            tau.append(std)
            lag_list.append(lag)
    if len(tau) < 2:
        return np.nan
    slope = np.polyfit(np.log(lag_list), np.log(tau), 1)[0]
    return float(slope)


def calculate_zero_crossing_frequency(spread: pd.Series) -> float:
    """Return mean-centered zero crossings per valid transition."""
    centered = spread.dropna() - spread.dropna().mean()
    if len(centered) < 2:
        return np.nan
    signs = np.sign(centered)
    signs = signs.replace(0, np.nan).ffill().dropna()
    if len(signs) < 2:
        return np.nan
    crossings = (signs.iloc[1:].to_numpy() * signs.iloc[:-1].to_numpy()) < 0
    return float(crossings.sum() / len(crossings))


def calculate_hit_rate(
    spread: pd.Series,
    entry_z: float = 2.0,
    lookahead: int = 20,
) -> tuple[float, int]:
    """Measure how often extreme spread events cross the mean within lookahead."""
    clean = spread.dropna()
    std = clean.std(ddof=0)
    if len(clean) < lookahead + 2 or std == 0 or pd.isna(std):
        return np.nan, 0

    zscore = (clean - clean.mean()) / std
    entries = np.where(np.abs(zscore.to_numpy()) >= entry_z)[0]
    if len(entries) == 0:
        return np.nan, 0

    hits = 0
    usable_entries = 0
    values = (clean - clean.mean()).to_numpy()
    for idx in entries:
        if idx + 1 >= len(values):
            continue
        end = min(len(values), idx + lookahead + 1)
        if idx + 1 >= end:
            continue
        usable_entries += 1
        start_sign = np.sign(values[idx])
        future = values[idx + 1:end]
        if start_sign == 0 or np.any(np.sign(future) == 0) or np.any(np.sign(future) != start_sign):
            hits += 1
    if usable_entries == 0:
        return np.nan, 0
    return float(hits / usable_entries), usable_entries


def cointegration_test(price_a: pd.Series, price_b: pd.Series) -> dict[str, Any]:
    """Engle-Granger style two-step cointegration test."""
    log_a = np.log(price_a.where(price_a > 0))
    log_b = np.log(price_b.where(price_b > 0))
    data = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    alpha, beta, spread = ols_two_column(data["a"], data["b"])
    adf = adf_test_spread(spread)
    return {
        "alpha": alpha,
        "beta": beta,
        "spread": spread,
        **adf,
    }


def build_pair_analysis(
    price_wide: pd.DataFrame,
    pairs: pd.DataFrame,
    min_observations: int,
    adf_pvalue_threshold: float,
    entry_z: float,
    hit_lookahead: int,
) -> pd.DataFrame:
    """Run cointegration and spread quality checks for every pair."""
    rows = []
    for pair in pairs.itertuples(index=False):
        stock_a = str(pair.stock_id_1)
        stock_b = str(pair.stock_id_2)
        price_data = price_wide[[stock_a, stock_b]].dropna()
        observations = int(len(price_data))
        if observations < min_observations:
            continue

        result = cointegration_test(price_data[stock_a], price_data[stock_b])
        spread = result["spread"]
        spread_std = float(spread.std(ddof=0))
        latest_zscore = (
            float((spread.iloc[-1] - spread.mean()) / spread_std)
            if spread_std and pd.notna(spread_std)
            else np.nan
        )
        hit_rate, hit_events = calculate_hit_rate(
            spread,
            entry_z=entry_z,
            lookahead=hit_lookahead,
        )
        half_life = calculate_half_life(spread)
        hurst = calculate_hurst_exponent(spread)
        zero_cross = calculate_zero_crossing_frequency(spread)
        adf_pvalue = result["adf_pvalue"]
        cointegrated = bool(pd.notna(adf_pvalue) and adf_pvalue < adf_pvalue_threshold)

        rows.append(
            {
                "rank_raw_corr": pair.rank,
                "stock_id_1": stock_a,
                "stock_name_1": pair.stock_name_1,
                "industry_1": pair.industry_1,
                "stock_id_2": stock_b,
                "stock_name_2": pair.stock_name_2,
                "industry_2": pair.industry_2,
                "raw_correlation": pair.correlation,
                "cross_industry": pair.cross_industry,
                "observations": observations,
                "eg_alpha": result["alpha"],
                "eg_beta": result["beta"],
                "adf_stat": result["adf_stat"],
                "adf_pvalue_approx": adf_pvalue,
                "cointegration_pass_approx": cointegrated,
                "spread_mean": float(spread.mean()),
                "spread_std": spread_std,
                "latest_spread": float(spread.iloc[-1]),
                "latest_zscore": latest_zscore,
                "half_life_days": half_life,
                "hurst_exponent": hurst,
                "zero_crossing_frequency": zero_cross,
                "hit_rate_z2_cross_mean_20d": hit_rate,
                "hit_rate_event_count": hit_events,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    return result_df.sort_values(
        [
            "cointegration_pass_approx",
            "adf_pvalue_approx",
            "half_life_days",
            "raw_correlation",
        ],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def write_report(
    output_dir: Path,
    results: pd.DataFrame,
    adf_pvalue_threshold: float,
    entry_z: float,
    hit_lookahead: int,
) -> None:
    """Write a Chinese explanation/report for the analysis."""
    total = len(results)
    passed = int(results["cointegration_pass_approx"].sum()) if total else 0
    top = results[results["cointegration_pass_approx"]].head(20)
    suspicious = results[
        results["cointegration_pass_approx"]
        & results["half_life_days"].replace(np.inf, np.nan).between(2, 60)
        & results["hurst_exponent"].lt(0.5)
    ].head(30)

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_沒有符合條件的配對。_"
        cols = [
            "stock_name_1",
            "stock_name_2",
            "raw_correlation",
            "adf_pvalue_approx",
            "half_life_days",
            "hurst_exponent",
            "zero_crossing_frequency",
            "hit_rate_z2_cross_mean_20d",
        ]
        show = df[cols].copy()
        show.columns = [
            "股票A",
            "股票B",
            "原始相關",
            "ADF p值約略",
            "半衰期(日)",
            "Hurst",
            "過均值頻率",
            "Z=2回歸命中率",
        ]
        for col in ["原始相關", "ADF p值約略", "半衰期(日)", "Hurst", "過均值頻率", "Z=2回歸命中率"]:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
        lines = [
            "| " + " | ".join(show.columns) + " |",
            "| " + " | ".join(["---"] * len(show.columns)) + " |",
        ]
        for row in show.itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    report = f"""# Cointegration / Pair Trading 第二層分析報告

資料來源：`stock_pair_correlations_gt_0_5.csv`  
分析範圍：原始相關係數大於 `0.5` 的股票配對  
輸出資料夾：`output/pair_trading/cointegration_gt_0_5`

## 一、結果總覽

- 進行檢查的高相關配對數：`{total}`
- ADF p-value 約略小於 `{adf_pvalue_threshold}` 的配對數：`{passed}`
- 進一步符合較佳均值回歸條件的候選數：`{len(suspicious)}`

> 注意：目前環境沒有 `statsmodels`，所以本報告的 ADF p-value 是依照常見 ADF critical values 做的近似估計。正式交易前建議再用 `statsmodels.tsa.stattools.adfuller` 複核。

## 二、cointegration 到底是什麼？

一般 correlation 看的是「兩檔股票每天是否一起漲跌」。

但是 pair trading 更在意的是：

> 兩檔股票的價格差距，是否存在一個長期會拉回的均衡關係。

所以 cointegration 不是直接看報酬率是否相關，而是看 **log price 的線性組合是否穩定**。

以 A、B 兩檔股票為例：

```text
log(P_A) = alpha + beta * log(P_B) + epsilon
```

這裡的 `epsilon` 就是 spread。

如果 `epsilon` 是 stationary，意思是：

```text
spread 不會一直漂走
spread 偏離平均後，有傾向回到平均
```

這種 pair 才比較像 pair trading 的候選。

反過來說，如果兩檔股票 correlation 很高，但 spread 一路越走越遠，那它們可能只是一起漲或一起跌，卻不適合做均值回歸交易。

## 三、Engle-Granger two-step test 怎麼做？

### Step 1：用 log price 做回歸

```text
log(P_A) = alpha + beta * log(P_B) + residual
```

得到：

- `alpha`：截距
- `beta`：兩檔股票的長期價格比例關係
- `residual / spread`：A 相對 B 偏離長期關係的程度

### Step 2：檢查 residual 是否 stationary

本分析用 ADF 概念檢查：

```text
delta_spread[t] = c + gamma * spread[t-1] + error[t]
```

如果 `gamma` 顯著小於 0，代表 spread 偏離後有拉回傾向。

本版先用：

```text
ADF p-value < {adf_pvalue_threshold}
```

當作通過 cointegration 的第一層門檻。

## 四、Layer 3：均值回歸品質指標

cointegration 通過不代表一定能交易，還要看 spread 回得夠不夠好。

本次輸出包含：

### 1. spread z-score

```text
z = (spread - spread_mean) / spread_std
```

用來看目前 spread 偏離平均幾個標準差。

### 2. half-life

半衰期代表 spread 偏離後，理論上回到一半距離大約需要幾天。

太短可能很吵，太長資金占用時間太久。

### 3. Hurst exponent

粗略判讀：

- `H < 0.5`：偏均值回歸
- `H ≈ 0.5`：接近隨機漫步
- `H > 0.5`：偏趨勢延續

### 4. zero-crossing frequency

spread 穿越平均值的頻率。越高代表越常回到均值附近。

### 5. hit rate

本版定義：

```text
當 |z-score| >= {entry_z} 後，
未來 {hit_lookahead} 個交易日內是否穿越均值。
```

這是一個簡單的交易直覺檢查。

## 五、ADF 通過的前 20 組

{table(top)}

## 六、比較像 pair trading 候選的配對

以下條件只是初步篩選：

- ADF p-value 約略 < `{adf_pvalue_threshold}`
- half-life 介於 2 到 60 天
- Hurst exponent < 0.5

{table(suspicious)}

## 七、如何使用這份結果？

建議順序：

1. 先看 `cointegration_pass_approx = True`
2. 再看 `half_life_days` 是否合理
3. 再看 `hurst_exponent` 是否低於 0.5
4. 再看 `hit_rate_z2_cross_mean_20d`
5. 最後人工檢查這兩家公司是否有合理基本面關係

## 八、輸出檔案

- `cointegration_pair_results.csv`：全部 pair 的 Layer 2/3 結果
- `cointegration_pass_pairs.csv`：ADF p-value 約略通過的 pair
- `pair_trading_candidates.csv`：再加上 half-life / Hurst 條件後的候選
- `cointegration_explanation_report.md`：本說明報告
"""
    (output_dir / "cointegration_explanation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(
        args.pair_path,
        dtype={"stock_id_1": str, "stock_id_2": str},
    )
    metadata = load_metadata(args.metadata)
    raw_price = load_price_data(args.price_path)
    clean_df = clean_price_data(raw_price, metadata=metadata, common_only=True)
    price_wide = make_price_matrix(clean_df)

    results = build_pair_analysis(
        price_wide,
        pairs,
        min_observations=args.min_observations,
        adf_pvalue_threshold=args.adf_pvalue_threshold,
        entry_z=args.entry_z,
        hit_lookahead=args.hit_lookahead,
    )

    results_path = output_dir / "cointegration_pair_results.csv"
    pass_path = output_dir / "cointegration_pass_pairs.csv"
    candidate_path = output_dir / "pair_trading_candidates.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    passed = results[results["cointegration_pass_approx"]].copy()
    passed.to_csv(pass_path, index=False, encoding="utf-8-sig")

    candidates = passed[
        passed["half_life_days"].replace(np.inf, np.nan).between(2, 60)
        & passed["hurst_exponent"].lt(0.5)
    ].copy()
    candidates = candidates.sort_values(
        ["adf_pvalue_approx", "half_life_days", "hit_rate_z2_cross_mean_20d"],
        ascending=[True, True, False],
    )
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    write_report(
        output_dir,
        results,
        adf_pvalue_threshold=args.adf_pvalue_threshold,
        entry_z=args.entry_z,
        hit_lookahead=args.hit_lookahead,
    )

    print(f"Cointegration results written to {output_dir}.")
    print(f"Pairs tested: {len(results)}")
    print(f"Cointegration pass approx: {len(passed)}")
    print(f"Pair-trading candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
