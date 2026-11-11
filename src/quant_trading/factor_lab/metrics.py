from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable

import pandas as pd


def safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def winsorized_zscore(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([None] * len(values), index=values.index, dtype="float64")
    clipped = numeric.clip(numeric.quantile(lower_q), numeric.quantile(upper_q))
    std = clipped.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([0.0 if pd.notna(v) else None for v in clipped], index=values.index, dtype="float64")
    return (clipped - clipped.mean()) / std


def percentile_rank(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(pct=True, ascending=higher_is_better)
    return ranks * 100.0


def rank_ic(factors: Iterable[float], returns: Iterable[float]) -> float | None:
    df = pd.DataFrame({"factor": list(factors), "ret": list(returns)}).dropna()
    if len(df) < 3 or df["factor"].nunique() < 2 or df["ret"].nunique() < 2:
        return None
    value = df["factor"].rank().corr(df["ret"].rank(), method="pearson")
    return safe_float(value)


def ic_ir(values: Iterable[float]) -> float | None:
    clean = [float(v) for v in values if safe_float(v) is not None]
    if len(clean) < 2:
        return None
    std = pstdev(clean)
    if std == 0:
        return None
    return mean(clean) / std


def max_drawdown(returns: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        ret = safe_float(value) or 0.0
        equity *= 1.0 + ret
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def monotonic_quantiles(quantile_returns: dict[str, float | None]) -> bool:
    ordered = []
    for key in sorted(quantile_returns, key=lambda item: int(str(item).replace("q", ""))):
        value = safe_float(quantile_returns[key])
        if value is None:
            return False
        ordered.append(value)
    return all(left <= right for left, right in zip(ordered, ordered[1:]))


def verdict_from_metrics(rank_ic_mean: float | None, icir: float | None, top_return: float | None, monotonic: bool) -> str:
    if rank_ic_mean is None or icir is None or top_return is None:
        return "弃用"
    if rank_ic_mean > 0.03 and icir > 0.20 and top_return > 0 and monotonic:
        return "可用"
    if rank_ic_mean > 0 and top_return > 0:
        return "观察"
    return "弃用"
