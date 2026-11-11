"""Top5 opportunity-retrieval metrics and promotion gates for v8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..factor_lab.metrics import max_drawdown


V8_TOP_N = 5
V8_RECALL_N = 50
V11_STABLE_SLOTS = 3
V11_AGGRESSIVE_SLOTS = 2


@dataclass(frozen=True)
class V8GateConfig:
    min_positive_fold_rate: float = 0.60
    required_drawdown_improvement: float = 0.25
    v5_reference_drawdown: float = -0.128136
    required_all_loss_improvement: float = 0.10


def diversified_top5(
    frame: pd.DataFrame,
    *,
    score_column: str = "prediction",
    top_n: int = V8_TOP_N,
    recall_n: int = V8_RECALL_N,
) -> pd.DataFrame:
    """Select a diverse short list after unrestricted full-market ranking."""
    if frame.empty:
        return frame.copy()
    ranked = frame.sort_values(score_column, ascending=False).head(recall_n).copy()
    selected: list[int] = []
    industries: set[str] = set()
    for index, row in ranked.iterrows():
        industry = str(row.get("industry") or "UNKNOWN")
        if industry in industries:
            continue
        selected.append(index)
        industries.add(industry)
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        selected_set = set(selected)
        selected.extend(index for index in ranked.index if index not in selected_set and len(selected) < top_n)
    result = ranked.loc[selected].copy()
    result["candidate_rank"] = range(1, len(result) + 1)
    return result


def dual_head_top5(
    frame: pd.DataFrame,
    *,
    stable_score_column: str = "v11_stable_score",
    aggressive_score_column: str = "v11_aggressive_score",
    recall_n: int = V8_RECALL_N,
    stable_slots: int = V11_STABLE_SLOTS,
    aggressive_slots: int = V11_AGGRESSIVE_SLOTS,
) -> pd.DataFrame:
    """Select a 70/30-style Top5 after unrestricted full-market stable/aggressive scoring."""
    if frame.empty:
        return frame.copy()
    required = {stable_score_column, aggressive_score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v11 dual-head selection missing columns: {missing}")
    working = frame.copy()
    working[stable_score_column] = pd.to_numeric(working[stable_score_column], errors="coerce")
    working[aggressive_score_column] = pd.to_numeric(working[aggressive_score_column], errors="coerce")
    working = working[working[stable_score_column].notna() | working[aggressive_score_column].notna()]
    selected: list[int] = []
    industries: set[str] = set()

    def add_from(column: str, slots: int, candidate_type: str) -> None:
        ranked = working.sort_values(column, ascending=False).head(recall_n)
        for index, row in ranked.iterrows():
            if index in selected:
                continue
            industry = str(row.get("industry") or "UNKNOWN")
            if industry in industries:
                continue
            selected.append(index)
            industries.add(industry)
            working.loc[index, "candidate_type"] = candidate_type
            if sum(working.loc[item, "candidate_type"] == candidate_type for item in selected) >= slots:
                break

    add_from(stable_score_column, stable_slots, "stable")
    add_from(aggressive_score_column, aggressive_slots, "aggressive")
    if len(selected) < stable_slots + aggressive_slots:
        fallback = working.assign(
            _fallback_score=working[[stable_score_column, aggressive_score_column]].rank(pct=True).mean(axis=1)
        ).sort_values("_fallback_score", ascending=False).head(recall_n)
        for index, row in fallback.iterrows():
            if index in selected:
                continue
            selected.append(index)
            working.loc[index, "candidate_type"] = "fallback"
            if len(selected) >= stable_slots + aggressive_slots:
                break
    result = working.loc[selected].copy()
    result["candidate_rank"] = range(1, len(result) + 1)
    return result


def evaluate_top5(
    frame: pd.DataFrame,
    *,
    score_column: str = "prediction",
    selection_mode: str = "diversified",
) -> dict[str, Any]:
    required = {
        "trade_date",
        "ts_code",
        score_column,
        "label_5d_net_return",
        "label_10d_mfe_close",
        "label_10d_hit_5pct",
        "label_20d_hit_8pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v8 Top5 metrics missing columns: {missing}")
    daily_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        working = group.copy()
        working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
        working = working[working[score_column].notna()]
        if selection_mode == "v11_dual_70_30":
            selected = dual_head_top5(working)
        elif selection_mode == "diversified":
            selected = diversified_top5(working, score_column=score_column)
        else:
            raise ValueError(f"Unsupported Top5 selection mode: {selection_mode}")
        if selected.empty:
            continue
        fixed = pd.to_numeric(selected["label_5d_net_return"], errors="coerce")
        market_fixed = pd.to_numeric(working["label_5d_net_return"], errors="coerce")
        mfe = pd.to_numeric(selected["label_10d_mfe_close"], errors="coerce")
        market_mfe = pd.to_numeric(working["label_10d_mfe_close"], errors="coerce")
        hit5 = _bool_series(selected["label_10d_hit_5pct"])
        hit8 = _bool_series(selected["label_20d_hit_8pct"])
        truth = working.assign(_truth=market_mfe).sort_values("_truth", ascending=False).head(V8_TOP_N)
        truth_codes = set(truth["ts_code"].astype(str))
        selected_codes = set(selected["ts_code"].astype(str))
        recall = len(truth_codes & selected_codes) / max(len(truth_codes), 1)
        ndcg = _ndcg_at_k(selected["label_10d_mfe_close"], truth["label_10d_mfe_close"], V8_TOP_N)
        best_selected = float(mfe.max()) if mfe.notna().any() else None
        best_market = float(market_mfe.max()) if market_mfe.notna().any() else None
        daily_rows.append(
            {
                "trade_date": str(trade_date),
                "candidate_count": int(len(selected)),
                "mean_net_return_5d": float(fixed.mean()) if fixed.notna().any() else None,
                "market_mean_net_return_5d": float(market_fixed.mean()) if market_fixed.notna().any() else None,
                "best_net_return_5d": float(fixed.max()) if fixed.notna().any() else None,
                "best_mfe_10d": best_selected,
                "hit5pct_10d": bool(hit5.any()),
                "hit8pct_20d": bool(hit8.any()),
                "all_loss_5d": bool(fixed.notna().any() and (fixed < 0).all()),
                "recall_at5": float(recall),
                "ndcg_at5": ndcg,
                "regret_at5": best_market - best_selected if best_market is not None and best_selected is not None else None,
                "industry_count": int(selected.get("industry", pd.Series(dtype=str)).fillna("UNKNOWN").nunique()),
            }
        )
        selection_columns = [
            name
            for name in (
                "ts_code",
                "industry",
                score_column,
                "candidate_rank",
                "model_disagreement",
                "label_5d_net_return",
                "label_10d_mfe_close",
                "label_10d_mae_close",
                "label_10d_hit_5pct",
                "label_20d_hit_8pct",
                "candidate_type",
            )
            if name in selected.columns
        ]
        for row in selected[selection_columns].to_dict("records"):
            selections.append({"trade_date": str(trade_date), **row})
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {"days": 0, "daily": [], "selections": []}
    returns = pd.to_numeric(daily["mean_net_return_5d"], errors="coerce").fillna(0.0)
    summary = {
        "days": int(len(daily)),
        "mean_net_return_at5": _mean(daily["mean_net_return_5d"]),
        "best_return_at5": _mean(daily["best_net_return_5d"]),
        "best_mfe_10d_at5": _mean(daily["best_mfe_10d"]),
        "positive_day_rate": round(float((returns > 0).mean()), 6),
        "hit5pct_10d_at5": round(float(daily["hit5pct_10d"].mean()), 6),
        "hit8pct_20d_at5": round(float(daily["hit8pct_20d"].mean()), 6),
        "all_loss_rate_at5": round(float(daily["all_loss_5d"].mean()), 6),
        "recall_at5": _mean(daily["recall_at5"]),
        "ndcg_at5": _mean(daily["ndcg_at5"]),
        "regret_at5": _mean(daily["regret_at5"]),
        "avg_industry_count": _mean(daily["industry_count"]),
        "max_drawdown": round(float(max_drawdown(returns.tolist())), 6),
        "daily": daily.to_dict("records"),
        "selections": selections,
    }
    return summary


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [row for row in folds if row.get("status") == "trained" and row.get("metrics", {}).get("days")]
    if not trained:
        return {"folds": 0, "positive_fold_rate": 0.0}
    metrics = [row["metrics"] for row in trained]
    return {
        "folds": len(trained),
        "positive_fold_rate": round(float(np.mean([(item.get("mean_net_return_at5") or 0.0) > 0 for item in metrics])), 6),
        **{
            key: round(float(np.mean([float(item.get(key) or 0.0) for item in metrics])), 6)
            for key in (
                "mean_net_return_at5",
                "best_return_at5",
                "best_mfe_10d_at5",
                "hit5pct_10d_at5",
                "hit8pct_20d_at5",
                "all_loss_rate_at5",
                "recall_at5",
                "ndcg_at5",
                "regret_at5",
                "avg_industry_count",
                "max_drawdown",
            )
        },
    }


def v8_promotion_gate(
    candidate: dict[str, Any],
    baselines: list[dict[str, Any]],
    *,
    audits_passed: bool,
    regime_guard_passed: bool,
    config: V8GateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or V8GateConfig()
    best_hit5 = max((float(row.get("hit5pct_10d_at5") or 0.0) for row in baselines), default=0.0)
    best_hit8 = max((float(row.get("hit8pct_20d_at5") or 0.0) for row in baselines), default=0.0)
    best_all_loss = min((float(row.get("all_loss_rate_at5")) for row in baselines if row.get("all_loss_rate_at5") is not None), default=1.0)
    drawdown_limit = cfg.v5_reference_drawdown * (1.0 - cfg.required_drawdown_improvement)
    checks = {
        "audits_passed": audits_passed,
        "mean_net_return_at5_positive": float(candidate.get("mean_net_return_at5") or 0.0) > 0,
        "positive_fold_rate_at_least_60pct": float(candidate.get("positive_fold_rate") or 0.0) >= cfg.min_positive_fold_rate,
        "hit5pct_10d_beats_baselines": float(candidate.get("hit5pct_10d_at5") or 0.0) > best_hit5,
        "hit8pct_20d_beats_baselines": float(candidate.get("hit8pct_20d_at5") or 0.0) > best_hit8,
        "all_loss_rate_improves_10pct": float(candidate.get("all_loss_rate_at5") or 1.0) <= best_all_loss * (1.0 - cfg.required_all_loss_improvement),
        "drawdown_improves_25pct": float(candidate.get("max_drawdown") or -1.0) >= drawdown_limit,
        "regime_guard_passed": regime_guard_passed,
    }
    status = "candidate" if all(checks.values()) else "observe_only"
    return {
        "status": status,
        "checks": checks,
        "blocking": [name for name, passed in checks.items() if not passed],
        "baseline_reference": {"best_hit5pct_10d": best_hit5, "best_hit8pct_20d": best_hit8, "best_all_loss_rate": best_all_loss},
        "drawdown_limit": round(drawdown_limit, 6),
        "promotion_allowed": False,
        "policy": "candidate permits paper observation only; usable requires at least 40 settled signal days",
    }


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1"})


def _ndcg_at_k(selected_relevance: pd.Series, ideal_relevance: pd.Series, k: int) -> float | None:
    selected = pd.to_numeric(selected_relevance, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()[:k]
    ideal = pd.to_numeric(ideal_relevance, errors="coerce").fillna(0.0).clip(lower=0.0).sort_values(ascending=False).to_numpy()[:k]
    if not len(ideal):
        return None
    discounts_selected = np.log2(np.arange(2, len(selected) + 2))
    discounts_ideal = np.log2(np.arange(2, len(ideal) + 2))
    dcg = float(np.sum(np.expm1(selected) / discounts_selected))
    idcg = float(np.sum(np.expm1(ideal) / discounts_ideal))
    return round(dcg / idcg, 6) if idcg > 0 else 0.0


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(clean.mean()), 6) if not clean.empty else None
