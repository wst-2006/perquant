"""Reproducible descriptive audit for a frozen short-execution OOS ranking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .short_execution import SHORT_EXECUTION_CONTRACT
from .v10_market_gate import V10MarketGateResearch


class FailureWindowAuditError(RuntimeError):
    pass


FEATURE_COLUMNS = [
    "trade_date",
    "ts_code",
    "label_entry_tradable",
    "feature_return_20d",
    "feature_market_breadth_return_5d",
    "feature_market_breadth_flow_5d",
    "feature_market_drawdown_20d",
    "feature_market_return_5d",
    "feature_market_volatility_20d",
    "feature_limit_down_count_5d",
    "feature_limit_pressure_5d",
]

LABEL_COLUMNS = [
    "trade_date",
    "ts_code",
    "short_contract",
    "short_settled",
    "short_realized_net_return",
    "short_positive",
    "short_take_profit_hit",
    "short_stop_loss_hit",
    "short_gap_through_stop",
    "short_ambiguous_daily_bar",
    "short_exit_reason",
    "short_exit_untradable",
    "short_blocked_exit_days",
]


@dataclass(frozen=True)
class ShortExecutionFailureWindowAuditor:
    random_state: int = 42
    top_n: int = 5
    raw_state_factor_set: str | None = "factor_lab_v2"

    def run(
        self,
        selection_path: str | Path,
        label_dir: str | Path,
        feature_dir: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        selections = pd.read_csv(selection_path, dtype={"trade_date": str, "ts_code": str})
        required = {"trade_date", "ts_code", "candidate_rank", "prediction", "model_disagreement"}
        if not required.issubset(selections.columns):
            raise FailureWindowAuditError(f"Selection file missing columns: {sorted(required - set(selections.columns))}")
        if selections.duplicated(["trade_date", "ts_code"]).any():
            raise FailureWindowAuditError("Selection file has duplicate trade_date/ts_code keys")
        dates = sorted(selections["trade_date"].astype(str).unique())
        if not dates:
            raise FailureWindowAuditError("Selection file has no dates")

        labels = self._load_partitions(Path(label_dir), dates, LABEL_COLUMNS)
        if not labels["short_contract"].eq(SHORT_EXECUTION_CONTRACT).all():
            raise FailureWindowAuditError("Labels do not match the short-execution contract")
        features = self._load_partitions(Path(feature_dir), dates, FEATURE_COLUMNS)
        selected = selections.merge(features, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        selected = selected.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one", suffixes=("", "_label"))
        if selected["short_realized_net_return"].isna().any():
            raise FailureWindowAuditError("Some selected candidates have no settled short-execution label")

        daily = self._daily_summary(selected)
        daily, market_state = self._attach_raw_market_state(daily)
        selected = selected.merge(daily[["trade_date", "market_regime", "daily_net_return", "industry_top_share"]], on="trade_date", how="left")
        universe = features.merge(labels, on=["trade_date", "ts_code"], validate="one_to_one")
        eligible = self._eligible(universe)
        baselines = self._baseline_metrics(eligible)
        report = self._report(selected, daily, baselines, market_state, selection_path, label_dir, feature_dir)

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "failure_window_audit.json"
        daily_path = target / "failure_window_daily.csv"
        candidate_path = target / "failure_window_candidates.csv"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        selected.sort_values(["trade_date", "candidate_rank"]).to_csv(candidate_path, index=False, encoding="utf-8-sig")
        return {**report, "report_path": str(report_path), "daily_path": str(daily_path), "candidate_path": str(candidate_path)}

    def _load_partitions(self, root: Path, dates: list[str], columns: list[str]) -> pd.DataFrame:
        frames = []
        for date in dates:
            path = root / f"trade_date={date}" / "part-000.parquet"
            if not path.exists():
                raise FailureWindowAuditError(f"Missing partition: {path}")
            available = set(pd.read_parquet(path, columns=None).columns)
            missing = set(columns) - available
            if missing:
                raise FailureWindowAuditError(f"Partition {path} missing columns: {sorted(missing)}")
            frames.append(pd.read_parquet(path, columns=columns))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
        entry = frame["label_entry_tradable"].astype(str).str.lower().isin({"true", "1"})
        settled = frame["short_settled"].fillna(False).astype(bool)
        has_return = pd.to_numeric(frame["short_realized_net_return"], errors="coerce").notna()
        return frame[entry & settled & has_return].copy()

    def _daily_summary(self, selected: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for date, group in selected.groupby("trade_date", sort=True):
            if len(group) != self.top_n:
                raise FailureWindowAuditError(f"Expected {self.top_n} candidates for {date}, got {len(group)}")
            net = pd.to_numeric(group["short_realized_net_return"], errors="coerce")
            industry_share = group["industry"].fillna("unknown").value_counts(normalize=True).max()
            market = group.iloc[0]
            rows.append(
                {
                    "trade_date": str(date),
                    "daily_net_return": float(net.mean()),
                    "all_loss": bool((net < 0).all()),
                    "industry_count": int(group["industry"].fillna("unknown").nunique()),
                    "industry_top_share": float(industry_share),
                    "head_net_return": float(net[group["candidate_rank"] <= 3].mean()),
                    "tail_net_return": float(net[group["candidate_rank"] >= 4].mean()),
                    "mean_disagreement": float(pd.to_numeric(group["model_disagreement"], errors="coerce").mean()),
                    "take_profit_rate": float(group["short_take_profit_hit"].fillna(False).astype(bool).mean()),
                    "stop_loss_rate": float(group["short_stop_loss_hit"].fillna(False).astype(bool).mean()),
                    "gap_stop_rate": float(group["short_gap_through_stop"].fillna(False).astype(bool).mean()),
                    "blocked_exit_rate": float(group["short_exit_untradable"].fillna(False).astype(bool).mean()),
                    "market_breadth_return_5d": _number(market["feature_market_breadth_return_5d"]),
                    "market_breadth_flow_5d": _number(market["feature_market_breadth_flow_5d"]),
                    "market_drawdown_20d": _number(market["feature_market_drawdown_20d"]),
                    "market_return_5d": _number(market["feature_market_return_5d"]),
                    "market_volatility_20d": _number(market["feature_market_volatility_20d"]),
                }
            )
        daily = pd.DataFrame(rows)
        daily["market_regime"] = "market_state_unavailable"
        return daily

    def _attach_raw_market_state(self, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        snapshot_quality = _market_feature_quality(daily)
        if not self.raw_state_factor_set:
            return daily, {"status": "blocked", "reason": "raw market-state loader disabled", "snapshot_quality": snapshot_quality}
        raw, provenance = V10MarketGateResearch.attach_v10_market_state(
            daily[["trade_date"]].copy(), self.raw_state_factor_set
        )
        columns = {
            "feature_market_breadth_return_5d": "market_breadth_return_5d",
            "feature_market_breadth_flow_5d": "market_breadth_flow_5d",
            "feature_market_drawdown_20d": "market_drawdown_20d",
            "feature_market_return_5d": "market_return_5d",
            "feature_market_volatility_20d": "market_volatility_20d",
            "feature_limit_pressure_5d": "market_limit_pressure_5d",
        }
        raw = raw[["trade_date", *columns]].rename(columns=columns)
        existing = [column for column in columns.values() if column in daily.columns]
        merged = daily.drop(columns=existing).merge(raw, on="trade_date", how="left", validate="one_to_one")
        raw_quality = _market_feature_quality(merged)
        if raw_quality["status"] != "pass":
            raise FailureWindowAuditError(f"Raw market-state values failed quality checks: {raw_quality}")
        breadth_pct = merged["market_breadth_return_5d"].rank(pct=True, method="average")
        drawdown_pct = merged["market_drawdown_20d"].rank(pct=True, method="average")
        merged["market_regime"] = np.select(
            [breadth_pct <= 0.25, drawdown_pct <= 0.25, breadth_pct >= 0.75],
            ["breadth_bottom_quartile", "drawdown_bottom_quartile", "breadth_top_quartile"],
            default="middle_range",
        )
        return merged, {
            "status": "pass",
            "source": provenance.get("source"),
            "factor_set": self.raw_state_factor_set,
            "coverage": provenance.get("coverage"),
            "snapshot_quality": snapshot_quality,
            "raw_quality": raw_quality,
        }

    def _baseline_metrics(self, eligible: pd.DataFrame) -> dict[str, dict[str, Any]]:
        rng = np.random.default_rng(self.random_state)
        random_rows = []
        momentum_rows = []
        for _, group in eligible.groupby("trade_date", sort=True):
            if len(group) < self.top_n:
                raise FailureWindowAuditError("Fewer than five eligible stocks for a signal date")
            random_rows.append(group.iloc[np.sort(rng.choice(len(group), self.top_n, replace=False))])
            momentum_rows.append(group.sort_values(["feature_return_20d", "ts_code"], ascending=[False, True]).head(self.top_n))
        return {
            "random_top5_fixed_seed": _metrics(pd.concat(random_rows, ignore_index=True)),
            "simple_return_20d_top5": _metrics(pd.concat(momentum_rows, ignore_index=True)),
            "policy": "same signal dates, same entry-tradability, settled D+2 labels, fixed random seed",
        }

    def _report(
        self,
        selected: pd.DataFrame,
        daily: pd.DataFrame,
        baselines: dict[str, dict[str, Any]],
        market_state: dict[str, Any],
        selection_path: str | Path,
        label_dir: str | Path,
        feature_dir: str | Path,
    ) -> dict[str, Any]:
        selected = selected.copy()
        selected["disagreement_bucket"] = _upper_quartile_bucket(selected["model_disagreement"])
        selected["rank_segment"] = np.where(selected["candidate_rank"] <= 3, "rank_1_to_3", "rank_4_to_5")
        selected["industry_concentration"] = np.where(selected["industry_top_share"] >= 0.60, "top_industry_ge_60pct", "top_industry_lt_60pct")
        market = _group_metrics(daily, "market_regime", "daily_net_return")
        tail = _group_metrics(selected, "rank_segment", "short_realized_net_return")
        disagreement = _group_metrics(selected, "disagreement_bucket", "short_realized_net_return")
        concentration = _group_metrics(selected, "industry_concentration", "short_realized_net_return")
        exits = _exit_metrics(selected)
        overall = _metrics(selected)
        loss_days = daily[daily["daily_net_return"] < 0].sort_values("daily_net_return")
        return {
            "task": "short_execution_failure_window_audit",
            "report_contract": "short_execution_failure_window_audit_v1",
            "execution_contract": SHORT_EXECUTION_CONTRACT,
            "selection_path": str(selection_path),
            "label_dir": str(label_dir),
            "feature_dir": str(feature_dir),
            "oos_date_range": {"start": str(daily["trade_date"].min()), "end": str(daily["trade_date"].max())},
            "signal_days": int(len(daily)),
            "candidate_rows": int(len(selected)),
            "overall": overall,
            "baselines": baselines,
            "market_state_data": market_state,
            "classifications": {
                "market_regime": market,
                "candidate_rank_segment": tail,
                "model_disagreement": disagreement,
                "industry_concentration": concentration,
                "exit_path": exits,
                "daily_loss_summary": {
                    "loss_days": int(len(loss_days)),
                    "all_loss_days": int(daily["all_loss"].sum()),
                    "worst_days": loss_days[["trade_date", "daily_net_return", "market_regime", "industry_top_share", "tail_net_return"]].head(20).round(8).to_dict("records"),
                },
            },
            "interpretation_limits": [
                "This is descriptive classification of a frozen final OOS period, not a rule search or a new market gate.",
                "Daily open remains an explicit proxy for the requested 09:35 entry; intraday threshold order is unknown and same-bar stop-first is conservative.",
                "Quartile labels are computed only to describe this OOS sample from raw full-market snapshots and must not be used as fitted production thresholds.",
                "The Top5 constructor already enforces industry diversification, so this report cannot estimate the benefit of removing that constraint.",
                "Financial revision-vintage data is not included because official historical disclosure timing is still unverified.",
            ],
            "conclusion": {
                "status": "observe_only",
                "promotion_allowed": False,
                "next_action": "Use these frozen failure categories to pre-register a market-gate ablation and a separately held blind period; do not tune feature weights on this final OOS sample.",
            },
        }


def _number(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _upper_quartile_bucket(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    threshold = numeric.quantile(0.75)
    return np.where(numeric >= threshold, "top_quartile", "lower_three_quartiles")


def _market_feature_quality(frame: pd.DataFrame) -> dict[str, Any]:
    ranges = {
        "market_breadth_return_5d": (0.0, 1.0),
        "market_breadth_flow_5d": (0.0, 1.0),
        "market_drawdown_20d": (-1.0, 0.0),
        "market_return_5d": (-1.0, 1.0),
        "market_volatility_20d": (0.0, 1.0),
    }
    details = {}
    failures = []
    for column, (lower, upper) in ranges.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.dropna()
        in_range = bool(not finite.empty and finite.between(lower, upper).all())
        varying = bool(finite.nunique() > 1)
        details[column] = {"non_null": int(finite.size), "in_expected_range": in_range, "varies_across_dates": varying}
        if not in_range or not varying:
            failures.append(column)
    return {"status": "pass" if not failures else "blocked", "failures": failures, "details": details}


def _group_metrics(frame: pd.DataFrame, group_column: str, return_column: str) -> list[dict[str, Any]]:
    rows = []
    for name, group in frame.groupby(group_column, dropna=False, sort=True):
        values = pd.to_numeric(group[return_column], errors="coerce")
        rows.append(
            {
                "group": str(name),
                "rows": int(len(group)),
                "mean_net_return": round(float(values.mean()), 8),
                "positive_rate": round(float((values > 0).mean()), 8),
            }
        )
    return rows


def _exit_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for name, group in frame.groupby("short_exit_reason", dropna=False, sort=True):
        values = pd.to_numeric(group["short_realized_net_return"], errors="coerce")
        rows.append({"exit_reason": str(name), "rows": int(len(group)), "mean_net_return": round(float(values.mean()), 8)})
    return rows


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame["short_realized_net_return"], errors="coerce")
    daily = frame.assign(_net=values).groupby("trade_date")["_net"].mean()
    equity = (1.0 + daily.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    all_loss = frame.assign(_positive=values > 0).groupby("trade_date")["_positive"].max().eq(False)
    return {
        "signal_days": int(len(daily)),
        "candidate_rows": int(len(frame)),
        "mean_daily_net_return": round(float(daily.mean()), 8),
        "positive_signal_day_rate": round(float((daily > 0).mean()), 8),
        "positive_candidate_rate": round(float((values > 0).mean()), 8),
        "all_loss_rate": round(float(all_loss.mean()), 8),
        "cumulative_return": round(float(equity.iloc[-1] - 1.0), 8),
        "max_drawdown": round(float(drawdown.min()), 8),
        "take_profit_rate": round(float(frame["short_take_profit_hit"].fillna(False).astype(bool).mean()), 8),
        "stop_loss_rate": round(float(frame["short_stop_loss_hit"].fillna(False).astype(bool).mean()), 8),
        "gap_stop_rate": round(float(frame["short_gap_through_stop"].fillna(False).astype(bool).mean()), 8),
        "blocked_exit_rate": round(float(frame["short_exit_untradable"].fillna(False).astype(bool).mean()), 8),
    }
