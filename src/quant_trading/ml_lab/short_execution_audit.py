"""Isolated OOS audit of existing rankings under the short execution policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .short_execution import SHORT_EXECUTION_CONTRACT


class ShortExecutionAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShortExecutionOOSAuditor:
    random_state: int = 42

    def run(
        self,
        selection_path: str | Path,
        label_dir: str | Path,
        feature_dir: str | Path,
        output_dir: str | Path,
        *,
        ranking_report_path: str | Path | None = None,
        market_gate_daily_path: str | Path | None = None,
    ) -> dict[str, Any]:
        selection_file = Path(selection_path)
        labels_root = Path(label_dir)
        features_root = Path(feature_dir)
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        selections = pd.read_csv(selection_file, dtype={"trade_date": str, "ts_code": str})
        required = {"trade_date", "ts_code", "candidate_rank"}
        if not required.issubset(selections.columns):
            raise ShortExecutionAuditError(f"Selection file missing columns: {sorted(required - set(selections.columns))}")
        if selections.duplicated(["trade_date", "ts_code"]).any():
            raise ShortExecutionAuditError("Selection file has duplicate trade_date/ts_code keys")
        dates = sorted(selections["trade_date"].astype(str).unique())
        labels = self._load_partitions(labels_root, dates)
        merged = selections.merge(labels, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        if merged["short_contract"].isna().any():
            raise ShortExecutionAuditError("Some OOS selections have no short-execution label")
        unsettled = int((~merged["short_settled"].fillna(False).astype(bool)).sum())
        if unsettled:
            raise ShortExecutionAuditError(f"OOS selections include {unsettled} unsettled short-execution labels")

        momentum_rows = []
        random_rows = []
        rng = np.random.default_rng(self.random_state)
        labels_by_date = {date: group for date, group in labels.groupby("trade_date", sort=False)}
        for date in dates:
            feature_path = features_root / f"trade_date={date}" / "part-000.parquet"
            if not feature_path.exists():
                raise ShortExecutionAuditError(f"Missing feature partition: {feature_path}")
            features = pd.read_parquet(
                feature_path,
                columns=["trade_date", "ts_code", "label_entry_tradable", "feature_return_20d"],
            )
            daily = features.merge(labels_by_date[date], on=["trade_date", "ts_code"], validate="one_to_one")
            eligible = daily[
                daily["label_entry_tradable"].astype(str).str.lower().isin({"true", "1"})
                & daily["short_settled"].fillna(False).astype(bool)
                & pd.to_numeric(daily["short_realized_net_return"], errors="coerce").notna()
            ].copy()
            if len(eligible) < 5:
                raise ShortExecutionAuditError(f"Fewer than five eligible stocks on {date}")
            momentum_rows.append(
                eligible.sort_values(["feature_return_20d", "ts_code"], ascending=[False, True]).head(5)
            )
            random_rows.append(eligible.iloc[np.sort(rng.choice(len(eligible), size=5, replace=False))])
        momentum = pd.concat(momentum_rows, ignore_index=True)
        random_top5 = pd.concat(random_rows, ignore_index=True)

        gate_result: dict[str, Any] = {"available": False}
        if market_gate_daily_path:
            gate = pd.read_csv(market_gate_daily_path, dtype={"trade_date": str})
            gate = gate[gate["trade_date"].isin(dates)].copy()
            eligible_dates = set(gate.loc[gate["eligible"].fillna(False).astype(bool), "trade_date"])
            gated = merged[merged["trade_date"].isin(eligible_dates)]
            gated_metrics = _metrics(gated)
            signal_daily = gated.groupby("trade_date")["short_realized_net_return"].mean()
            calendar = pd.Series(0.0, index=gate["trade_date"].tolist(), dtype=float)
            calendar.loc[signal_daily.index] = signal_daily
            gate_result = {
                "available": True,
                "eligible_dates": sorted(eligible_dates),
                "coverage": round(len(eligible_dates) / len(gate), 8) if len(gate) else None,
                "signal_day_metrics": gated_metrics,
                "calendar_metrics": _daily_metrics(calendar),
                "no_signal_return_policy": "zero",
            }

        ranking_audit = self._ranking_audit(ranking_report_path)
        report = {
            "task": "short_execution_existing_model_oos_audit",
            "report_contract": "short_execution_oos_audit_v1",
            "execution_contract": SHORT_EXECUTION_CONTRACT,
            "selection_path": str(selection_file),
            "label_dir": str(labels_root),
            "feature_dir": str(features_root),
            "oos_date_range": {"start": min(dates), "end": max(dates)},
            "oos_signal_days": len(dates),
            "oos_candidate_rows": len(merged),
            "existing_v11_top5": _metrics(merged),
            "v10_market_gate": gate_result,
            "baselines": {
                "random_top5_fixed_seed": _metrics(random_top5),
                "simple_return_20d_top5": _metrics(momentum),
                "policy": "same dates, same entry-tradability and short-execution labels; random seed is fixed and is diagnostic only",
            },
            "ranking_model_overfit_audit": ranking_audit,
            "conclusion": {
                "status": "observe_only",
                "promotion_allowed": False,
                "blocking": [
                    "current v11 heads do not target the D+2 execution policy",
                    "existing v11 short-policy OOS mean return is not positive",
                    "v10 gated short-policy sample has fewer than 25 signal days and coverage is below 25 percent",
                    "historical 09:35 fills remain unavailable; daily open is only an explicit proxy",
                ],
            },
        }
        report_path = target / "short_execution_oos_audit.json"
        daily = merged.groupby("trade_date").agg(
            mean_net_return=("short_realized_net_return", "mean"),
            positive_candidates=("short_positive", "mean"),
            take_profit_rate=("short_take_profit_hit", "mean"),
            stop_loss_rate=("short_stop_loss_hit", "mean"),
            gap_stop_rate=("short_gap_through_stop", "mean"),
        ).reset_index()
        daily_path = target / "short_execution_v11_oos_daily.csv"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        return {**report, "report_path": str(report_path), "daily_path": str(daily_path)}

    @staticmethod
    def _load_partitions(root: Path, dates: list[str]) -> pd.DataFrame:
        frames = []
        for date in dates:
            path = root / f"trade_date={date}" / "part-000.parquet"
            if not path.exists():
                raise ShortExecutionAuditError(f"Missing short label partition: {path}")
            frames.append(pd.read_parquet(path))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _ranking_audit(path: str | Path | None) -> dict[str, Any]:
        if not path:
            return {"available": False}
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        selection = report.get("best_model") or {}
        final = report.get("final_audit") or {}
        importance = ((report.get("feature_diagnostics") or {}).get("builtin_importance") or [])
        permutation = ((report.get("feature_diagnostics") or {}).get("permutation_diagnostics") or [])
        top10_share = sum(float(row.get("mean_importance") or 0.0) for row in importance[:10])
        thin_permutation = sum(int(row.get("folds") or 0) < 3 for row in permutation)
        return {
            "available": True,
            "feature_count": report.get("feature_count"),
            "selection_mean_net_return_at5": selection.get("mean_net_return_at5"),
            "final_mean_net_return_at5": final.get("mean_net_return_at5"),
            "selection_positive_fold_rate": selection.get("positive_fold_rate"),
            "final_positive_fold_rate": final.get("positive_fold_rate"),
            "top10_builtin_importance_share": round(top10_share, 8),
            "permutation_features_with_fewer_than_3_folds": thin_permutation,
            "permutation_feature_count": len(permutation),
            "warnings": [
                "large model-selection to final-audit degradation is direct overfit/model-selection risk",
                "permutation conclusions based on fewer than three final folds are unstable",
                "in-sample calibration in the current artifact is not trusted as OOS probability calibration",
            ],
        }


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"signal_days": 0, "candidate_rows": 0}
    values = pd.to_numeric(frame["short_realized_net_return"], errors="coerce")
    daily = frame.assign(_net=values).groupby("trade_date")["_net"].mean()
    positive = values > 0
    all_loss = frame.assign(_positive=positive).groupby("trade_date")["_positive"].max().eq(False)
    gains = values[values > 0]
    losses = values[values < 0]
    avg_gain = float(gains.mean()) if len(gains) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    break_even = abs(avg_loss) / (avg_gain + abs(avg_loss)) if avg_gain and avg_loss else None
    return {
        "signal_days": int(daily.size),
        "candidate_rows": int(len(frame)),
        "mean_daily_net_return": round(float(daily.mean()), 8),
        "positive_signal_day_rate": round(float((daily > 0).mean()), 8),
        "positive_candidate_rate": round(float(positive.mean()), 8),
        "all_loss_rate": round(float(all_loss.mean()), 8),
        "take_profit_rate": round(float(frame["short_take_profit_hit"].fillna(False).astype(bool).mean()), 8),
        "stop_loss_rate": round(float(frame["short_stop_loss_hit"].fillna(False).astype(bool).mean()), 8),
        "gap_stop_rate": round(float(frame["short_gap_through_stop"].fillna(False).astype(bool).mean()), 8),
        "ambiguous_bar_rate": round(float(frame["short_ambiguous_daily_bar"].fillna(False).astype(bool).mean()), 8),
        "average_gain": round(avg_gain, 8) if avg_gain is not None else None,
        "average_loss": round(avg_loss, 8) if avg_loss is not None else None,
        "break_even_positive_rate": round(float(break_even), 8) if break_even is not None else None,
        **_daily_metrics(daily),
        "worst_signal_days": [
            {"trade_date": str(date), "mean_net_return": round(float(value), 8)}
            for date, value in daily.nsmallest(min(10, len(daily))).items()
        ],
    }


def _daily_metrics(daily: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(daily, errors="coerce").fillna(0.0)
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "calendar_days": int(len(values)),
        "calendar_mean_net_return": round(float(values.mean()), 8) if len(values) else None,
        "cumulative_return": round(float(equity.iloc[-1] - 1.0), 8) if len(equity) else None,
        "max_drawdown": round(float(drawdown.min()), 8) if len(drawdown) else None,
    }
