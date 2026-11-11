"""Walk-forward research for independent short-execution prediction heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .research import MARKET_STATE_FEATURES
from .short_execution import SHORT_EXECUTION_CONTRACT
from .v8_metrics import diversified_top5
from .v8_research import V8ResearchConfig, V8ResearchHarness


SHORT_RESEARCH_CONTRACT = "short_execution_walk_forward_v1"
SHORT_HEADS = {
    "execution_return": {"kind": "rank", "target": "short_realized_net_return", "weight": 0.35, "direction": 1.0},
    "positive": {"kind": "classification", "target": "short_positive", "weight": 0.20, "direction": 1.0},
    "tp_before_sl": {"kind": "classification", "target": "short_take_profit_hit", "weight": 0.20, "direction": 1.0},
    "path_safety": {"kind": "regression", "target": "short_max_adverse_path", "weight": 0.15, "direction": 1.0},
    "gap_stop_risk": {"kind": "classification", "target": "short_gap_through_stop", "weight": 0.10, "direction": -1.0},
}


class ShortExecutionResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShortExecutionResearchConfig:
    train_windows: tuple[int, ...] = (120,)
    validation_days: int = 20
    step_days: int = 20
    models: tuple[str, ...] = ("lightgbm",)
    device: str = "gpu"
    n_estimators: int = 120
    random_state: int = 42
    max_folds: int | None = None
    final_audit_fraction: float = 0.30
    min_final_audit_folds: int = 3

    def v8_config(self) -> V8ResearchConfig:
        return V8ResearchConfig(
            train_windows=self.train_windows,
            validation_days=self.validation_days,
            step_days=self.step_days,
            models=self.models,
            device=self.device,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_folds=self.max_folds,
            require_dataset_audit=False,
            final_audit_fraction=self.final_audit_fraction,
            min_final_audit_folds=self.min_final_audit_folds,
        )


class ShortExecutionDataset:
    def __init__(self, feature_dir: str | Path, label_dir: str | Path) -> None:
        self.feature_dir = Path(feature_dir)
        self.label_dir = Path(label_dir)
        label_manifest_path = self.label_dir / "manifest.json"
        if not label_manifest_path.exists():
            raise ShortExecutionResearchError(f"Missing short label manifest: {label_manifest_path}")
        self.label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
        if self.label_manifest.get("contract") != SHORT_EXECUTION_CONTRACT:
            raise ShortExecutionResearchError(f"Unsupported short label contract: {self.label_manifest.get('contract')}")
        self.feature_parts = {
            path.name.split("=", 1)[1]: path / "part-000.parquet"
            for path in self.feature_dir.glob("trade_date=*")
            if (path / "part-000.parquet").exists()
        }
        self.label_parts = {
            path.name.split("=", 1)[1]: path / "part-000.parquet"
            for path in self.label_dir.glob("trade_date=*")
            if (path / "part-000.parquet").exists()
        }
        self.dates = sorted(set(self.feature_parts) & set(self.label_parts))
        if not self.dates:
            raise ShortExecutionResearchError("Feature and short-label datasets have no common dates")
        sample = pd.read_parquet(self.feature_parts[self.dates[0]])
        self.feature_columns = sorted(
            name for name in sample.columns if name.startswith("feature_") and name not in set(MARKET_STATE_FEATURES)
        )
        self.base_columns = ["trade_date", "ts_code", "industry", "label_entry_tradable", *self.feature_columns]
        self.label_columns = [
            "trade_date",
            "ts_code",
            "short_settled",
            "short_exit_date",
            "short_exit_reason",
            "short_terminal_target_date",
            "short_realized_net_return",
            "short_positive",
            "short_take_profit_hit",
            "short_stop_loss_hit",
            "short_gap_through_stop",
            "short_ambiguous_daily_bar",
            "short_max_adverse_path",
        ]

    def load_dates(self, dates: list[str]) -> pd.DataFrame:
        frames = []
        for date in dates:
            base = pd.read_parquet(self.feature_parts[date], columns=self.base_columns)
            labels = pd.read_parquet(self.label_parts[date], columns=self.label_columns)
            frames.append(base.merge(labels, on=["trade_date", "ts_code"], validate="one_to_one"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class ShortExecutionResearchHarness:
    def __init__(self) -> None:
        self.models = V8ResearchHarness()

    def run(
        self,
        feature_dir: str | Path,
        label_dir: str | Path,
        output_dir: str | Path,
        config: ShortExecutionResearchConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or ShortExecutionResearchConfig()
        v8_cfg = cfg.v8_config()
        dataset = ShortExecutionDataset(feature_dir, label_dir)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        progress_path = output / "progress.jsonl"
        progress_path.write_text("", encoding="utf-8")

        common_windows = self.common_windows(dataset.dates, cfg, v8_cfg)
        for train_window in cfg.train_windows:
            windows = common_windows[train_window]
            for fold_index, window in enumerate(windows, start=1):
                train = dataset.load_dates(window["train_dates"])
                valid = dataset.load_dates(window["validation_dates"])
                validation_start = window["validation_dates"][0]
                train, embargo = self.apply_embargo(train, validation_start)
                train = self.clean_rows(train)
                valid = self.clean_rows(valid)
                for model_name in cfg.models:
                    result = self.evaluate_family(
                        train,
                        valid,
                        dataset.feature_columns,
                        model_name,
                        v8_cfg,
                    )
                    result.update(
                        train_window=train_window,
                        fold_index=fold_index,
                        validation_start=validation_start,
                        validation_end=window["validation_dates"][-1],
                        embargo=embargo,
                    )
                    results.append(result)
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": "fold_completed", **_compact_result(result)}, ensure_ascii=False) + "\n")

        partition = self.partition(results, cfg)
        selection_rows = [row for row in results if row["validation_start"] in partition["selection_validation_starts"]]
        summaries = self.summarize_configurations(selection_rows)
        if not summaries:
            raise ShortExecutionResearchError("No trained short-execution configuration")
        best = max(summaries, key=lambda row: (row["mean_daily_net_return"], row["positive_fold_rate"]))
        final_rows = [
            row for row in results
            if row["model"] == best["model"]
            and row["train_window"] == best["train_window"]
            and row["validation_start"] in partition["audit_validation_starts"]
        ]
        final = self.summarize_folds(final_rows)
        baselines = {
            "random_top5": self.summarize_baseline(final_rows, "random_metrics"),
            "simple_return_20d_top5": self.summarize_baseline(final_rows, "momentum_metrics"),
        }
        gate = self.promotion_gate(final, baselines, partition)
        window_final_audits = self.window_final_audits(results, partition)
        selections = [item for row in final_rows for item in row["selections"]]
        daily = [item for row in final_rows for item in row["daily"]]
        report = {
            "task": "short_execution_walk_forward_research",
            "report_contract": SHORT_RESEARCH_CONTRACT,
            "execution_contract": SHORT_EXECUTION_CONTRACT,
            "feature_dir": str(feature_dir),
            "label_dir": str(label_dir),
            "config": asdict(cfg),
            "feature_count": len(dataset.feature_columns),
            "heads": SHORT_HEADS,
            "common_validation_starts": [window["validation_dates"][0] for window in common_windows[max(cfg.train_windows)]],
            "split": partition,
            "model_selection_summary": summaries,
            "best_model": best,
            "final_audit": final,
            "final_audit_baselines": baselines,
            "window_final_audits": window_final_audits,
            "gate": gate,
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        report_path = output / "short_execution_research_report.json"
        daily_path = output / "short_execution_oos_daily.csv"
        selection_path = output / "short_execution_oos_top5.csv"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(daily).to_csv(daily_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(selections).to_csv(selection_path, index=False, encoding="utf-8-sig")
        return {**report, "report_path": str(report_path), "daily_path": str(daily_path), "selection_path": str(selection_path)}

    @staticmethod
    def common_windows(
        dates: list[str],
        config: ShortExecutionResearchConfig,
        v8_config: V8ResearchConfig,
    ) -> dict[int, list[dict[str, list[str]]]]:
        """Anchor every candidate window to the longest-window validation dates."""
        unique_windows = tuple(sorted(set(config.train_windows)))
        if not unique_windows or unique_windows[0] <= 0:
            raise ShortExecutionResearchError("train_windows must contain positive day counts")
        max_window = max(unique_windows)
        anchors = V8ResearchHarness.walk_forward_windows(dates, max_window, v8_config)
        if config.max_folds is not None:
            anchors = anchors[-config.max_folds :]
        index_by_date = {date: index for index, date in enumerate(dates)}
        output: dict[int, list[dict[str, list[str]]]] = {}
        for train_window in unique_windows:
            windows = []
            for anchor in anchors:
                validation_dates = list(anchor["validation_dates"])
                validation_start = validation_dates[0]
                start_index = index_by_date[validation_start]
                train_dates = dates[start_index - train_window : start_index]
                if len(train_dates) != train_window:
                    raise ShortExecutionResearchError(f"Insufficient history for common window {train_window} at {validation_start}")
                windows.append({"train_dates": train_dates, "validation_dates": validation_dates})
            output[train_window] = windows
        return output

    @staticmethod
    def apply_embargo(frame: pd.DataFrame, validation_start: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        end_date = frame["short_exit_date"].fillna(frame["short_terminal_target_date"]).astype(str)
        keep = end_date < str(validation_start)
        output = frame[keep].copy()
        return output, {
            "status": "pass" if (output["short_exit_date"].fillna(output["short_terminal_target_date"]).astype(str) < str(validation_start)).all() else "blocked",
            "rows_before": len(frame),
            "rows_after": len(output),
            "embargoed_rows": int((~keep).sum()),
            "policy": "actual short-policy exit date must precede validation start",
        }

    @staticmethod
    def clean_rows(frame: pd.DataFrame) -> pd.DataFrame:
        valid = frame["label_entry_tradable"].astype(str).str.lower().isin({"true", "1"})
        valid &= frame["short_settled"].fillna(False).astype(bool)
        for spec in SHORT_HEADS.values():
            valid &= pd.to_numeric(frame[spec["target"]], errors="coerce").notna()
        return frame[valid].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def evaluate_family(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        feature_columns: list[str],
        model_name: str,
        config: V8ResearchConfig,
    ) -> dict[str, Any]:
        scores = {}
        importances = []
        for name, spec in SHORT_HEADS.items():
            model = self.models.fit_head(train, feature_columns, model_name, spec, config)
            scores[name] = self.models.predict_head(model, valid, feature_columns, spec)
            values = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))), dtype=float)
            if len(values) == len(feature_columns) and np.abs(values).sum() > 0:
                importances.append(np.abs(values) / np.abs(values).sum())
        scored = self.composite(valid, scores)
        metrics = evaluate_short_top5(scored)
        random_scored = valid.copy()
        random_scored["prediction"] = np.random.default_rng(config.random_state).random(len(valid))
        random_scored["model_disagreement"] = 0.0
        momentum_scored = valid.copy()
        momentum_scored["prediction"] = pd.to_numeric(momentum_scored["feature_return_20d"], errors="coerce")
        momentum_scored["model_disagreement"] = 0.0
        random_evaluation = evaluate_short_top5(random_scored)
        momentum_evaluation = evaluate_short_top5(momentum_scored)
        mean_importance = np.vstack(importances).mean(axis=0) if importances else np.zeros(len(feature_columns))
        order = np.argsort(-mean_importance)[:50]
        return {
            "status": "trained",
            "model": model_name,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "metrics": metrics["summary"],
            "daily": metrics["daily"],
            "selections": metrics["selections"],
            "random_metrics": random_evaluation["summary"],
            "random_daily": random_evaluation["daily"],
            "momentum_metrics": momentum_evaluation["summary"],
            "momentum_daily": momentum_evaluation["daily"],
            "feature_importance": [
                {"feature": feature_columns[index], "importance": round(float(mean_importance[index]), 8)}
                for index in order
            ],
        }

    @staticmethod
    def composite(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
        output = frame.copy()
        components = []
        dates = output["trade_date"].astype(str)
        for name, spec in SHORT_HEADS.items():
            raw = pd.Series(scores[name], index=output.index)
            ranked = raw.groupby(dates, sort=False).rank(pct=True)
            if spec["direction"] < 0:
                ranked = 1.0 - ranked
            output[f"prediction_{name}"] = raw
            output[f"normalized_{name}"] = ranked
            components.append(ranked * spec["weight"])
        output["prediction"] = sum(components)
        output["model_disagreement"] = output[[f"normalized_{name}" for name in SHORT_HEADS]].std(axis=1, ddof=0)
        return output

    @staticmethod
    def partition(results: list[dict[str, Any]], config: ShortExecutionResearchConfig) -> dict[str, Any]:
        dates = sorted({row["validation_start"] for row in results})
        audit_count = max(config.min_final_audit_folds, int(np.ceil(len(dates) * config.final_audit_fraction)))
        if len(dates) <= audit_count:
            raise ShortExecutionResearchError("Insufficient folds for isolated model selection and final audit")
        return {
            "status": "pass",
            "selection_validation_starts": dates[:-audit_count],
            "audit_validation_starts": dates[-audit_count:],
            "policy": "earlier walk-forward folds select configuration; latest folds remain untouched final audit",
        }

    @staticmethod
    def summarize_configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((row["model"], row["train_window"]), []).append(row)
        for (model, window), folds in groups.items():
            output.append({"model": model, "train_window": window, **ShortExecutionResearchHarness.summarize_folds(folds)})
        return output

    @staticmethod
    def summarize_folds(rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = [row["metrics"] for row in rows if row.get("status") == "trained"]
        if not metrics:
            return {"folds": 0, "positive_fold_rate": 0.0}
        daily = pd.DataFrame([item for row in rows for item in row.get("daily", [])])
        summary = {
            "folds": len(metrics),
            "positive_fold_rate": round(float(np.mean([row["mean_daily_net_return"] > 0 for row in metrics])), 8),
        }
        if daily.empty:
            return {
                **summary,
                **{
                key: round(float(np.mean([row[key] for row in metrics])), 8)
                for key in (
                    "mean_daily_net_return",
                    "positive_signal_day_rate",
                    "positive_candidate_rate",
                    "all_loss_rate",
                    "take_profit_rate",
                    "stop_loss_rate",
                    "gap_stop_rate",
                    "max_drawdown",
                )
                },
            }
        returns = pd.to_numeric(daily["mean_net_return"], errors="coerce").fillna(0.0)
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        return {
            **summary,
            "signal_days": len(daily),
            "mean_daily_net_return": round(float(returns.mean()), 8),
            "positive_signal_day_rate": round(float((returns > 0).mean()), 8),
            "positive_candidate_rate": round(float(daily["positive_candidates"].mean()), 8),
            "all_loss_rate": round(float(daily["all_loss"].mean()), 8),
            "take_profit_rate": round(float(daily["take_profit_rate"].mean()), 8),
            "stop_loss_rate": round(float(daily["stop_loss_rate"].mean()), 8),
            "gap_stop_rate": round(float(daily["gap_stop_rate"].mean()), 8),
            "cumulative_return": round(float(equity.iloc[-1] - 1.0), 8),
            "max_drawdown": round(float(drawdown.min()), 8),
        }

    @staticmethod
    def summarize_baseline(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return ShortExecutionResearchHarness.summarize_folds(
            [
                {
                    "status": "trained",
                    "metrics": row[key],
                    "daily": row[key.replace("_metrics", "_daily")],
                }
                for row in rows
            ]
        )

    @staticmethod
    def window_final_audits(results: list[dict[str, Any]], partition: dict[str, Any]) -> list[dict[str, Any]]:
        """Expose every predeclared window on exactly the same held-out folds."""
        audit_starts = set(partition.get("audit_validation_starts") or [])
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in results:
            if row.get("validation_start") in audit_starts:
                groups.setdefault((str(row["model"]), int(row["train_window"])), []).append(row)
        output = []
        for (model, train_window), rows in sorted(groups.items()):
            baselines = {
                "random_top5": ShortExecutionResearchHarness.summarize_baseline(rows, "random_metrics"),
                "simple_return_20d_top5": ShortExecutionResearchHarness.summarize_baseline(rows, "momentum_metrics"),
            }
            final = ShortExecutionResearchHarness.summarize_folds(rows)
            output.append(
                {
                    "model": model,
                    "train_window": train_window,
                    "final_audit": final,
                    "baselines": baselines,
                    "comparison": {
                        "beats_best_baseline_return": float(final.get("mean_daily_net_return") or -1.0)
                        > max(float(item.get("mean_daily_net_return") or -1.0) for item in baselines.values()),
                        "beats_best_baseline_drawdown": float(final.get("max_drawdown") or -1.0)
                        > max(float(item.get("max_drawdown") or -1.0) for item in baselines.values()),
                    },
                }
            )
        return output

    @staticmethod
    def promotion_gate(final: dict[str, Any], baselines: dict[str, dict[str, Any]], partition: dict[str, Any]) -> dict[str, Any]:
        best_baseline_return = max(float(row.get("mean_daily_net_return") or -1.0) for row in baselines.values())
        best_baseline_drawdown = max(float(row.get("max_drawdown") or -1.0) for row in baselines.values())
        checks = {
            "split_passed": partition.get("status") == "pass",
            "mean_net_return_positive": float(final.get("mean_daily_net_return") or 0.0) > 0,
            "positive_fold_rate_at_least_60pct": float(final.get("positive_fold_rate") or 0.0) >= 0.60,
            "beats_random_and_momentum_return": float(final.get("mean_daily_net_return") or -1.0) > best_baseline_return,
            "drawdown_beats_baselines": float(final.get("max_drawdown") or -1.0) > best_baseline_drawdown,
            "all_loss_rate_below_10pct": float(final.get("all_loss_rate") or 1.0) < 0.10,
        }
        return {
            "status": "candidate" if all(checks.values()) else "observe_only",
            "checks": checks,
            "blocking": [name for name, passed in checks.items() if not passed],
            "promotion_allowed": False,
            "policy": "research candidate only; packaging and forward paper validation are separate steps",
        }


def evaluate_short_top5(frame: pd.DataFrame) -> dict[str, Any]:
    daily_rows = []
    selections = []
    for date, group in frame.groupby("trade_date", sort=True):
        working = group[pd.to_numeric(group["prediction"], errors="coerce").notna()].copy()
        selected = diversified_top5(working, score_column="prediction")
        net = pd.to_numeric(selected["short_realized_net_return"], errors="coerce")
        daily_rows.append(
            {
                "trade_date": str(date),
                "mean_net_return": float(net.mean()),
                "positive_candidates": float((net > 0).mean()),
                "all_loss": bool((net < 0).all()),
                "take_profit_rate": float(selected["short_take_profit_hit"].astype(bool).mean()),
                "stop_loss_rate": float(selected["short_stop_loss_hit"].astype(bool).mean()),
                "gap_stop_rate": float(selected["short_gap_through_stop"].astype(bool).mean()),
            }
        )
        for row in selected[["ts_code", "industry", "prediction", "candidate_rank", "model_disagreement", "short_realized_net_return", "short_exit_reason", "short_take_profit_hit", "short_stop_loss_hit", "short_gap_through_stop"]].to_dict("records"):
            selections.append({"trade_date": str(date), **row})
    daily = pd.DataFrame(daily_rows)
    returns = daily["mean_net_return"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    summary = {
        "days": len(daily),
        "mean_daily_net_return": round(float(returns.mean()), 8),
        "positive_signal_day_rate": round(float((returns > 0).mean()), 8),
        "positive_candidate_rate": round(float(daily["positive_candidates"].mean()), 8),
        "all_loss_rate": round(float(daily["all_loss"].mean()), 8),
        "take_profit_rate": round(float(daily["take_profit_rate"].mean()), 8),
        "stop_loss_rate": round(float(daily["stop_loss_rate"].mean()), 8),
        "gap_stop_rate": round(float(daily["gap_stop_rate"].mean()), 8),
        "max_drawdown": round(float(drawdown.min()), 8),
    }
    return {"summary": summary, "daily": daily.to_dict("records"), "selections": selections}


def _compact_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("model", "train_window", "fold_index", "validation_start", "validation_end", "metrics")
        if key in row
    }
