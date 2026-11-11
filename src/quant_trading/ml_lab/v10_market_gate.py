"""v10 market-state admission research for v9/v8 Top5 OOS streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..db import FactorSnapshot, session_scope
from ..factor_lab.metrics import max_drawdown
from .research import MARKET_STATE_FEATURES


V10_MARKET_GATE_CONTRACT = "ml_v10_market_gate_research_v1"
V10_EXTRA_STATE_FEATURES = (
    "feature_market_amount_change_5d_20d",
    "feature_industry_positive_rate_5d",
)
V10_STATE_FEATURES = tuple(MARKET_STATE_FEATURES) + V10_EXTRA_STATE_FEATURES
V10_DECISIONS = ("eligible_stable", "eligible_aggressive", "no_signal")


class V10MarketGateError(RuntimeError):
    """Raised when v10 market-state admission cannot be audited safely."""


@dataclass(frozen=True)
class V10MarketGateConfig:
    base_train_days: int = 40
    model_selection_days: int = 20
    final_audit_days: int = 40
    stable_coverages: tuple[float, ...] = (0.25, 0.40, 0.55)
    aggressive_coverages: tuple[float, ...] = (0.00, 0.10, 0.20)
    candidate_models: tuple[str, ...] = ("rule", "logistic", "lightgbm")
    min_coverage: float = 0.25
    min_positive_fold_rate: float = 0.60
    required_drawdown_improvement: float = 0.35
    required_all_loss_improvement: float = 0.10
    aggressive_best_return_threshold: float = 0.05
    aggressive_mfe_threshold: float = 0.10
    require_positive_selection_return: bool = True
    rolling_min_history_days: int = 40
    rolling_label_embargo_days: int = 20
    random_state: int = 42


@dataclass(frozen=True)
class ConstantBinaryModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(x), float(self.probability), dtype=float)
        return np.column_stack([1.0 - positive, positive])


class V10MarketGateResearch:
    """Chronologically isolated stable/aggressive/no_signal admission research."""

    def run(
        self,
        ranking_report_path: str | Path,
        output_dir: str | Path,
        config: V10MarketGateConfig | None = None,
        raw_state_factor_set: str = "factor_lab_v2",
    ) -> dict[str, Any]:
        cfg = config or V10MarketGateConfig()
        report_path = Path(ranking_report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        daily = self.prepare_daily_rows(report, cfg)
        required_days = cfg.base_train_days + cfg.model_selection_days + cfg.final_audit_days
        if len(daily) < required_days:
            raise V10MarketGateError(f"Need at least {required_days} OOS days, found {len(daily)}")
        daily = daily.iloc[-required_days:].copy().reset_index(drop=True)
        state, provenance = self.attach_v10_market_state(daily, factor_set=raw_state_factor_set)
        state = state.sort_values("trade_date").reset_index(drop=True)

        base_end = cfg.base_train_days
        selection_end = cfg.base_train_days + cfg.model_selection_days
        base = state.iloc[:base_end].copy()
        selection = state.iloc[base_end:selection_end].copy()
        final = state.iloc[selection_end:].copy()
        self.validate_training_targets(base)

        candidate_results: dict[str, dict[str, Any]] = {}
        candidate_daily: dict[str, list[dict[str, Any]]] = {}
        for model_name in cfg.candidate_models:
            model = self.fit_candidate(model_name, base, cfg)
            base_scores = self.predict_candidate(model_name, model, base)
            selection_scores = self.predict_candidate(model_name, model, selection)
            for stable_coverage in cfg.stable_coverages:
                for aggressive_coverage in cfg.aggressive_coverages:
                    thresholds = self.thresholds_from_scores(
                        base_scores,
                        stable_coverage=stable_coverage,
                        aggressive_coverage=aggressive_coverage,
                    )
                    decided = self.decision_frame(selection, selection_scores, thresholds)
                    metrics = self.evaluate_decisions(decided)
                    key = self.candidate_key(model_name, stable_coverage, aggressive_coverage)
                    candidate_results[key] = {
                        "model": model_name,
                        "stable_coverage": stable_coverage,
                        "aggressive_coverage": aggressive_coverage,
                        "thresholds": thresholds,
                        "selection_metrics": metrics,
                    }
                    candidate_daily[key] = decided.to_dict("records")

        selected_key = self.select_candidate(candidate_results, cfg)
        selected = candidate_results[selected_key]
        model_selection_gate = self.model_selection_gate(selected.get("selection_metrics") or {}, cfg)
        training = pd.concat([base, selection], ignore_index=True)
        final_model = self.fit_candidate(str(selected["model"]), training, cfg)
        training_scores = self.predict_candidate(str(selected["model"]), final_model, training)
        final_thresholds = self.thresholds_from_scores(
            training_scores,
            stable_coverage=float(selected["stable_coverage"]),
            aggressive_coverage=float(selected["aggressive_coverage"]),
        )
        final_scores = self.predict_candidate(str(selected["model"]), final_model, final)
        final_decisions = self.decision_frame(final, final_scores, final_thresholds)
        final_metrics = self.evaluate_decisions(final_decisions)
        rolling = self.rolling_prior_oos_evaluation(state, cfg)
        ungated = final.copy()
        ungated["decision"] = "eligible_stable"
        ungated["eligible"] = True
        ungated["gate_score_stable"] = 1.0
        ungated["gate_score_aggressive"] = 1.0
        ungated["calendar_net_return_5d"] = pd.to_numeric(ungated["mean_net_return_5d"], errors="coerce").fillna(0.0)
        ungated_metrics = self.evaluate_decisions(ungated)
        baseline_comparison = self.compare_ranker_baselines(report, final_metrics)
        gate = self.gate(final_metrics, ungated_metrics, baseline_comparison, cfg, model_selection_gate)
        failure_diagnostics = self.failure_window_diagnostics(final_decisions)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "v10_market_gate.joblib"
        joblib.dump(
            {
                "model": final_model,
                "model_name": selected["model"],
                "thresholds": final_thresholds,
                "features": list(V10_STATE_FEATURES),
                "config": asdict(cfg),
                "state_feature_provenance": provenance,
            },
            model_path,
        )
        final_daily_path = output / "v10_market_gate_final_daily.csv"
        selection_daily_path = output / "v10_market_gate_selection_daily.csv"
        rolling_summary_path = output / "v10_market_gate_rolling_summary.csv"
        rolling_daily_path = output / "v10_market_gate_rolling_best_daily.csv"
        final_decisions.to_csv(final_daily_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(candidate_daily[selected_key]).to_csv(selection_daily_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(rolling["summary"]).to_csv(rolling_summary_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(rolling["best_daily"]).to_csv(rolling_daily_path, index=False, encoding="utf-8-sig")

        payload = {
            "task": "v10_market_state_admission_research",
            "report_contract": V10_MARKET_GATE_CONTRACT,
            "ranking_report_path": str(report_path),
            "ranking_report_contract": report.get("report_contract"),
            "ranking_model": report.get("best_model"),
            "data_policy": (
                "Uses only ranking-model OOS daily outcomes plus signal-date-or-earlier "
                "Tushare-backed market state features; no simulated market, flow, or event data."
            ),
            "decision_policy": (
                "eligible_stable/eligible_aggressive/no_signal only; no orders, quantities, "
                "positions, buy/sell instructions, or automatic trading actions."
            ),
            "config": asdict(cfg),
            "split": {
                "base_train": [str(base["trade_date"].iloc[0]), str(base["trade_date"].iloc[-1])],
                "model_selection": [str(selection["trade_date"].iloc[0]), str(selection["trade_date"].iloc[-1])],
                "final_audit": [str(final["trade_date"].iloc[0]), str(final["trade_date"].iloc[-1])],
                "status": "pass",
                "policy": "Gate candidates are selected on the middle OOS segment and audited on the later untouched OOS segment.",
            },
            "features": list(V10_STATE_FEATURES),
            "state_feature_provenance": provenance,
            "candidate_selection": candidate_results,
            "selected_gate_key": selected_key,
            "selected_gate": selected["model"],
            "model_selection_gate": model_selection_gate,
            "selected_stable_coverage": selected["stable_coverage"],
            "selected_aggressive_coverage": selected["aggressive_coverage"],
            "thresholds": final_thresholds,
            "model_path": str(model_path),
            "final_audit": final_metrics,
            "ungated_final_audit": ungated_metrics,
            "rolling_prior_oos": {
                key: value for key, value in rolling.items() if key not in {"all_daily", "best_daily"}
            },
            "baseline_comparison": baseline_comparison,
            "failure_window_diagnostics": failure_diagnostics,
            "gate": gate,
            "selection_blocking": model_selection_gate.get("blocking", []),
            "daily_path": str(final_daily_path),
            "selection_daily_path": str(selection_daily_path),
            "rolling_summary_path": str(rolling_summary_path),
            "rolling_best_daily_path": str(rolling_daily_path),
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        report_output = output / "v10_market_gate_report.json"
        report_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {**payload, "report_path": str(report_output)}

    @staticmethod
    def prepare_daily_rows(report: dict[str, Any], config: V10MarketGateConfig) -> pd.DataFrame:
        rows = pd.DataFrame(report.get("oos_daily") or [])
        if rows.empty:
            raise V10MarketGateError("Ranking report has no oos_daily stream for v10 admission research.")
        required = {
            "trade_date",
            "fold_index",
            "mean_net_return_5d",
            "best_net_return_5d",
            "best_mfe_10d",
            "hit5pct_10d",
            "hit8pct_20d",
            "all_loss_5d",
        }
        missing = sorted(required - set(rows.columns))
        if missing:
            raise V10MarketGateError(f"OOS daily stream missing v10 columns: {missing}")
        daily = rows.copy()
        daily["trade_date"] = daily["trade_date"].astype(str)
        if daily["trade_date"].duplicated().any():
            raise V10MarketGateError("OOS daily stream contains duplicated trade dates.")
        for column in ("fold_index", "candidate_count"):
            if column in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0).astype(int)
        for column in (
            "mean_net_return_5d",
            "market_mean_net_return_5d",
            "best_net_return_5d",
            "best_mfe_10d",
            "recall_at5",
            "ndcg_at5",
            "regret_at5",
        ):
            if column in daily.columns:
                daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["hit5pct_10d"] = _bool_series(daily["hit5pct_10d"])
        daily["hit8pct_20d"] = _bool_series(daily["hit8pct_20d"])
        daily["all_loss_5d"] = _bool_series(daily["all_loss_5d"])
        daily["stable_positive"] = (
            (daily["mean_net_return_5d"] > 0.0)
            & (~daily["all_loss_5d"])
        ).astype(int)
        daily["aggressive_positive"] = (
            (~daily["all_loss_5d"])
            & (
                (daily["best_net_return_5d"] >= config.aggressive_best_return_threshold)
                | (daily["best_mfe_10d"] >= config.aggressive_mfe_threshold)
                | (daily["hit5pct_10d"] & daily["hit8pct_20d"])
            )
        ).astype(int)
        return daily.sort_values("trade_date").reset_index(drop=True)

    @staticmethod
    def validate_training_targets(frame: pd.DataFrame) -> None:
        if frame["stable_positive"].nunique() < 2 and frame["aggressive_positive"].nunique() < 2:
            raise V10MarketGateError("Base training segment has single-class stable and aggressive targets.")

    @staticmethod
    def attach_v10_market_state(daily: pd.DataFrame, factor_set: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        dates = daily["trade_date"].astype(str).tolist()
        macro_features = list(MARKET_STATE_FEATURES[:5])
        values: dict[str, dict[str, float]] = {}
        aggregates: dict[str, dict[str, Any]] = {}
        with session_scope() as session:
            query = (
                session.query(FactorSnapshot.trade_date, FactorSnapshot.industry, FactorSnapshot.raw_factors)
                .filter(FactorSnapshot.factor_set == factor_set, FactorSnapshot.trade_date.in_(dates))
                .yield_per(20_000)
            )
            for trade_date, industry, raw in query:
                date = str(trade_date)
                source = raw or {}
                bucket = values.setdefault(date, {})
                for feature in macro_features:
                    raw_name = feature.replace("feature_", "", 1)
                    value = source.get(raw_name)
                    if value is not None and feature not in bucket:
                        bucket[feature] = float(value)
                stats = aggregates.setdefault(
                    date,
                    {
                        "return_count": 0.0,
                        "return_positive": 0.0,
                        "return_sum": 0.0,
                        "return_square_sum": 0.0,
                        "flow_count": 0.0,
                        "flow_positive": 0.0,
                        "pressure_count": 0.0,
                        "pressure_sum": 0.0,
                        "amount_5d_sum": 0.0,
                        "amount_20d_sum": 0.0,
                        "amount_count": 0.0,
                        "industry_returns": {},
                    },
                )
                return_5d = source.get("return_5d")
                if return_5d is not None:
                    value = float(return_5d)
                    stats["return_count"] += 1.0
                    stats["return_positive"] += float(value > 0)
                    stats["return_sum"] += value
                    stats["return_square_sum"] += value * value
                    industry_key = str(industry or "UNKNOWN")
                    stats["industry_returns"].setdefault(industry_key, []).append(value)
                flow = source.get("main_net_amount_sum_5d")
                if flow is not None:
                    stats["flow_count"] += 1.0
                    stats["flow_positive"] += float(float(flow) > 0)
                up_count = source.get("limit_up_count_5d")
                down_count = source.get("limit_down_count_5d")
                if up_count is not None and down_count is not None:
                    stats["pressure_count"] += 1.0
                    stats["pressure_sum"] += float(up_count) - float(down_count)
                amount_5d = source.get("amount_avg_5d")
                amount_20d = source.get("amount_avg_20d")
                if amount_5d is not None and amount_20d not in (None, 0):
                    stats["amount_5d_sum"] += float(amount_5d)
                    stats["amount_20d_sum"] += float(amount_20d)
                    stats["amount_count"] += 1.0
        state_rows: list[dict[str, Any]] = []
        coverage_details: dict[str, Any] = {}
        for date in dates:
            row = {"trade_date": date, **values.get(date, {})}
            stats = aggregates.get(date, {})
            return_count = float(stats.get("return_count", 0.0) or 0.0)
            return_sum = float(stats.get("return_sum", 0.0) or 0.0)
            mean_return = return_sum / return_count if return_count else None
            variance = None
            if return_count and mean_return is not None:
                variance = float(stats.get("return_square_sum", 0.0)) / return_count - mean_return * mean_return
            row["feature_market_breadth_return_5d"] = (
                float(stats.get("return_positive", 0.0)) / return_count if return_count else None
            )
            flow_count = float(stats.get("flow_count", 0.0) or 0.0)
            row["feature_market_breadth_flow_5d"] = (
                float(stats.get("flow_positive", 0.0)) / flow_count if flow_count else None
            )
            row["feature_market_return_dispersion_5d"] = (
                max(float(variance), 0.0) ** 0.5 if variance is not None else None
            )
            pressure_count = float(stats.get("pressure_count", 0.0) or 0.0)
            row["feature_limit_pressure_5d"] = (
                float(stats.get("pressure_sum", 0.0)) / pressure_count if pressure_count else None
            )
            amount_count = float(stats.get("amount_count", 0.0) or 0.0)
            if amount_count:
                amount_5d = float(stats.get("amount_5d_sum", 0.0)) / amount_count
                amount_20d = float(stats.get("amount_20d_sum", 0.0)) / amount_count
                row["feature_market_amount_change_5d_20d"] = amount_5d / amount_20d - 1.0 if amount_20d else None
            else:
                row["feature_market_amount_change_5d_20d"] = None
            industry_returns = stats.get("industry_returns") or {}
            industry_means = [float(np.mean(values)) for values in industry_returns.values() if values]
            row["feature_industry_positive_rate_5d"] = (
                float(np.mean([value > 0 for value in industry_means])) if industry_means else None
            )
            coverage_details[date] = {
                "market_row_count": int(return_count),
                "industry_count": int(len(industry_means)),
            }
            state_rows.append(row)
        state = pd.DataFrame(state_rows).drop_duplicates("trade_date")
        merged = daily.drop(columns=[feature for feature in V10_STATE_FEATURES if feature in daily.columns]).merge(
            state,
            on="trade_date",
            how="left",
        )
        coverage = {
            feature: round(float(merged[feature].notna().mean()), 6) if feature in merged else 0.0
            for feature in V10_STATE_FEATURES
        }
        if any(rate < 0.95 for rate in coverage.values()):
            raise V10MarketGateError(f"v10 market-state feature coverage below 95%: {coverage}")
        return merged, {
            "source": "factor_snapshot.raw_factors; real Tushare-backed signal-date full-market aggregation",
            "factor_set": factor_set,
            "features": list(V10_STATE_FEATURES),
            "coverage": coverage,
            "daily_coverage_sample": coverage_details,
            "simulated_values_used": False,
        }

    @staticmethod
    def fit_candidate(name: str, frame: pd.DataFrame, config: V10MarketGateConfig) -> Any:
        if name == "rule":
            x = V10MarketGateResearch.clean_features(frame)
            return {
                "feature_stats": {
                    column: {
                        "median": float(x[column].median()),
                        "scale": max(float(x[column].quantile(0.75) - x[column].quantile(0.25)), 1e-9),
                    }
                    for column in V10_STATE_FEATURES
                }
            }
        x = V10MarketGateResearch.clean_features(frame)
        return {
            "stable": V10MarketGateResearch.fit_classifier(name, x, frame["stable_positive"].astype(int), config),
            "aggressive": V10MarketGateResearch.fit_classifier(name, x, frame["aggressive_positive"].astype(int), config),
        }

    @staticmethod
    def fit_classifier(name: str, x: pd.DataFrame, y: pd.Series, config: V10MarketGateConfig) -> Any:
        if y.nunique() < 2:
            return ConstantBinaryModel(float(y.mean()))
        if name == "logistic":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.35, max_iter=2000, random_state=config.random_state),
            )
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=2,
                num_leaves=7,
                min_child_samples=10,
                reg_lambda=3.0,
                random_state=config.random_state,
                verbose=-1,
            )
        else:
            raise V10MarketGateError(f"Unsupported v10 market-gate model: {name}")
        model.fit(x, y)
        return model

    @staticmethod
    def predict_candidate(name: str, model: Any, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if name == "rule":
            return {
                "stable": V10MarketGateResearch.rule_score(model, frame, target="stable"),
                "aggressive": V10MarketGateResearch.rule_score(model, frame, target="aggressive"),
            }
        x = V10MarketGateResearch.clean_features(frame)
        return {
            "stable": np.asarray(model["stable"].predict_proba(x))[:, 1],
            "aggressive": np.asarray(model["aggressive"].predict_proba(x))[:, 1],
        }

    @staticmethod
    def rule_score(model: dict[str, Any], frame: pd.DataFrame, target: str) -> np.ndarray:
        stats = model["feature_stats"]
        stable_higher = {
            "feature_market_return_1d",
            "feature_market_return_5d",
            "feature_market_return_20d",
            "feature_market_drawdown_20d",
            "feature_market_breadth_return_5d",
            "feature_market_breadth_flow_5d",
            "feature_limit_pressure_5d",
            "feature_market_amount_change_5d_20d",
            "feature_industry_positive_rate_5d",
        }
        stable_lower = {"feature_market_volatility_20d", "feature_market_return_dispersion_5d"}
        aggressive_higher = {
            "feature_market_return_1d",
            "feature_market_return_5d",
            "feature_market_breadth_flow_5d",
            "feature_limit_pressure_5d",
            "feature_market_amount_change_5d_20d",
            "feature_industry_positive_rate_5d",
            "feature_market_return_dispersion_5d",
        }
        aggressive_lower = {"feature_market_volatility_20d", "feature_market_drawdown_20d"}
        higher = stable_higher if target == "stable" else aggressive_higher
        lower = stable_lower if target == "stable" else aggressive_lower
        parts = []
        for column in V10_STATE_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(stats[column]["median"])
            zscore = (values - stats[column]["median"]) / stats[column]["scale"]
            if column in lower:
                zscore = -zscore
            elif column not in higher:
                zscore = zscore * 0.0
            parts.append(zscore.clip(-5.0, 5.0))
        return pd.concat(parts, axis=1).mean(axis=1).to_numpy(dtype=float)

    @staticmethod
    def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[list(V10_STATE_FEATURES)].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    @staticmethod
    def thresholds_from_scores(
        scores: dict[str, np.ndarray],
        *,
        stable_coverage: float,
        aggressive_coverage: float,
    ) -> dict[str, float]:
        stable_threshold = float(pd.Series(scores["stable"]).quantile(1.0 - stable_coverage))
        aggressive_threshold = (
            float(pd.Series(scores["aggressive"]).quantile(1.0 - aggressive_coverage))
            if aggressive_coverage > 0.0
            else None
        )
        return {
            "stable_threshold": stable_threshold,
            "aggressive_threshold": aggressive_threshold,
            "stable_target_coverage": float(stable_coverage),
            "aggressive_target_coverage": float(aggressive_coverage),
        }

    @staticmethod
    def decision_frame(
        frame: pd.DataFrame,
        scores: dict[str, np.ndarray],
        thresholds: dict[str, float],
    ) -> pd.DataFrame:
        output = frame.copy()
        output["gate_score_stable"] = np.asarray(scores["stable"], dtype=float)
        output["gate_score_aggressive"] = np.asarray(scores["aggressive"], dtype=float)
        stable = output["gate_score_stable"] >= float(thresholds["stable_threshold"])
        aggressive_threshold = thresholds.get("aggressive_threshold")
        aggressive = (
            (~stable) & (output["gate_score_aggressive"] >= float(aggressive_threshold))
            if aggressive_threshold is not None
            else pd.Series(False, index=output.index)
        )
        output["decision"] = np.select(
            [stable, aggressive],
            ["eligible_stable", "eligible_aggressive"],
            default="no_signal",
        )
        output["eligible"] = output["decision"] != "no_signal"
        returns = pd.to_numeric(output["mean_net_return_5d"], errors="coerce").fillna(0.0)
        output["calendar_net_return_5d"] = np.where(output["eligible"], returns, 0.0)
        output["threshold_stable"] = float(thresholds["stable_threshold"])
        output["threshold_aggressive"] = (
            float(thresholds["aggressive_threshold"])
            if thresholds.get("aggressive_threshold") is not None
            else np.nan
        )
        output["no_signal_return_policy"] = "zero_return_on_full_calendar"
        return output

    @staticmethod
    def evaluate_decisions(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"days": 0, "coverage": 0.0, "decision_counts": {}}
        returns = pd.to_numeric(frame["mean_net_return_5d"], errors="coerce").fillna(0.0)
        calendar = pd.to_numeric(frame["calendar_net_return_5d"], errors="coerce").fillna(0.0)
        eligible = frame["decision"].astype(str) != "no_signal"
        stable = frame["decision"].astype(str) == "eligible_stable"
        aggressive = frame["decision"].astype(str) == "eligible_aggressive"
        eligible_returns = returns[eligible]
        stable_returns = returns[stable]
        aggressive_returns = returns[aggressive]
        if "fold_index" in frame.columns:
            fold_means = frame.groupby("fold_index", sort=True)["calendar_net_return_5d"].mean().tolist()
        else:
            fold_means = [float(np.mean(chunk)) for chunk in np.array_split(calendar, max(1, int(np.ceil(len(calendar) / 20))))]
        all_loss = _bool_series(frame["all_loss_5d"])
        hit5 = _bool_series(frame["hit5pct_10d"])
        hit8 = _bool_series(frame["hit8pct_20d"])
        best_returns = pd.to_numeric(frame.get("best_net_return_5d", pd.Series(dtype=float)), errors="coerce")
        best_mfe = pd.to_numeric(frame.get("best_mfe_10d", pd.Series(dtype=float)), errors="coerce")
        return {
            "days": int(len(frame)),
            "eligible_days": int(eligible.sum()),
            "stable_days": int(stable.sum()),
            "aggressive_days": int(aggressive.sum()),
            "no_signal_days": int((~eligible).sum()),
            "coverage": _round(float(eligible.mean())),
            "stable_coverage": _round(float(stable.mean())),
            "aggressive_coverage": _round(float(aggressive.mean())),
            "all_calendar_mean_net_return": _round(float(calendar.mean())),
            "eligible_mean_net_return": _round(float(eligible_returns.mean())) if eligible.any() else 0.0,
            "stable_mean_net_return": _round(float(stable_returns.mean())) if stable.any() else None,
            "aggressive_mean_net_return": _round(float(aggressive_returns.mean())) if aggressive.any() else None,
            "positive_day_rate_on_eligible": _round(float((eligible_returns > 0).mean())) if eligible.any() else 0.0,
            "positive_fold_rate": _round(float(np.mean([value > 0 for value in fold_means]))),
            "hit5pct_10d_at5_on_eligible": _round(float(hit5[eligible].mean())) if eligible.any() else 0.0,
            "hit8pct_20d_at5_on_eligible": _round(float(hit8[eligible].mean())) if eligible.any() else 0.0,
            "all_loss_rate_at5_on_eligible": _round(float(all_loss[eligible].mean())) if eligible.any() else 1.0,
            "best_return_5d_on_eligible": _round(float(best_returns[eligible].mean())) if eligible.any() else None,
            "best_mfe_10d_on_eligible": _round(float(best_mfe[eligible].mean())) if eligible.any() else None,
            "calendar_max_drawdown": _round(float(max_drawdown(calendar.tolist()))),
            "calendar_cumulative_return": _round(float(np.prod(1.0 + calendar.to_numpy(dtype=float)) - 1.0)),
            "decision_counts": dict(sorted(frame["decision"].astype(str).value_counts().to_dict().items())),
            "no_signal_return_policy": "zero_return_on_full_calendar",
        }

    @staticmethod
    def select_candidate(results: dict[str, dict[str, Any]], config: V10MarketGateConfig) -> str:
        viable = [
            key for key, item in results.items()
            if float((item.get("selection_metrics") or {}).get("coverage") or 0.0) >= config.min_coverage
        ]
        pool = viable or list(results)
        return max(pool, key=lambda key: V10MarketGateResearch.selection_key(results[key]))

    def rolling_prior_oos_evaluation(self, state: pd.DataFrame, config: V10MarketGateConfig) -> dict[str, Any]:
        rows_by_key: dict[str, list[dict[str, Any]]] = {}
        state = state.sort_values("trade_date").reset_index(drop=True)
        for position in range(config.rolling_min_history_days + config.rolling_label_embargo_days, len(state)):
            train_end = position - config.rolling_label_embargo_days
            train = state.iloc[:train_end].copy()
            current = state.iloc[position : position + 1].copy()
            if len(train) < config.rolling_min_history_days:
                continue
            for model_name in config.candidate_models:
                try:
                    model = self.fit_candidate(model_name, train, config)
                except Exception:
                    continue
                train_scores = self.predict_candidate(model_name, model, train)
                current_scores = self.predict_candidate(model_name, model, current)
                for stable_coverage in config.stable_coverages:
                    for aggressive_coverage in config.aggressive_coverages:
                        thresholds = self.thresholds_from_scores(
                            train_scores,
                            stable_coverage=stable_coverage,
                            aggressive_coverage=aggressive_coverage,
                        )
                        decided = self.decision_frame(current, current_scores, thresholds)
                        decided["rolling_train_start"] = str(train["trade_date"].iloc[0])
                        decided["rolling_train_end"] = str(train["trade_date"].iloc[-1])
                        decided["rolling_label_embargo_days"] = int(config.rolling_label_embargo_days)
                        key = self.candidate_key(model_name, stable_coverage, aggressive_coverage)
                        rows_by_key.setdefault(key, []).extend(decided.to_dict("records"))
        summary = []
        for key, rows in sorted(rows_by_key.items()):
            frame = pd.DataFrame(rows)
            model, stable_coverage, aggressive_coverage = self.parse_candidate_key(key)
            summary.append(
                {
                    "candidate_key": key,
                    "model": model,
                    "target_stable_coverage": stable_coverage,
                    "target_aggressive_coverage": aggressive_coverage,
                    **self.evaluate_decisions(frame),
                }
            )
        best_summary = max(summary, key=self.rolling_selection_key) if summary else {}
        best_key = str(best_summary.get("candidate_key") or "")
        best_daily = rows_by_key.get(best_key, [])
        return {
            "policy": "Each signal date is decided only from earlier OOS days whose 20-day labels have matured; no future or same-day outcome is used.",
            "min_history_days": int(config.rolling_min_history_days),
            "label_embargo_days": int(config.rolling_label_embargo_days),
            "candidate_count": int(len(summary)),
            "summary": summary,
            "best": best_summary,
            "best_daily": best_daily,
            "all_daily": rows_by_key,
        }

    @staticmethod
    def rolling_selection_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(item.get("all_calendar_mean_net_return") or -999.0),
            float(item.get("eligible_mean_net_return") or -999.0),
            float(item.get("positive_fold_rate") or 0.0),
            -float(item.get("all_loss_rate_at5_on_eligible") or 1.0),
            float(item.get("calendar_max_drawdown") or -999.0),
            float(item.get("coverage") or 0.0),
        )

    @staticmethod
    def parse_candidate_key(key: str) -> tuple[str, float, float]:
        model, stable_part, aggressive_part = key.split("@")
        return model, float(stable_part.replace("stable", "")), float(aggressive_part.replace("aggressive", ""))

    @staticmethod
    def model_selection_gate(metrics: dict[str, Any], config: V10MarketGateConfig) -> dict[str, Any]:
        checks = {
            "selection_coverage_at_least_minimum": float(metrics.get("coverage") or 0.0) >= config.min_coverage,
            "selection_calendar_return_positive": float(metrics.get("all_calendar_mean_net_return") or 0.0) > 0.0,
            "selection_eligible_return_positive": float(metrics.get("eligible_mean_net_return") or 0.0) > 0.0,
            "selection_positive_fold_rate_nonzero": float(metrics.get("positive_fold_rate") or 0.0) > 0.0,
        }
        if not config.require_positive_selection_return:
            checks["selection_calendar_return_positive"] = True
            checks["selection_eligible_return_positive"] = True
        return {
            "status": "pass" if all(checks.values()) else "blocked",
            "checks": checks,
            "blocking": [name for name, passed in checks.items() if not passed],
            "policy": "A v10 gate is not promoted when the chronological model-selection segment cannot find a viable admission rule.",
        }

    @staticmethod
    def selection_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        metrics = item.get("selection_metrics") or {}
        return (
            float(metrics.get("all_calendar_mean_net_return") or -999.0),
            float(metrics.get("eligible_mean_net_return") or -999.0),
            float(metrics.get("positive_fold_rate") or 0.0),
            -float(metrics.get("all_loss_rate_at5_on_eligible") or 1.0),
            float(metrics.get("calendar_max_drawdown") or -999.0),
            float(metrics.get("coverage") or 0.0),
        )

    @staticmethod
    def candidate_key(model: str, stable_coverage: float, aggressive_coverage: float) -> str:
        return f"{model}@stable{stable_coverage:.2f}@aggressive{aggressive_coverage:.2f}"

    @staticmethod
    def compare_ranker_baselines(report: dict[str, Any], final_metrics: dict[str, Any]) -> dict[str, Any]:
        baselines = report.get("final_audit_baseline_summary") or []
        categories = {"random": [], "rule": [], "simple_factor": []}
        for row in baselines:
            name = str(row.get("baseline") or "")
            if name == "random":
                categories["random"].append(row)
            elif name.startswith("rule"):
                categories["rule"].append(row)
            elif name.startswith("simple"):
                categories["simple_factor"].append(row)
        candidate_mean = float(final_metrics.get("all_calendar_mean_net_return") or 0.0)
        best_by_category = {}
        deltas = {}
        for category, rows in categories.items():
            if not rows:
                continue
            best = max(rows, key=lambda row: float(row.get("mean_net_return_at5") or -999.0))
            best_by_category[category] = best
            deltas[category] = _round(candidate_mean - float(best.get("mean_net_return_at5") or 0.0))
        best_baseline_mean = max(
            (float(row.get("mean_net_return_at5") or -999.0) for row in baselines),
            default=0.0,
        )
        return {
            "ranker_final_audit_baselines": baselines,
            "best_baseline_by_category": best_by_category,
            "candidate_calendar_minus_baseline": deltas,
            "beats_all_baseline_categories": bool(deltas) and all(value > 0 for value in deltas.values()),
            "best_baseline_mean_net_return_at5": _round(best_baseline_mean),
            "candidate_minus_best_baseline": _round(candidate_mean - best_baseline_mean),
        }

    @staticmethod
    def gate(
        final_metrics: dict[str, Any],
        ungated_metrics: dict[str, Any],
        baseline_comparison: dict[str, Any],
        config: V10MarketGateConfig,
        model_selection_gate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        drawdown_improvement = V10MarketGateResearch.relative_loss_reduction(
            float(ungated_metrics.get("calendar_max_drawdown") or 0.0),
            float(final_metrics.get("calendar_max_drawdown") or 0.0),
        )
        all_loss_improvement = V10MarketGateResearch.relative_loss_reduction(
            float(ungated_metrics.get("all_loss_rate_at5_on_eligible") or 0.0),
            float(final_metrics.get("all_loss_rate_at5_on_eligible") or 0.0),
        )
        checks = {
            "model_selection_gate_passed": (model_selection_gate or {"status": "pass"}).get("status") == "pass",
            "coverage_at_least_25pct": float(final_metrics.get("coverage") or 0.0) >= config.min_coverage,
            "all_calendar_mean_net_return_positive": float(final_metrics.get("all_calendar_mean_net_return") or 0.0) > 0.0,
            "eligible_mean_net_return_positive": float(final_metrics.get("eligible_mean_net_return") or 0.0) > 0.0,
            "positive_fold_rate_at_least_60pct": float(final_metrics.get("positive_fold_rate") or 0.0) >= config.min_positive_fold_rate,
            "drawdown_improves_35pct_vs_ungated_v9": drawdown_improvement >= config.required_drawdown_improvement,
            "all_loss_rate_improves_10pct_vs_ungated_v9": all_loss_improvement >= config.required_all_loss_improvement,
            "beats_ungated_v9_calendar_mean": float(final_metrics.get("all_calendar_mean_net_return") or 0.0) > float(ungated_metrics.get("all_calendar_mean_net_return") or 0.0),
            "beats_random_rule_simple_baseline_categories": bool(baseline_comparison.get("beats_all_baseline_categories")),
        }
        return {
            "status": "candidate" if all(checks.values()) else "observe_only",
            "checks": checks,
            "blocking": [name for name, passed in checks.items() if not passed],
            "drawdown_improvement_vs_ungated_v9": _round(drawdown_improvement),
            "all_loss_improvement_vs_ungated_v9": _round(all_loss_improvement),
            "promotion_allowed": False,
            "policy": "candidate only allows paper observation; usable requires forward settled validation and user-defined capital risk limits.",
        }

    @staticmethod
    def relative_loss_reduction(reference: float, candidate: float) -> float:
        if reference == 0:
            return 0.0
        if reference < 0:
            return (abs(reference) - abs(candidate)) / abs(reference)
        return (reference - candidate) / reference

    @staticmethod
    def failure_window_diagnostics(final_decisions: pd.DataFrame) -> dict[str, Any]:
        frame = final_decisions.copy()
        frame["month"] = frame["trade_date"].astype(str).str.slice(0, 6)
        monthly = []
        for month, rows in frame.groupby("month", sort=True):
            monthly.append(
                {
                    "month": month,
                    "days": int(len(rows)),
                    "decision_counts": dict(sorted(rows["decision"].astype(str).value_counts().to_dict().items())),
                    "mean_net_return_5d": _round(float(pd.to_numeric(rows["mean_net_return_5d"], errors="coerce").mean())),
                    "calendar_mean_net_return_5d": _round(float(pd.to_numeric(rows["calendar_net_return_5d"], errors="coerce").mean())),
                    "all_loss_rate": _round(float(_bool_series(rows["all_loss_5d"]).mean())),
                }
            )
        stress = frame[(frame["trade_date"].astype(str) >= "20260101") & (frame["trade_date"].astype(str) <= "20260331")]
        worst = frame.assign(
            _return=pd.to_numeric(frame["mean_net_return_5d"], errors="coerce")
        ).sort_values("_return").head(15)
        return {
            "monthly": monthly,
            "stress_window_202601_202603": V10MarketGateResearch.evaluate_decisions(stress) if not stress.empty else None,
            "worst_days": [
                {
                    "trade_date": str(row["trade_date"]),
                    "decision": str(row["decision"]),
                    "mean_net_return_5d": _round(float(row.get("mean_net_return_5d") or 0.0)),
                    "calendar_net_return_5d": _round(float(row.get("calendar_net_return_5d") or 0.0)),
                    "all_loss_5d": bool(row.get("all_loss_5d")),
                    "gate_score_stable": _round(float(row.get("gate_score_stable") or 0.0)),
                    "gate_score_aggressive": _round(float(row.get("gate_score_aggressive") or 0.0)),
                }
                for row in worst.to_dict("records")
            ],
        }


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)
