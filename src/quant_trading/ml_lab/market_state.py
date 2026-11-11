"""Research-only market-state eligibility layer for stable Top10 rankings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from ..db import FactorSnapshot, session_scope
from ..factor_lab.metrics import max_drawdown
from .research import MARKET_STATE_FEATURES, MlResearchError, MlResearchHarness


@dataclass(frozen=True)
class MarketStateConfig:
    target_coverage: float = 0.25
    min_coverage: float = 0.25
    min_history_days: int = 60
    min_positive_fold_rate: float = 0.55
    required_drawdown_improvement: float = 0.25
    v5_reference_drawdown: float = -0.128136
    random_state: int = 42


class MarketStateResearch:
    """Nested walk-forward eligibility research; never emits orders or position sizes."""

    def run(
        self,
        ranking_report_path: str | Path,
        baseline_audit_path: str | Path,
        output_dir: str | Path,
        config: MarketStateConfig | None = None,
        report_context: dict[str, Any] | None = None,
        raw_state_factor_set: str = "factor_lab_v2",
    ) -> dict:
        config = config or MarketStateConfig()
        ranking = json.loads(Path(ranking_report_path).read_text(encoding="utf-8"))
        rows = pd.DataFrame(ranking.get("oos_daily_selection") or [])
        if rows.empty:
            raise MlResearchError("Ranking report has no OOS daily selection stream. Re-run research-harness with the v6 contract.")
        daily = self.prepare_daily_rows(rows)
        daily, state_provenance = self.attach_raw_market_state(
            daily,
            factor_set=raw_state_factor_set,
        )
        results = self.walk_forward_eligibility(daily, config)
        summaries = [self.summarize_method(name, rows, config) for name, rows in results.items()]
        baseline_comparison = self.compare_external_baselines(summaries, baseline_audit_path)
        gate = self.gate(summaries, baseline_comparison, config)
        payload = {
            "task": "market_state_eligibility_research",
            "policy": "research-only eligible/no_signal gate; no orders, quantities, positions, or automatic trading instructions",
            "ranking_report_path": str(ranking_report_path),
            "baseline_audit_path": str(baseline_audit_path),
            "config": asdict(config),
            "state_features": list(MARKET_STATE_FEATURES),
            "state_feature_provenance": state_provenance,
            "daily_results": {name: rows.to_dict("records") for name, rows in results.items()},
            "summary": summaries,
            "baseline_comparison": baseline_comparison,
            "gate": gate,
            "diagnostic_only": True,
            "promotion_allowed": False,
        }
        if report_context:
            payload.update(report_context)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "market_state_report.json"
        summary_path = directory / "market_state_summary.csv"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
        for name, output in results.items():
            output.to_csv(directory / f"market_state_{name}_daily.csv", index=False, encoding="utf-8-sig")
        return {**payload, "report_path": str(report_path), "summary_path": str(summary_path)}

    @staticmethod
    def attach_raw_market_state(daily: pd.DataFrame, factor_set: str) -> tuple[pd.DataFrame, dict]:
        """Replace cross-section-normalized macro fields with persisted raw Tushare inputs."""
        macro_features = [
            "feature_market_return_1d", "feature_market_return_5d", "feature_market_return_20d",
            "feature_market_drawdown_20d", "feature_market_volatility_20d",
        ]
        values: dict[str, dict[str, float]] = {}
        aggregates: dict[str, dict[str, float]] = {}
        dates = daily["trade_date"].astype(str).tolist()
        with session_scope() as session:
            query = (
                session.query(FactorSnapshot.trade_date, FactorSnapshot.raw_factors)
                .filter(FactorSnapshot.factor_set == factor_set, FactorSnapshot.trade_date.in_(dates))
                .yield_per(10_000)
            )
            for trade_date, raw in query:
                date = str(trade_date)
                source = raw or {}
                bucket = values.setdefault(date, {})
                for feature in macro_features:
                    name = feature.replace("feature_", "", 1)
                    value = source.get(name)
                    if value is not None and feature not in bucket:
                        bucket[feature] = float(value)
                stats = aggregates.setdefault(date, {"return_count": 0.0, "return_positive": 0.0, "return_sum": 0.0, "return_square_sum": 0.0, "flow_count": 0.0, "flow_positive": 0.0, "pressure_count": 0.0, "pressure_sum": 0.0})
                return_5d = source.get("return_5d")
                if return_5d is not None:
                    value = float(return_5d)
                    stats["return_count"] += 1.0
                    stats["return_positive"] += float(value > 0)
                    stats["return_sum"] += value
                    stats["return_square_sum"] += value * value
                # v2 snapshots persist the real five-day main-net amount, not its derived turnover ratio.
                flow = source.get("main_net_amount_sum_5d")
                if flow is not None:
                    value = float(flow)
                    stats["flow_count"] += 1.0
                    stats["flow_positive"] += float(value > 0)
                up, down = source.get("limit_up_count_5d"), source.get("limit_down_count_5d")
                if up is not None and down is not None:
                    stats["pressure_count"] += 1.0
                    stats["pressure_sum"] += float(up) - float(down)
        state_rows = []
        for date in dates:
            row = {"trade_date": date, **values.get(date, {})}
            stats = aggregates.get(date, {})
            return_count = stats.get("return_count", 0.0)
            mean_return = stats.get("return_sum", 0.0) / return_count if return_count else None
            variance = stats.get("return_square_sum", 0.0) / return_count - mean_return * mean_return if return_count and mean_return is not None else None
            row["feature_market_breadth_return_5d"] = stats.get("return_positive", 0.0) / return_count if return_count else None
            row["feature_market_breadth_flow_5d"] = stats.get("flow_positive", 0.0) / stats.get("flow_count", 0.0) if stats.get("flow_count", 0.0) else None
            row["feature_market_return_dispersion_5d"] = max(variance, 0.0) ** 0.5 if variance is not None else None
            row["feature_limit_pressure_5d"] = stats.get("pressure_sum", 0.0) / stats.get("pressure_count", 0.0) if stats.get("pressure_count", 0.0) else None
            state_rows.append(row)
        state = pd.DataFrame(state_rows).drop_duplicates("trade_date")
        merged = daily.drop(columns=[feature for feature in MARKET_STATE_FEATURES if feature in daily.columns]).merge(state, on="trade_date", how="left")
        coverage = {feature: round(float(merged[feature].notna().mean()), 6) if feature in merged else 0.0 for feature in MARKET_STATE_FEATURES}
        if any(rate < 0.95 for rate in coverage.values()):
            raise MlResearchError(f"Raw market-state feature coverage below 95%: {coverage}")
        return merged, {
            "source": "factor_snapshot.raw_factors; real Tushare-backed values persisted at the signal date and aggregated across the full market",
            "factor_set": factor_set,
            "coverage": coverage,
            "normalized_training_frame_macro_features_used": False,
        }

    @staticmethod
    def prepare_daily_rows(rows: pd.DataFrame) -> pd.DataFrame:
        required = {"trade_date", "fold_index", "mean_net_return", "selection_positive"}
        missing = sorted(required - set(rows.columns))
        if missing:
            raise MlResearchError(f"OOS daily selection stream missing columns: {missing}")
        daily = rows.copy()
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily["fold_index"] = pd.to_numeric(daily["fold_index"], errors="coerce").astype(int)
        if daily["trade_date"].duplicated().any():
            raise MlResearchError("OOS daily selection stream contains duplicated trade dates; overlapping validation windows are not permitted.")
        daily["mean_net_return"] = pd.to_numeric(daily["mean_net_return"], errors="coerce")
        daily["selection_positive"] = pd.to_numeric(daily["selection_positive"], errors="coerce").fillna(0).astype(int)
        return daily.sort_values(["fold_index", "trade_date"]).reset_index(drop=True)

    def walk_forward_eligibility(self, daily: pd.DataFrame, config: MarketStateConfig) -> dict[str, pd.DataFrame]:
        methods = {"unconditional_v5": [], "rule": [], "lightgbm": []}
        features = [column for column in MARKET_STATE_FEATURES if column in daily.columns]
        if not features:
            raise MlResearchError("No signal-date market-state features are present in the OOS daily selection stream.")
        for fold_index, valid in daily.groupby("fold_index", sort=True):
            train = daily[daily["fold_index"] < fold_index].copy()
            for _, row in valid.iterrows():
                methods["unconditional_v5"].append({**row.to_dict(), "decision": "eligible", "eligible": True, "calendar_return": row["mean_net_return"], "reason": "unconditional"})
            if len(train) < config.min_history_days or train["selection_positive"].nunique() < 2:
                for name in ("rule", "lightgbm"):
                    methods[name].extend(self.no_signal_rows(valid, "insufficient_prior_oos_history"))
                continue
            rule_scores, valid_rule_scores = self.rule_scores(train, valid, features)
            rule_threshold = float(rule_scores.quantile(1.0 - config.target_coverage))
            methods["rule"].extend(self.decision_rows(valid, valid_rule_scores >= rule_threshold, "rule_threshold"))
            probabilities = self.model_probabilities(train, valid, features, config)
            train_probabilities = self.model_probabilities(train, train, features, config)
            model_threshold = float(pd.Series(train_probabilities).quantile(1.0 - config.target_coverage))
            methods["lightgbm"].extend(self.decision_rows(valid, probabilities >= model_threshold, "model_threshold"))
        return {name: pd.DataFrame(rows) for name, rows in methods.items()}

    @staticmethod
    def rule_scores(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
        higher_is_safer = {
            "feature_market_return_1d", "feature_market_return_5d", "feature_market_return_20d", "feature_market_drawdown_20d",
            "feature_market_breadth_return_5d", "feature_market_breadth_flow_5d", "feature_market_return_dispersion_5d", "feature_limit_pressure_5d",
        }
        train_scores = pd.Series(0.0, index=train.index)
        valid_scores = pd.Series(0.0, index=valid.index)
        for feature in features:
            reference = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy()
            if len(reference) == 0:
                continue
            train_values = pd.to_numeric(train[feature], errors="coerce")
            valid_values = pd.to_numeric(valid[feature], errors="coerce")
            train_rank = train_values.map(lambda value: float(np.mean(reference <= value)) if pd.notna(value) else 0.0)
            valid_rank = valid_values.map(lambda value: float(np.mean(reference <= value)) if pd.notna(value) else 0.0)
            if feature not in higher_is_safer:
                train_rank, valid_rank = 1.0 - train_rank, 1.0 - valid_rank
            train_scores += train_rank
            valid_scores += valid_rank
        return train_scores / len(features), valid_scores / len(features)

    @staticmethod
    def model_probabilities(train: pd.DataFrame, predict: pd.DataFrame, features: list[str], config: MarketStateConfig) -> np.ndarray:
        from lightgbm import LGBMClassifier

        x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_predict = predict[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model = LGBMClassifier(n_estimators=60, learning_rate=0.04, max_depth=2, num_leaves=7, min_child_samples=20, random_state=config.random_state, verbose=-1)
        model.fit(x_train, train["selection_positive"])
        return model.predict_proba(x_predict)[:, 1]

    @staticmethod
    def no_signal_rows(rows: pd.DataFrame, reason: str) -> list[dict]:
        return MarketStateResearch.decision_rows(rows, pd.Series(False, index=rows.index), reason)

    @staticmethod
    def decision_rows(rows: pd.DataFrame, eligible: pd.Series | np.ndarray, reason: str) -> list[dict]:
        output = rows.copy()
        output["eligible"] = np.asarray(eligible, dtype=bool)
        output["decision"] = np.where(output["eligible"], "eligible", "no_signal")
        output["calendar_return"] = np.where(output["eligible"], output["mean_net_return"], 0.0)
        output["reason"] = reason
        return output.to_dict("records")

    @staticmethod
    def summarize_method(name: str, rows: pd.DataFrame, config: MarketStateConfig) -> dict:
        if rows.empty:
            return {"method": name, "status": "rejected", "coverage_rate": 0.0}
        calendar = pd.to_numeric(rows["calendar_return"], errors="coerce").fillna(0.0)
        eligible = rows[rows["eligible"]]
        eligible_return = pd.to_numeric(eligible["mean_net_return"], errors="coerce")
        fold_returns = rows.groupby("fold_index")["calendar_return"].mean()
        return {
            "method": name,
            "days": int(len(rows)),
            "eligible_days": int(len(eligible)),
            "coverage_rate": round(float(rows["eligible"].mean()), 6),
            "calendar_mean_net_return": round(float(calendar.mean()), 6),
            "eligible_mean_net_return": round(float(eligible_return.mean()), 6) if not eligible.empty else None,
            "eligible_positive_day_rate": round(float((eligible_return > 0).mean()), 6) if not eligible.empty else None,
            "positive_fold_rate": round(float((fold_returns > 0).mean()), 6),
            "calendar_max_drawdown": round(float(max_drawdown(calendar.tolist())), 6),
            "no_signal_days": int((~rows["eligible"]).sum()),
        }

    @staticmethod
    def compare_external_baselines(summaries: list[dict], baseline_audit_path: str | Path) -> dict:
        report = json.loads(Path(baseline_audit_path).read_text(encoding="utf-8"))
        baselines = report.get("baseline_comparison", {}).get("best_baseline_by_category", {})
        candidate = next((item for item in summaries if item.get("method") == "lightgbm"), {})
        candidate_return = float(candidate.get("calendar_mean_net_return") or 0.0)
        deltas = {category: round(candidate_return - float(row.get("avg_mean_net_return") or 0.0), 6) for category, row in baselines.items()}
        return {"baselines": baselines, "candidate_minus_baseline": deltas, "beats_all_baseline_categories": bool(deltas) and all(value > 0 for value in deltas.values())}

    @staticmethod
    def gate(summaries: list[dict], comparison: dict, config: MarketStateConfig) -> dict:
        candidate = next((item for item in summaries if item.get("method") == "lightgbm"), {})
        unconditional = next((item for item in summaries if item.get("method") == "unconditional_v5"), {})
        reference_drawdown = float(unconditional.get("calendar_max_drawdown") or config.v5_reference_drawdown)
        drawdown_limit = reference_drawdown * (1.0 - config.required_drawdown_improvement)
        checks = {
            "coverage_at_least_minimum": float(candidate.get("coverage_rate") or 0.0) >= config.min_coverage,
            "calendar_return_positive": float(candidate.get("calendar_mean_net_return") or 0.0) > 0,
            "eligible_return_positive": float(candidate.get("eligible_mean_net_return") or 0.0) > 0,
            "positive_fold_rate_at_least_threshold": float(candidate.get("positive_fold_rate") or 0.0) >= config.min_positive_fold_rate,
            "drawdown_improved": float(candidate.get("calendar_max_drawdown") or 0.0) >= drawdown_limit,
            "beats_all_baseline_categories": bool(comparison.get("beats_all_baseline_categories")),
        }
        status = "candidate" if all(checks.values()) else "observe_only"
        return {"status": status, "checks": checks, "blocking": [name for name, passed in checks.items() if not passed], "reference_drawdown": round(reference_drawdown, 6), "drawdown_limit": round(drawdown_limit, 6), "policy": "eligible/no_signal only; no production promotion without an independent candidate review"}
