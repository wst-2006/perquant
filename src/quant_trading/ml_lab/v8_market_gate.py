"""Chronologically isolated market-entry research for v8 OOS Top5 streams."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..factor_lab.metrics import max_drawdown
from .market_state import MarketStateResearch
from .research import MARKET_STATE_FEATURES


V8_MARKET_GATE_CONTRACT = "ml_v8_market_gate_research_v1"


class V8MarketGateError(RuntimeError):
    """Raised when an audited v8 market gate cannot be evaluated."""


@dataclass(frozen=True)
class V8MarketGateConfig:
    base_train_days: int = 40
    model_selection_days: int = 20
    final_audit_days: int = 40
    target_coverages: tuple[float, ...] = (0.25, 0.40, 0.55)
    min_coverage: float = 0.25
    random_state: int = 42


class V8MarketGateResearch:
    def run(
        self,
        ranking_report_path: str | Path,
        output_dir: str | Path,
        config: V8MarketGateConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or V8MarketGateConfig()
        report_path = Path(ranking_report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        daily = pd.DataFrame(report.get("oos_daily") or []).sort_values("trade_date").reset_index(drop=True)
        required_days = cfg.base_train_days + cfg.model_selection_days + cfg.final_audit_days
        if len(daily) < required_days:
            raise V8MarketGateError(f"Need at least {required_days} ranker OOS days, found {len(daily)}")
        daily = daily.iloc[-required_days:].copy()
        daily["selection_positive"] = (
            (pd.to_numeric(daily["mean_net_return_5d"], errors="coerce") > 0)
            & daily["hit5pct_10d"].astype(str).str.lower().isin({"true", "1"})
        ).astype(int)
        state, provenance = MarketStateResearch.attach_raw_market_state(daily, factor_set="factor_lab_v2")
        state = state.sort_values("trade_date").reset_index(drop=True)
        base = state.iloc[: cfg.base_train_days].copy()
        selection = state.iloc[cfg.base_train_days : cfg.base_train_days + cfg.model_selection_days].copy()
        final = state.iloc[-cfg.final_audit_days :].copy()
        if base["selection_positive"].nunique() < 2:
            raise V8MarketGateError("Base market-gate training segment has a single target class")

        candidates = {}
        fitted = {}
        for name in ("rule", "logistic", "lightgbm"):
            model = self.fit_candidate(name, base, cfg)
            train_scores = self.predict_candidate(name, model, base)
            for target_coverage in cfg.target_coverages:
                threshold = float(pd.Series(train_scores).quantile(1.0 - target_coverage))
                selection_scores = self.predict_candidate(name, model, selection)
                metrics = self.evaluate_decisions(selection, selection_scores >= threshold)
                key = f"{name}@{target_coverage:.2f}"
                candidates[key] = {
                    "model": name,
                    "target_coverage": target_coverage,
                    "threshold": threshold,
                    "selection_metrics": metrics,
                }
                fitted[key] = model
        viable = [
            key for key, item in candidates.items()
            if item["selection_metrics"]["coverage"] >= cfg.min_coverage
        ]
        selection_pool = viable or list(candidates)
        selected_key = max(
            selection_pool,
            key=lambda key: (
                candidates[key]["selection_metrics"]["all_calendar_mean_net_return"],
                candidates[key]["selection_metrics"]["eligible_mean_net_return"],
                candidates[key]["selection_metrics"]["coverage"],
                candidates[key]["selection_metrics"]["max_drawdown"],
            ),
        )
        selected_name = candidates[selected_key]["model"]
        selected_target_coverage = float(candidates[selected_key]["target_coverage"])

        training = pd.concat([base, selection], ignore_index=True)
        final_model = self.fit_candidate(selected_name, training, cfg)
        training_scores = self.predict_candidate(selected_name, final_model, training)
        final_threshold = float(pd.Series(training_scores).quantile(1.0 - selected_target_coverage))
        final_scores = self.predict_candidate(selected_name, final_model, final)
        final_metrics = self.evaluate_decisions(final, final_scores >= final_threshold)
        ungated_metrics = self.evaluate_decisions(final, np.ones(len(final), dtype=bool))
        baseline_mean = max(
            (float(row.get("mean_net_return_at5") or 0.0) for row in report.get("final_audit_baseline_summary") or []),
            default=0.0,
        )
        drawdown_improvement = self.drawdown_improvement(
            ungated_metrics["max_drawdown"], final_metrics["max_drawdown"]
        )
        checks = {
            "coverage_at_least_25pct": final_metrics["coverage"] >= cfg.min_coverage,
            "all_calendar_mean_net_return_positive": final_metrics["all_calendar_mean_net_return"] > 0.0,
            "eligible_mean_net_return_positive": final_metrics["eligible_mean_net_return"] > 0.0,
            "positive_fold_rate_at_least_55pct": final_metrics["positive_fold_rate"] >= 0.55,
            "beats_ungated_all_calendar_mean": final_metrics["all_calendar_mean_net_return"] > ungated_metrics["all_calendar_mean_net_return"],
            "eligible_mean_beats_best_baseline": final_metrics["eligible_mean_net_return"] > baseline_mean,
            "drawdown_improves_25pct_vs_ungated": drawdown_improvement >= 0.25,
        }
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "market_gate.joblib"
        if selected_name != "rule":
            joblib.dump(final_model, model_path)
        final_daily = final[["trade_date", "mean_net_return_5d", "hit5pct_10d", "all_loss_5d"]].copy()
        final_daily["gate_score"] = final_scores
        final_daily["decision"] = np.where(final_scores >= final_threshold, "eligible", "no_signal")
        final_daily["all_calendar_net_return"] = np.where(
            final_daily["decision"] == "eligible",
            pd.to_numeric(final_daily["mean_net_return_5d"], errors="coerce").fillna(0.0),
            0.0,
        )
        daily_path = output / "v8_market_gate_oos_daily.csv"
        final_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        payload = {
            "task": "v8_market_gate_research",
            "report_contract": V8_MARKET_GATE_CONTRACT,
            "ranking_report_path": str(report_path),
            "ranking_model": report.get("best_model"),
            "config": cfg.__dict__,
            "split": {
                "base_train": [str(base["trade_date"].iloc[0]), str(base["trade_date"].iloc[-1])],
                "model_selection": [str(selection["trade_date"].iloc[0]), str(selection["trade_date"].iloc[-1])],
                "final_audit": [str(final["trade_date"].iloc[0]), str(final["trade_date"].iloc[-1])],
                "status": "pass",
            },
            "candidate_selection": candidates,
            "selection_coverage_constraint_satisfied": bool(viable),
            "selected_gate_key": selected_key,
            "selected_gate": selected_name,
            "selected_target_coverage": selected_target_coverage,
            "threshold": final_threshold,
            "model_path": str(model_path) if selected_name != "rule" else None,
            "features": list(MARKET_STATE_FEATURES),
            "state_provenance": provenance,
            "final_audit": final_metrics,
            "ungated_final_audit": ungated_metrics,
            "best_baseline_mean_net_return_at5": baseline_mean,
            "drawdown_improvement_vs_ungated": drawdown_improvement,
            "gate": {
                "status": "diagnostic_pass" if all(checks.values()) else "observe_only",
                "checks": checks,
                "blocking": [name for name, passed in checks.items() if not passed],
                "promotion_allowed": False,
            },
            "diagnostic_only": True,
            "promotion_allowed": False,
            "policy": "Market participation only; no_signal is zero return on the full OOS calendar and never produces an order.",
            "daily_path": str(daily_path),
        }
        output_report = output / "v8_market_gate_report.json"
        output_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {**payload, "report_path": str(output_report)}

    @staticmethod
    def fit_candidate(name: str, frame: pd.DataFrame, config: V8MarketGateConfig) -> Any:
        x = V8MarketGateResearch.clean_features(frame)
        y = frame["selection_positive"].astype(int)
        if name == "rule":
            return {
                column: {
                    "median": float(x[column].median()),
                    "scale": max(float(x[column].quantile(0.75) - x[column].quantile(0.25)), 1e-9),
                }
                for column in MARKET_STATE_FEATURES
            }
        if name == "logistic":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=2000, random_state=config.random_state))
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=60,
                learning_rate=0.03,
                max_depth=2,
                num_leaves=5,
                min_child_samples=12,
                reg_lambda=2.0,
                random_state=config.random_state,
                verbose=-1,
            )
        else:
            raise V8MarketGateError(f"Unsupported market gate candidate: {name}")
        model.fit(x, y)
        return model

    @staticmethod
    def predict_candidate(name: str, model: Any, frame: pd.DataFrame) -> np.ndarray:
        if name == "rule":
            higher = [
                "feature_market_return_5d",
                "feature_market_return_20d",
                "feature_market_drawdown_20d",
                "feature_market_breadth_return_5d",
                "feature_market_breadth_flow_5d",
                "feature_limit_pressure_5d",
            ]
            lower = ["feature_market_volatility_20d", "feature_market_return_dispersion_5d"]
            parts = []
            for column in higher:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(model[column]["median"])
                parts.append((values - model[column]["median"]) / model[column]["scale"])
            for column in lower:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(model[column]["median"])
                parts.append(-(values - model[column]["median"]) / model[column]["scale"])
            return pd.concat(parts, axis=1).mean(axis=1).to_numpy(dtype=float)
        return np.asarray(model.predict_proba(V8MarketGateResearch.clean_features(frame)))[:, 1]

    @staticmethod
    def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[list(MARKET_STATE_FEATURES)].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    @staticmethod
    def evaluate_decisions(frame: pd.DataFrame, eligible: np.ndarray) -> dict[str, Any]:
        decisions = np.asarray(eligible, dtype=bool)
        returns = pd.to_numeric(frame["mean_net_return_5d"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        calendar = np.where(decisions, returns, 0.0)
        eligible_returns = returns[decisions]
        hit = frame["hit5pct_10d"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
        all_loss = frame["all_loss_5d"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
        fold_means = [float(np.mean(chunk)) for chunk in np.array_split(calendar, max(1, int(np.ceil(len(calendar) / 20))))]
        return {
            "days": int(len(frame)),
            "eligible_days": int(decisions.sum()),
            "coverage": round(float(decisions.mean()), 6),
            "all_calendar_mean_net_return": round(float(calendar.mean()), 6),
            "eligible_mean_net_return": round(float(eligible_returns.mean()), 6) if len(eligible_returns) else 0.0,
            "positive_fold_rate": round(float(np.mean([value > 0 for value in fold_means])), 6),
            "hit5pct_10d_at5_on_eligible": round(float(hit[decisions].mean()), 6) if decisions.any() else 0.0,
            "all_loss_rate_at5_on_eligible": round(float(all_loss[decisions].mean()), 6) if decisions.any() else 1.0,
            "max_drawdown": round(float(max_drawdown(calendar.tolist())), 6),
            "no_signal_return_policy": "zero_return_on_full_calendar",
        }

    @staticmethod
    def drawdown_improvement(ungated: float, gated: float) -> float:
        if ungated >= 0:
            return 0.0
        return round((abs(ungated) - abs(gated)) / abs(ungated), 6)
