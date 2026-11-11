"""Versioned v8 inference, paper candidates, settlement, and manual review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from ..db import (
    DailyQuote,
    MlV8CandidateOutcome,
    MlV8CandidateSignal,
    MlV8ManualReview,
    StockLimitDaily,
    session_scope,
)
from ..services.foundation_gate import FoundationGateConfig, inspect_diagnostic_foundation
from .market_state import MarketStateResearch
from .research import MARKET_STATE_FEATURES
from .short_execution import compute_short_execution_label
from .v8_features import build_live_v8_feature_frame
from .v8_labels import V8_HORIZONS, compute_path_labels
from .v8_metrics import diversified_top5, dual_head_top5
from .v8_research import V11_DUAL_VARIANT, V8_HEADS, V8ResearchHarness
from .v10_market_gate import V10MarketGateResearch


class V8InferenceError(RuntimeError):
    """Raised when a v8 paper inference contract is not satisfied."""


CRITICAL_DRIFT_FEATURES = frozenset(
    {
        "feature_limit_down_count_5d",
        "feature_limit_down_count_10d",
        "feature_limit_down_count_20d",
        "feature_limit_pressure_5d",
        "feature_cs_rank_limit_pressure_5d",
        "feature_v9_limit_down_count_20d",
        "feature_v9_limit_pressure_rank",
    }
)


@dataclass
class V8InferenceService:
    artifact_path: str | Path = "reports/ml_lab/v8_artifact/artifact.json"

    def __post_init__(self) -> None:
        self.artifact_path = Path(self.artifact_path)
        if not self.artifact_path.exists():
            raise V8InferenceError(f"v8 artifact not found: {self.artifact_path}")
        self.artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        if self.artifact.get("artifact_contract") != "ml_v8_model_artifact_v1":
            raise V8InferenceError(f"Unsupported v8 artifact contract: {self.artifact.get('artifact_contract')}")

    def status(self) -> dict[str, Any]:
        version = self.artifact["model_version"]
        with session_scope() as session:
            version_query = session.query(MlV8CandidateSignal).filter(MlV8CandidateSignal.model_version == version)
            observed_days = version_query.with_entities(func.count(func.distinct(MlV8CandidateSignal.signal_date))).scalar() or 0
            observed_rows = version_query.with_entities(func.count()).scalar() or 0
            paper_days = version_query.filter(MlV8CandidateSignal.decision == "eligible_paper").with_entities(
                func.count(func.distinct(MlV8CandidateSignal.signal_date))
            ).scalar() or 0
            paper_rows = version_query.filter(MlV8CandidateSignal.decision == "eligible_paper").with_entities(func.count()).scalar() or 0
            settled_rows = (
                session.query(func.count())
                .select_from(MlV8CandidateOutcome)
                .join(MlV8CandidateSignal, MlV8CandidateSignal.id == MlV8CandidateOutcome.signal_id)
                .filter(MlV8CandidateSignal.model_version == version)
                .scalar()
                or 0
            )
            manual_rows = (
                session.query(func.count())
                .select_from(MlV8ManualReview)
                .join(MlV8CandidateSignal, MlV8CandidateSignal.id == MlV8ManualReview.signal_id)
                .filter(MlV8CandidateSignal.model_version == version)
                .scalar()
                or 0
            )
        outcomes = self.outcome_summary()
        return {
            "task": "v8_status",
            "model_version": self.artifact["model_version"],
            "research_gate": self.artifact.get("research_gate"),
            "market_gate": self.artifact.get("market_gate"),
            "diagnostic_only": bool(self.artifact.get("diagnostic_only", True)),
            "promotion_allowed": False,
            "observed_signal_days": int(observed_days),
            "observed_signal_rows": int(observed_rows),
            "latest_observed_signal_date": self._latest_observed_signal_date(version),
            "paper_signal_days": int(paper_days),
            "paper_signal_rows": int(paper_rows),
            "settled_outcome_rows": int(settled_rows),
            "manual_review_rows": int(manual_rows),
            "usable_minimum_signal_days": 40,
            "paper_outcomes": outcomes,
            "research_top5_outcomes": self.outcome_summary(
                decisions={"eligible_paper", "diagnostic_only", "no_signal"},
                domain="research_top5",
            ),
        }

    def score_date(self, trade_date: str, *, persist: bool = True) -> dict[str, Any]:
        foundation = inspect_diagnostic_foundation(
            FoundationGateConfig(start_date="20240401", end_date=trade_date),
            factor_set=self.artifact.get("feature_set") or "factor_lab_v8",
        )
        if foundation.get("gate_status") != "pass":
            raise V8InferenceError(f"v8 diagnostic foundation blocked: {foundation.get('gate_blocking')}")
        feature_columns = list(self.artifact["feature_columns"])
        frame = build_live_v8_feature_frame(trade_date, feature_columns)
        data_drift = self._feature_drift(frame, feature_columns)
        scored = self._ensemble_predictions(frame, feature_columns)
        diagnostic = self._top5_for_artifact(scored)
        market_gate = self._market_gate_decision(trade_date)
        paper_eligibility = self._paper_eligibility(self.artifact.get("research_gate") or {}, market_gate, data_drift)
        paper_eligible = all(paper_eligibility.values())
        raw_market_decision = market_gate.get("raw_decision") or market_gate.get("decision")
        decision = (
            "eligible_paper"
            if paper_eligible
            else "no_signal"
            if raw_market_decision == "no_signal" or not paper_eligibility["data_drift_passed"]
            else "diagnostic_only"
        )
        signal_day_context = self._signal_day_context(trade_date, diagnostic["ts_code"].astype(str).tolist())
        candidates = [
            self._candidate_payload(
                row,
                decision,
                data_drift,
                signal_day_context.get(str(row.get("ts_code"))),
            )
            for row in diagnostic.to_dict("records")
        ]
        paper_ledger = self._build_v12_paper_ledger(
            candidates,
            paper_eligible=paper_eligible,
            market_gate=market_gate,
        )
        if persist:
            self._persist_candidates(trade_date, candidates)
        return {
            "task": "v8_score_date",
            "signal_date": trade_date,
            "model_version": self.artifact["model_version"],
            "full_market_rows": int(len(frame)),
            "research_gate": self.artifact.get("research_gate"),
            "market_gate": market_gate,
            "decision": decision,
            "paper_eligible": paper_eligible,
            "paper_eligibility_checks": paper_eligibility,
            "data_drift": data_drift,
            "candidates": candidates if paper_eligible else [],
            "diagnostic_candidates": candidates,
            "paper_ledger": paper_ledger,
            "diagnostic_only": not paper_eligible,
            "promotion_allowed": False,
            "foundation_gate": foundation,
            "policy": "Top5 opportunity retrieval for manual review only; no orders, quantities, buy/sell instructions, or automatic trading",
        }

    @staticmethod
    def _paper_eligibility(research_gate: dict[str, Any], market_gate: dict[str, Any], data_drift: dict[str, Any]) -> dict[str, bool]:
        eligible_decisions = {"eligible", "eligible_stable", "eligible_aggressive"}
        return {
            "research_gate_candidate": research_gate.get("status") in {"candidate", "usable"},
            "market_gate_eligible": str(market_gate.get("decision") or "") in eligible_decisions,
            "data_drift_passed": data_drift.get("status") == "pass",
        }

    def _top5_for_artifact(self, scored: pd.DataFrame) -> pd.DataFrame:
        if str(self.artifact.get("score_variant") or "composite") == V11_DUAL_VARIANT:
            return dual_head_top5(scored)
        return diversified_top5(scored, score_column="prediction")

    @staticmethod
    def _latest_observed_signal_date(model_version: str) -> str | None:
        with session_scope() as session:
            value = (
                session.query(func.max(MlV8CandidateSignal.signal_date))
                .filter(MlV8CandidateSignal.model_version == model_version)
                .scalar()
            )
        return str(value) if value else None

    def get_candidates(self, signal_date: str, model_version: str | None = None) -> dict[str, Any]:
        version = model_version or self.artifact["model_version"]
        with session_scope() as session:
            rows = (
                session.query(MlV8CandidateSignal)
                .filter(MlV8CandidateSignal.signal_date == signal_date, MlV8CandidateSignal.model_version == version)
                .order_by(MlV8CandidateSignal.rank.asc())
                .all()
            )
        return {
            "signal_date": signal_date,
            "model_version": version,
            "candidates": [self._signal_dict(row) for row in rows],
            "diagnostic_only": bool(self.artifact.get("diagnostic_only", True)),
            "promotion_allowed": False,
        }

    def settle(self, as_of_date: str) -> dict[str, Any]:
        with session_scope() as session:
            signals = (
                session.query(MlV8CandidateSignal)
                .filter(MlV8CandidateSignal.signal_date < as_of_date)
                .order_by(MlV8CandidateSignal.signal_date.asc(), MlV8CandidateSignal.rank.asc())
                .all()
            )
        persisted = 0
        for signal in signals:
            labels = self._labels_for_signal(signal.ts_code, signal.signal_date, as_of_date)
            for horizon in V8_HORIZONS:
                end_date = labels.get(f"label_{horizon}d_end_date")
                if not end_date or str(end_date) > as_of_date:
                    continue
                self._persist_outcome(signal.id, horizon, labels)
                persisted += 1
        return {
            "task": "v8_settle_paper",
            "as_of_date": as_of_date,
            "model_version": self.artifact["model_version"],
            "signal_rows_scanned": len(signals),
            "outcomes_upserted": persisted,
            "summary": self.outcome_summary(),
            "policy": "paper settlement only; no automatic trading action",
        }

    def record_manual_review(self, signal_id: int, selected: bool, reason: str | None = None, notes: str | None = None) -> dict[str, Any]:
        with session_scope() as session:
            signal = session.query(MlV8CandidateSignal).filter(MlV8CandidateSignal.id == signal_id).first()
            if signal is None:
                raise V8InferenceError(f"Unknown v8 signal_id={signal_id}")
            review = MlV8ManualReview(signal_id=signal_id, selected=bool(selected), reason=reason, notes=notes)
            session.add(review)
            session.flush()
            review_id = int(review.id)
        return {
            "task": "v8_manual_review",
            "id": review_id,
            "signal_id": signal_id,
            "selected": bool(selected),
            "training_policy": "manual choice is stored for comparison and personalization analysis; it is not a market outcome label",
        }

    def outcome_summary(self, decisions: set[str] | None = None, *, domain: str = "default_70_30_portfolio") -> dict[str, Any]:
        with session_scope() as session:
            query = (
                session.query(MlV8CandidateOutcome, MlV8CandidateSignal)
                .join(MlV8CandidateSignal, MlV8CandidateSignal.id == MlV8CandidateOutcome.signal_id)
                .filter(MlV8CandidateSignal.model_version == self.artifact["model_version"])
            )
            if decisions is None:
                decisions = {"eligible_paper"}
            query = query.filter(MlV8CandidateSignal.decision.in_(sorted(decisions)))
            rows = query.all()
        if not rows:
            return {
                "domain": domain,
                "signal_days": 0,
                "horizons": {},
                "usable": False,
                "minimum_signal_days": 40,
                "checks": {"minimum_settled_days": False},
            }
        records = [
            {
                "signal_date": signal.signal_date,
                "horizon": outcome.horizon,
                "net_return": outcome.net_return,
                "hit_5pct": outcome.hit_5pct,
                "hit_8pct": outcome.hit_8pct,
            }
            for outcome, signal in rows
        ]
        frame = pd.DataFrame(records)
        summaries = {}
        for horizon, group in frame.groupby("horizon"):
            daily = group.groupby("signal_date").agg(
                mean_net_return=("net_return", "mean"),
                hit5=("hit_5pct", "max"),
                hit8=("hit_8pct", "max"),
            )
            summaries[str(horizon)] = {
                "signal_days": int(len(daily)),
                "mean_net_return_at5": _safe_float(daily["mean_net_return"].mean()),
                "positive_day_rate": _safe_float((daily["mean_net_return"] > 0).mean()),
                "hit5pct_at5": _safe_float(daily["hit5"].mean()),
                "hit8pct_at5": _safe_float(daily["hit8"].mean()),
            }
        signal_days = int(frame["signal_date"].nunique())
        five = summaries.get("5") or {}
        ten = summaries.get("10") or {}
        reference_hit = float((self.artifact.get("research_reference") or {}).get("hit5pct_10d_at5") or 0.0)
        checks = {
            "five_day_settled_signal_days_at_least_40": int(five.get("signal_days") or 0) >= 40,
            "ten_day_settled_signal_days_at_least_40": int(ten.get("signal_days") or 0) >= 40,
            "five_day_mean_net_return_positive": float(five.get("mean_net_return_at5") or 0.0) > 0.0,
            "hit5pct_10d_retains_90pct_of_oos": float(ten.get("hit5pct_at5") or 0.0) >= reference_hit * 0.90,
            "no_severe_data_drift": self._paper_drift_passes(),
        }
        return {
            "domain": domain,
            "signal_days": signal_days,
            "horizons": summaries,
            "usable": all(checks.values()),
            "minimum_signal_days": 40,
            "checks": checks,
            "oos_reference_hit5pct_10d_at5": reference_hit,
        }

    def _ensemble_predictions(self, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        family_scores = []
        v11_stable_scores = []
        v11_aggressive_scores = []
        calibrated: dict[str, list[np.ndarray]] = {name: [] for name in V8_HEADS}
        score_variant = str(self.artifact.get("score_variant") or "composite")
        for model_name, heads in self.artifact["models"].items():
            head_scores: dict[str, np.ndarray] = {}
            for head_name, model_path in heads.items():
                model = joblib.load(model_path)
                spec = V8_HEADS[head_name]
                raw = V8ResearchHarness.predict_head(model, frame, feature_columns, spec)
                head_scores[head_name] = raw
                calibration = (self.artifact.get("calibrations") or {}).get(model_name, {}).get(head_name)
                calibrated[head_name].append(_apply_calibration(raw, calibration))
            variants = V8ResearchHarness.score_variants(frame[["trade_date", "ts_code"]].copy(), head_scores)
            if score_variant not in variants:
                raise V8InferenceError(f"Unsupported artifact score variant: {score_variant}")
            family = variants[score_variant]
            family_values = family["prediction"]
            if score_variant == V11_DUAL_VARIANT:
                v11_stable_scores.append(family["v11_stable_score"].to_numpy(dtype=float))
                v11_aggressive_scores.append(family["v11_aggressive_score"].to_numpy(dtype=float))
            if self.artifact.get("selected_research_model") == "ensemble":
                family_values = family_values.groupby(frame["trade_date"].astype(str), sort=False).rank(pct=True)
            family_scores.append(family_values.to_numpy())
        if not family_scores:
            raise V8InferenceError("v8 artifact contains no model families")
        output = frame.copy()
        matrix = np.vstack(family_scores)
        output["prediction"] = matrix.mean(axis=0)
        output["model_disagreement"] = matrix.std(axis=0)
        if v11_stable_scores and v11_aggressive_scores:
            output["v11_stable_score"] = np.vstack(v11_stable_scores).mean(axis=0)
            output["v11_aggressive_score"] = np.vstack(v11_aggressive_scores).mean(axis=0)
        for head_name, arrays in calibrated.items():
            if arrays:
                output[f"estimate_{head_name}"] = np.vstack(arrays).mean(axis=0)
        return output

    def _market_gate_decision(self, trade_date: str) -> dict[str, Any]:
        gate = self.artifact.get("market_gate") or {}
        if gate.get("status") not in {"trained", "diagnostic_observe_only"}:
            return {"decision": "unavailable", **gate}
        if not gate.get("model_path"):
            return {"decision": "unavailable", **gate, "reason": "packaged_market_gate_model_unavailable"}
        daily = pd.DataFrame([{"trade_date": trade_date}])
        try:
            if gate.get("report_contract") == "ml_v10_market_gate_research_v1":
                state, provenance = V10MarketGateResearch.attach_v10_market_state(daily, factor_set="factor_lab_v2")
            else:
                state, provenance = MarketStateResearch.attach_raw_market_state(daily, factor_set="factor_lab_v2")
        except Exception as exc:
            return {"decision": "unavailable", "status": "error", "reason": str(exc)}
        model = joblib.load(gate["model_path"])
        if gate.get("report_contract") == "ml_v10_market_gate_research_v1":
            model_name = str(model.get("model_name") or gate.get("model_type") or "")
            scores = V10MarketGateResearch.predict_candidate(model_name, model["model"], state)
            thresholds = model.get("thresholds") or gate.get("thresholds") or {}
            if "mean_net_return_5d" not in state.columns:
                state = state.assign(mean_net_return_5d=0.0)
            decided = V10MarketGateResearch.decision_frame(state, scores, thresholds)
            raw_decision = str(decided["decision"].iloc[0])
            decision = raw_decision if gate.get("status") == "trained" else f"diagnostic_{raw_decision}"
            return {
                "status": "evaluated",
                "decision": decision,
                "raw_decision": raw_decision,
                "score_stable": _safe_float(decided["gate_score_stable"].iloc[0]),
                "score_aggressive": _safe_float(decided["gate_score_aggressive"].iloc[0]),
                "threshold_stable": _safe_float(decided["threshold_stable"].iloc[0]),
                "threshold_aggressive": _safe_float(decided["threshold_aggressive"].iloc[0]),
                "decision_contract": "eligible_stable/eligible_aggressive/no_signal",
                "state_provenance": provenance,
            }
        features = gate.get("features") or list(MARKET_STATE_FEATURES)
        x = state[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        probability = float(model.predict_proba(x)[0, 1])
        threshold = float(gate["threshold"])
        raw_decision = "eligible" if probability >= threshold else "no_signal"
        decision = raw_decision if gate.get("status") == "trained" else f"diagnostic_{raw_decision}"
        return {
            "status": "evaluated",
            "decision": decision,
            "raw_decision": raw_decision,
            "score": round(probability, 6),
            "threshold": round(threshold, 6),
            "state_provenance": provenance,
        }

    def _candidate_payload(
        self,
        row: dict[str, Any],
        decision: str,
        data_drift: dict[str, Any],
        signal_day: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_features = {}
        for name in (
            "feature_relative_industry_strength_20d",
            "feature_main_net_to_amount_sum_5d",
            "feature_return_20d",
            "feature_volatility_20d",
            "feature_amount_trend_5d_20d",
        ):
            if name in row:
                evidence_features[name] = _safe_float(row.get(name))
        evidence_groups = {
            "capital_flow": {
                name: _safe_float(row.get(name))
                for name in (
                    "feature_main_net_to_amount_sum_5d",
                    "feature_main_net_to_circ_mv_sum_5d",
                    "feature_moneyflow_positive_days_5d",
                    "feature_amount_trend_5d_20d",
                )
                if name in row
            },
            "industry": {
                name: _safe_float(row.get(name))
                for name in (
                    "feature_relative_industry_strength_20d",
                    "feature_industry_rank_pct_20d",
                    "feature_cross_industry_strength_flow_5d",
                )
                if name in row
            },
            "events": {
                name: _safe_float(row.get(name))
                for name in (
                    "feature_v9_top_list_net_amount_rank",
                    "feature_v9_top_inst_net_buy_rank",
                    "feature_v9_repurchase_amount_rank",
                    "feature_v9_holdertrade_net_change_rank",
                    "feature_v9_share_float_ratio_rank",
                )
                if name in row
            },
            "financials": {
                name: _safe_float(row.get(name))
                for name in (
                    "feature_quality_composite",
                    "feature_growth_composite",
                    "feature_q_roe",
                    "feature_netprofit_yoy",
                    "feature_ocf_to_profit",
                )
                if name in row
            },
        }
        risk_tags = list(row.get("risk_tags") or [])
        for tag in self._signal_day_risk_tags(signal_day or {}):
            if tag not in risk_tags:
                risk_tags.append(tag)
        return {
            "rank": int(row["candidate_rank"]),
            "ts_code": row["ts_code"],
            "name": row.get("name"),
            "industry": row.get("industry"),
            "candidate_type": str(row.get("candidate_type") or "stable"),
            "decision": decision,
            "overall_score": _safe_float(row.get("prediction")),
            "stable_score": _safe_float(row.get("v11_stable_score")),
            "aggressive_score": _safe_float(row.get("v11_aggressive_score")),
            "p_hit_5pct_10d": _safe_float(row.get("estimate_hit5_10d")),
            "expected_net_return_5d": _safe_float(row.get("estimate_return_5d")),
            "expected_mfe_10d": _safe_float(row.get("estimate_upside_10d")),
            "expected_mae_10d": _safe_float(row.get("estimate_safety_10d")),
            "model_disagreement": _safe_float(row.get("model_disagreement")),
            "evidence": {
                "features": evidence_features,
                "groups": evidence_groups,
                "group_policy": "point-in-time normalized feature evidence; empty groups mean the frozen model output did not expose that evidence",
                "event_policy": "structure-event factors are not announcement narratives; announcement authenticity still requires manual verification",
                "horizon_policy": "this artifact predicts 5d net return and 10d hit/upside/downside; 20d is shown only after audited settlement",
                "short_execution_prediction": {
                    "available": False,
                    "reason": "current v11 artifact has no independently trained and isolated D+2 execution head",
                    "promotion_allowed": False,
                },
                "short_execution_policy": {
                    "entry_time_target": "09:35",
                    "entry_price_source": "daily_open_proxy",
                    "first_legal_sell_day": "D+2",
                    "take_profit_rate": 0.01,
                    "stop_loss_rate": -0.05,
                    "terminal_exit": "D+5 close, deferred when untradable",
                    "same_bar_rule": "stop_first",
                },
                "data_drift": data_drift,
                "candidate_type": str(row.get("candidate_type") or "stable"),
                "stable_score": _safe_float(row.get("v11_stable_score")),
                "aggressive_score": _safe_float(row.get("v11_aggressive_score")),
                "signal_day": signal_day or {},
                "v12_default_portfolio_policy": "stable candidates share 70pct paper weight; aggressive candidates share 30pct paper weight; no order quantity or capital amount is emitted",
                "probability_calibration": "uncalibrated_training_probability",
                "estimate_policy": "ranker outputs are linearly mapped to historical target units and remain research estimates",
            },
            "risk_tags": risk_tags,
        }

    @staticmethod
    def _signal_day_risk_tags(context: dict[str, Any]) -> list[str]:
        pct_chg = _safe_float(context.get("pct_chg"))
        at_down_limit = bool(context.get("at_down_limit"))
        down_limit_room_used = _safe_float(context.get("down_limit_room_used"))
        tags = []
        if at_down_limit:
            tags.append("信号日触及跌停")
        elif down_limit_room_used is not None and down_limit_room_used >= 0.70:
            tags.append("信号日接近跌停")
        if pct_chg is not None and pct_chg <= -7.0:
            tags.append("信号日大幅下跌")
        return tags

    @staticmethod
    def _signal_day_context(trade_date: str, ts_codes: list[str]) -> dict[str, dict[str, Any]]:
        if not ts_codes:
            return {}
        with session_scope() as session:
            quotes = {
                row.ts_code: row
                for row in session.query(DailyQuote)
                .filter(DailyQuote.trade_date == trade_date, DailyQuote.ts_code.in_(ts_codes))
                .all()
            }
            limits = {
                row.ts_code: row
                for row in session.query(StockLimitDaily)
                .filter(StockLimitDaily.trade_date == trade_date, StockLimitDaily.ts_code.in_(ts_codes))
                .all()
            }
        result = {}
        for ts_code in ts_codes:
            quote = quotes.get(ts_code)
            limit = limits.get(ts_code)
            close = _safe_float(quote.close if quote else None)
            pre_close = _safe_float(quote.pre_close if quote else None)
            down_limit = _safe_float(limit.down_limit if limit else None)
            at_down_limit = bool(
                close is not None
                and down_limit is not None
                and close <= down_limit * 1.0001
            )
            room_used = None
            if close is not None and pre_close and down_limit is not None and down_limit < pre_close:
                allowed_drop = (pre_close - down_limit) / pre_close
                actual_drop = max((pre_close - close) / pre_close, 0.0)
                room_used = min(actual_drop / allowed_drop, 1.0) if allowed_drop > 0 else None
            result[ts_code] = {
                "trade_date": trade_date,
                "close": close,
                "pre_close": pre_close,
                "pct_chg": _safe_float(quote.pct_chg if quote else None),
                "down_limit": down_limit,
                "at_down_limit": at_down_limit,
                "down_limit_room_used": _safe_float(room_used),
            }
        return result

    @staticmethod
    def _build_v12_paper_ledger(
        candidates: list[dict[str, Any]],
        *,
        paper_eligible: bool,
        market_gate: dict[str, Any],
    ) -> dict[str, Any]:
        research_top5 = [dict(row) for row in candidates]
        default_rows: list[dict[str, Any]] = []
        if paper_eligible:
            stable = [row for row in candidates if row.get("candidate_type") == "stable"]
            aggressive = [row for row in candidates if row.get("candidate_type") == "aggressive"]
            fallback = [row for row in candidates if row not in stable and row not in aggressive]
            stable_weight = 0.70 / len(stable) if stable else 0.0
            aggressive_weight = 0.30 / len(aggressive) if aggressive else 0.0
            fallback_weight = 1.0 / len(fallback) if fallback and not (stable or aggressive) else 0.0
            for row in candidates:
                candidate_type = row.get("candidate_type")
                if candidate_type == "stable":
                    weight = stable_weight
                elif candidate_type == "aggressive":
                    weight = aggressive_weight
                else:
                    weight = fallback_weight
                default_rows.append(
                    {
                        "rank": row.get("rank"),
                        "ts_code": row.get("ts_code"),
                        "name": row.get("name"),
                        "candidate_type": candidate_type or "fallback",
                        "paper_weight": _safe_float(weight),
                        "budget_status": "included_by_weight_policy" if weight > 0 else "research_only",
                        "execution_status": "paper_pending_open_check",
                    }
                )
        return {
            "contract": "ml_v12_paper_ledger_v1",
            "research_top5": research_top5,
            "default_70_30_portfolio": default_rows,
            "manual_selection": {
                "status": "awaiting_user_review",
                "policy": "Manual choices are stored separately and never rewrite the original Top5 or market labels.",
            },
            "market_eligibility": market_gate.get("decision"),
            "paper_eligible": bool(paper_eligible),
            "diagnostic_only": not paper_eligible,
            "capital_policy": "percentage weights only; no order amount, quantity, or automatic trading action",
        }

    def _feature_drift(self, frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        reference = self.artifact.get("training_distribution") or {}
        if not reference:
            return {"status": "unavailable", "reason": "artifact_has_no_training_distribution"}
        severe = []
        warnings = []
        for feature in feature_columns:
            stats = reference.get(feature) or {}
            values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            current_median = values.median()
            train_median = stats.get("median")
            q25, q75 = stats.get("q25"), stats.get("q75")
            missing_delta = float(values.isna().mean()) - float(stats.get("missing_rate") or 0.0)
            shift = None
            if all(value is not None for value in (current_median, train_median, q25, q75)):
                scale = max(abs(float(q75) - float(q25)), 1e-6)
                shift = abs(float(current_median) - float(train_median)) / scale
            if missing_delta > 0.20 or (shift is not None and shift > 3.0):
                severe.append(feature)
            elif missing_delta > 0.10 or (shift is not None and shift > 2.0):
                warnings.append(feature)
        severe_rate = len(severe) / max(len(feature_columns), 1)
        critical_severe = sorted(set(severe) & CRITICAL_DRIFT_FEATURES)
        blocked = severe_rate > 0.10 or bool(critical_severe)
        return {
            "status": "blocked" if blocked else "pass",
            "feature_count": len(feature_columns),
            "severe_feature_count": len(severe),
            "warning_feature_count": len(warnings),
            "severe_feature_rate": round(severe_rate, 6),
            "severe_features": severe[:20],
            "critical_severe_features": critical_severe,
            "warning_features": warnings[:20],
            "policy": (
                "Block paper usability when more than 10% of live features are severe, or when any critical "
                "limit/tradability feature has a >3 IQR median shift or >20 percentage-point missingness increase."
            ),
        }

    def _paper_drift_passes(self) -> bool:
        with session_scope() as session:
            rows = (
                session.query(MlV8CandidateSignal.signal_date, MlV8CandidateSignal.evidence)
                .filter(
                    MlV8CandidateSignal.model_version == self.artifact["model_version"],
                    MlV8CandidateSignal.decision == "eligible_paper",
                )
                .order_by(MlV8CandidateSignal.signal_date.desc())
                .all()
            )
        by_date: dict[str, dict[str, Any]] = {}
        for signal_date, evidence in rows:
            by_date.setdefault(signal_date, ((evidence or {}).get("data_drift") or {}))
        recent = list(by_date.values())[:40]
        return len(recent) >= 40 and all(item.get("status") == "pass" for item in recent)

    def _persist_candidates(self, signal_date: str, candidates: list[dict[str, Any]]) -> None:
        now = datetime.utcnow()
        with session_scope() as session:
            current_codes = [row["ts_code"] for row in candidates]
            stale = session.query(MlV8CandidateSignal).filter(
                MlV8CandidateSignal.signal_date == signal_date,
                MlV8CandidateSignal.model_version == self.artifact["model_version"],
            )
            if current_codes:
                stale = stale.filter(~MlV8CandidateSignal.ts_code.in_(current_codes))
            stale.delete(synchronize_session=False)
            for row in candidates:
                stmt = insert(MlV8CandidateSignal).values(
                    signal_date=signal_date,
                    ts_code=row["ts_code"],
                    name=row.get("name"),
                    industry=row.get("industry"),
                    model_version=self.artifact["model_version"],
                    rank=row["rank"],
                    decision=row["decision"],
                    overall_score=row["overall_score"],
                    p_hit_5pct_10d=row.get("p_hit_5pct_10d"),
                    expected_net_return_5d=row.get("expected_net_return_5d"),
                    expected_mfe_10d=row.get("expected_mfe_10d"),
                    expected_mae_10d=row.get("expected_mae_10d"),
                    model_disagreement=row.get("model_disagreement"),
                    evidence=row.get("evidence"),
                    risk_tags=row.get("risk_tags"),
                    created_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_ml_v8_candidate_signal",
                    set_={
                        "name": stmt.excluded.name,
                        "industry": stmt.excluded.industry,
                        "rank": stmt.excluded.rank,
                        "decision": stmt.excluded.decision,
                        "overall_score": stmt.excluded.overall_score,
                        "p_hit_5pct_10d": stmt.excluded.p_hit_5pct_10d,
                        "expected_net_return_5d": stmt.excluded.expected_net_return_5d,
                        "expected_mfe_10d": stmt.excluded.expected_mfe_10d,
                        "expected_mae_10d": stmt.excluded.expected_mae_10d,
                        "model_disagreement": stmt.excluded.model_disagreement,
                        "evidence": stmt.excluded.evidence,
                        "risk_tags": stmt.excluded.risk_tags,
                        "created_at": now,
                    },
                )
                session.execute(stmt)

    def _labels_for_signal(self, ts_code: str, signal_date: str, as_of_date: str) -> dict[str, Any]:
        with session_scope() as session:
            dates = [
                row[0]
                for row in session.query(DailyQuote.trade_date)
                .filter(DailyQuote.trade_date > signal_date, DailyQuote.trade_date <= as_of_date)
                .group_by(DailyQuote.trade_date)
                .order_by(DailyQuote.trade_date.asc())
                .limit(max(V8_HORIZONS) + 1)
                .all()
            ]
            quotes = {
                row.trade_date: row
                for row in session.query(DailyQuote)
                .filter(DailyQuote.ts_code == ts_code, DailyQuote.trade_date.in_(dates))
                .all()
            }
            limits = {
                row.trade_date: row
                for row in session.query(StockLimitDaily)
                .filter(StockLimitDaily.ts_code == ts_code, StockLimitDaily.trade_date.in_(dates))
                .all()
            }
        if not dates:
            return {}

        def day(date_value: str) -> dict[str, Any]:
            quote, limit = quotes.get(date_value), limits.get(date_value)
            return {
                "trade_date": date_value,
                "open": quote.open if quote else None,
                "close": quote.close if quote else None,
                "high": quote.high if quote else None,
                "low": quote.low if quote else None,
                "up_limit": limit.up_limit if limit else None,
                "down_limit": limit.down_limit if limit else None,
            }

        entry = day(dates[0])
        future = [day(value) for value in dates[1:]]
        labels = compute_path_labels(entry, future)
        labels["short_execution"] = compute_short_execution_label(entry, future)
        return labels

    def _persist_outcome(self, signal_id: int, horizon: int, labels: dict[str, Any]) -> None:
        prefix = f"label_{horizon}d"
        now = datetime.utcnow()
        stmt = insert(MlV8CandidateOutcome).values(
            signal_id=signal_id,
            horizon=horizon,
            settlement_date=labels.get(f"{prefix}_end_date"),
            fixed_tradable=bool(labels.get(f"{prefix}_fixed_tradable")),
            net_return=labels.get(f"{prefix}_net_return"),
            mfe_close=labels.get(f"{prefix}_mfe_close"),
            mae_close=labels.get(f"{prefix}_mae_close"),
            hit_5pct=labels.get(f"{prefix}_hit_5pct"),
            hit_8pct=labels.get(f"{prefix}_hit_8pct"),
            hit_12pct=labels.get(f"{prefix}_hit_12pct"),
            details={
                "entry_date": labels.get("label_entry_date"),
                "entry_reason": labels.get("label_entry_reason"),
                **({"short_execution": labels.get("short_execution")} if horizon == 1 else {}),
            },
            created_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_ml_v8_candidate_outcome",
            set_={
                "settlement_date": stmt.excluded.settlement_date,
                "fixed_tradable": stmt.excluded.fixed_tradable,
                "net_return": stmt.excluded.net_return,
                "mfe_close": stmt.excluded.mfe_close,
                "mae_close": stmt.excluded.mae_close,
                "hit_5pct": stmt.excluded.hit_5pct,
                "hit_8pct": stmt.excluded.hit_8pct,
                "hit_12pct": stmt.excluded.hit_12pct,
                "details": stmt.excluded.details,
                "created_at": now,
            },
        )
        with session_scope() as session:
            session.execute(stmt)

    @staticmethod
    def _signal_dict(row: MlV8CandidateSignal) -> dict[str, Any]:
        return {
            "id": row.id,
            "signal_date": row.signal_date,
            "ts_code": row.ts_code,
            "name": row.name,
            "industry": row.industry,
            "model_version": row.model_version,
            "rank": row.rank,
            "decision": row.decision,
            "candidate_type": (row.evidence or {}).get("candidate_type"),
            "stable_score": (row.evidence or {}).get("stable_score"),
            "aggressive_score": (row.evidence or {}).get("aggressive_score"),
            "overall_score": row.overall_score,
            "p_hit_5pct_10d": row.p_hit_5pct_10d,
            "expected_net_return_5d": row.expected_net_return_5d,
            "expected_mfe_10d": row.expected_mfe_10d,
            "expected_mae_10d": row.expected_mae_10d,
            "model_disagreement": row.model_disagreement,
            "evidence": row.evidence or {},
            "risk_tags": row.risk_tags or [],
        }


def _apply_calibration(values: np.ndarray, calibration: dict[str, float] | None) -> np.ndarray:
    if not calibration:
        return np.asarray(values, dtype=float)
    return np.asarray(values, dtype=float) * float(calibration["slope"]) + float(calibration["intercept"])


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 6) if np.isfinite(result) else None
