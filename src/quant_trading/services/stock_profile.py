from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import or_

from ..db import DailyBasic, DailyQuote, FactorSnapshot, MlV8CandidateSignal, Stock, session_scope
from ..ml_lab.service import MlLabConfig, MlLabService


RADAR_DIMENSIONS = (
    ("momentum", "趋势动量", ("return_5d", "return_20d")),
    ("capital_flow", "资金强度", ("main_net_to_amount", "main_net_to_circ_mv", "moneyflow_positive_days_5d")),
    ("industry", "行业强度", ("relative_industry_strength_20d", "industry_rank_pct_20d")),
    ("liquidity", "流动性", ("amount", "turnover_rate", "amihud_illiquidity_20d")),
    ("quality", "质量成长", ("quality_composite", "growth_composite", "value_composite")),
    ("risk", "风险韧性", ("volatility_20d", "amplitude_20d", "drawdown_20d")),
)


@dataclass(frozen=True)
class StockProfileService:
    factor_set: str = "factor_lab_v2_financial_v2"
    model_version: str | None = None

    def get_profile(self, stock_key: str) -> dict[str, Any] | None:
        key = str(stock_key or "").strip().upper()
        if not key:
            return None
        with session_scope() as session:
            stock = (
                session.query(Stock)
                .filter(or_(Stock.ts_code == key, Stock.symbol == key))
                .first()
            )
            if stock is None:
                return None
            stock_data = {
                "ts_code": stock.ts_code,
                "symbol": stock.symbol,
                "name": stock.name,
                "industry": stock.industry,
                "market": stock.market,
                "area": stock.area,
                "list_date": stock.list_date,
                "list_status": stock.list_status,
            }
            snapshot = (
                session.query(FactorSnapshot)
                .filter(FactorSnapshot.ts_code == stock.ts_code, FactorSnapshot.factor_set == self.factor_set)
                .order_by(FactorSnapshot.trade_date.desc())
                .first()
            )
            as_of_date = snapshot.trade_date if snapshot else None
            quote = (
                session.query(DailyQuote)
                .filter(DailyQuote.ts_code == stock.ts_code)
                .filter(DailyQuote.trade_date <= as_of_date if as_of_date else True)
                .order_by(DailyQuote.trade_date.desc())
                .first()
            )
            basic = (
                session.query(DailyBasic)
                .filter(DailyBasic.ts_code == stock.ts_code)
                .filter(DailyBasic.trade_date <= as_of_date if as_of_date else True)
                .order_by(DailyBasic.trade_date.desc())
                .first()
            )
            candidate = None
            if self.model_version:
                candidate = (
                    session.query(MlV8CandidateSignal)
                    .filter(
                        MlV8CandidateSignal.ts_code == stock.ts_code,
                        MlV8CandidateSignal.model_version == self.model_version,
                    )
                    .order_by(MlV8CandidateSignal.signal_date.desc())
                    .first()
                )
            snapshot_data = None if snapshot is None else {
                "trade_date": snapshot.trade_date,
                "market_regime": snapshot.market_regime,
                "market_cap_bucket": snapshot.market_cap_bucket,
                "raw_factors": dict(snapshot.raw_factors or {}),
                "normalized_factors": dict(snapshot.normalized_factors or {}),
                "data_quality": dict(snapshot.data_quality or {}),
            }
            quote_data = None if quote is None else {
                "trade_date": quote.trade_date,
                "close": _number(quote.close),
                "pct_chg": _number(quote.pct_chg),
                "amount_yuan": _scaled(quote.amount, 1000.0),
            }
            basic_data = None if basic is None else {
                "trade_date": basic.trade_date,
                "turnover_rate": _number(basic.turnover_rate),
                "volume_ratio": _number(basic.volume_ratio),
                "pe_ttm": _number(basic.pe_ttm),
                "pb": _number(basic.pb),
                "dv_ttm": _number(basic.dv_ttm),
                "total_mv_yuan": _scaled(basic.total_mv, 10_000.0),
            }
            candidate_data = None if candidate is None else {
                "signal_date": candidate.signal_date,
                "rank": candidate.rank,
                "decision": candidate.decision,
                "overall_score": _number(candidate.overall_score),
                "p_hit_5pct_10d": _number(candidate.p_hit_5pct_10d),
                "expected_net_return_5d": _number(candidate.expected_net_return_5d),
                "expected_mfe_10d": _number(candidate.expected_mfe_10d),
                "expected_mae_10d": _number(candidate.expected_mae_10d),
                "risk_tags": list(candidate.risk_tags or []),
            }
        if snapshot_data is None:
            return {
                "stock": stock_data,
                "as_of_date": None,
                "available": False,
                "reason": f"no_{self.factor_set}_snapshot",
                "quote": quote_data,
                "valuation": basic_data,
                "v8_candidate": candidate_data,
                "diagnostic_only": True,
            }
        normalized = {name: _number(value) for name, value in snapshot_data["normalized_factors"].items()}
        raw = {name: _number(value) for name, value in snapshot_data["raw_factors"].items()}
        ml_service = MlLabService(MlLabConfig(factor_set=self.factor_set, feature_set=self.factor_set))
        scored = ml_service._rank_scores(
            pd.DataFrame(
                [
                    {
                        "trade_date": snapshot_data["trade_date"],
                        "ts_code": stock_data["ts_code"],
                        "name": stock_data["name"],
                        "industry": stock_data["industry"],
                        "features": normalized,
                        "raw_features": raw,
                        "data_quality": snapshot_data["data_quality"],
                    }
                ]
            )
        ).iloc[0]
        radar = self.build_radar(normalized)
        interpretation = self.interpret(
            radar,
            stable_score=_number(scored["stable_score"]),
            aggressive_score=_number(scored["aggressive_score"]),
        )
        missing_factors = list(snapshot_data["data_quality"].get("missing_factors") or [])
        constraints = dict(snapshot_data["data_quality"].get("constraints") or {})
        return {
            "stock": stock_data,
            "as_of_date": snapshot_data["trade_date"],
            "available": True,
            "factor_set": self.factor_set,
            "market_regime": snapshot_data["market_regime"],
            "market_cap_bucket": snapshot_data["market_cap_bucket"],
            "quote": quote_data,
            "valuation": basic_data,
            "scores": {
                "stable": round(float(scored["stable_score"]), 2),
                "aggressive": round(float(scored["aggressive_score"]), 2),
                "radar": radar,
                **interpretation,
            },
            "evidence": {
                "risk_tags": ml_service._risk_tags(raw, snapshot_data["data_quality"]),
                "missing_factor_count": len(missing_factors),
                "missing_factors": missing_factors,
                "moneyflow_missing": bool(constraints.get("moneyflow_missing", False)),
                "top_dimensions": _dimension_extremes(radar, reverse=True),
                "watch_dimensions": _dimension_extremes(radar, reverse=False),
            },
            "v8_candidate": candidate_data,
            "model_version": self.model_version,
            "diagnostic_only": True,
            "policy": (
                "Radar and stable/aggressive scores are same-day factor-lab descriptions. "
                "Only v8_candidate means the stock entered the frozen Top5; neither output is an order or return promise."
            ),
        }

    @staticmethod
    def build_radar(normalized: dict[str, float | None]) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "label": label,
                "value": _mean_score(normalized, factors),
                "components": [factor for factor in factors if normalized.get(factor) is not None],
            }
            for key, label, factors in RADAR_DIMENSIONS
        ]

    @staticmethod
    def interpret(
        radar: list[dict[str, Any]],
        *,
        stable_score: float | None,
        aggressive_score: float | None,
    ) -> dict[str, Any]:
        values = {item["key"]: item.get("value") for item in radar}
        strong_score = _mean_available(
            [values.get("momentum"), values.get("capital_flow"), values.get("industry")]
        )
        fit_score = _mean_available(
            [stable_score, aggressive_score, strong_score, values.get("risk"), values.get("quality")]
        )
        return {
            "strong_score": strong_score,
            "strong_state": _band(strong_score, ((70.0, "强势"), (55.0, "偏强"), (45.0, "震荡")), "偏弱"),
            "research_fit_score": fit_score,
            "research_fit_state": _band(
                fit_score,
                ((75.0, "研究匹配度较高"), (60.0, "可继续观察"), (45.0, "匹配度一般")),
                "当前特征不突出",
            ),
            "interpretation_policy": "Transparent factor summary for manual research; not a v11 promotion gate.",
        }


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(result):
        return None
    return result


def _scaled(value: Any, scale: float) -> float | None:
    number = _number(value)
    return None if number is None else number * scale


def _mean_score(values: dict[str, float | None], names: tuple[str, ...]) -> float | None:
    return _mean_available([values.get(name) for name in names])


def _mean_available(values: list[float | None]) -> float | None:
    clean = [max(0.0, min(100.0, float(value))) for value in values if value is not None]
    return round(sum(clean) / len(clean), 2) if clean else None


def _band(value: float | None, thresholds: tuple[tuple[float, str], ...], fallback: str) -> str:
    if value is None:
        return "数据不足"
    for threshold, label in thresholds:
        if value >= threshold:
            return label
    return fallback


def _dimension_extremes(radar: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    available = [item for item in radar if item.get("value") is not None]
    ordered = sorted(available, key=lambda item: float(item["value"]), reverse=reverse)
    return [{"key": item["key"], "label": item["label"], "value": item["value"]} for item in ordered[:2]]
