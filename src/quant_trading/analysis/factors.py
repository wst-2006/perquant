from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import settings
from ..data.tushare_client import TushareClient
from ..utils.dates import days_ago_yyyymmdd, today_yyyymmdd
from .capital_flow import CapitalFlowAnalyzer
from .indicators import normalize_score, technical_score
from .sector import SectorAnalyzer


@dataclass
class FactorEngine:
    client: TushareClient
    sector_analyzer: SectorAnalyzer
    capital_analyzer: CapitalFlowAnalyzer

    def score_stock(self, stock: dict, market_result: dict, sector_result: dict, end_date: str | None = None, strategy_type: str = "short_term_core") -> dict:
        end = end_date or today_yyyymmdd()
        daily = self.client.get_daily_quotes(ts_code=stock["ts_code"], start_date=days_ago_yyyymmdd(180), end_date=end)
        tech = technical_score(daily)

        basic = self._latest_daily_basic(stock["ts_code"], end)
        fina = self._latest_financial(stock["ts_code"], end)
        capital = self.capital_analyzer.score_stock(stock["ts_code"], end_date=end, days=5)

        sector_strength = self.sector_analyzer.stock_sector_strength(stock.get("industry"), sector_result)
        market_fit = self._market_fit_score(market_result, tech.get("score", 0.0))
        valuation = self._valuation_score(basic)
        growth = self._growth_score(fina)
        quality = self._quality_score(fina)

        raw_scores = {
            "technical_score": tech["score"] if tech.get("valid") else None,
            "capital_score": capital["score"] if capital.get("valid") else None,
            "sector_strength": sector_strength,
            "market_fit": market_fit,
            "valuation_score": valuation,
            "growth_score": growth,
            "quality_score": quality,
        }
        total, used_weights = self._weighted_score(raw_scores)
        reasons = self._selection_reasons(raw_scores, market_result, sector_strength)
        risk_tags = self._risk_tags(raw_scores, tech, capital, market_result)
        trade_profile = self._trade_profile(strategy_type, market_result.get("signal"))
        return {
            "ts_code": stock["ts_code"],
            "name": stock.get("name"),
            "industry": stock.get("industry"),
            "strategy_type": strategy_type,
            "total_score": round(float(total), 4),
            "selection_reasons": reasons,
            "risk_tags": risk_tags,
            "trade_profile": trade_profile,
            "action": "watch_only" if market_result.get("signal") == "red" else "watch",
            "scores": raw_scores,
            "used_weights": used_weights,
            "details": {
                "technical": tech,
                "capital": capital,
                "daily_basic": basic,
                "financial": fina,
                "market_signal": market_result.get("signal"),
            },
        }

    @staticmethod
    def _selection_reasons(scores: dict[str, float | None], market_result: dict, sector_strength: float) -> list[str]:
        reasons: list[str] = []
        if (scores.get("technical_score") or 0) >= 70:
            reasons.append("技术强度较高")
        if (scores.get("capital_score") or 0) >= 70:
            reasons.append("资金流评分较强")
        if sector_strength >= 70:
            reasons.append("所属板块强度靠前")
        if market_result.get("signal") == "green":
            reasons.append("市场环境支持积极筛选")
        elif market_result.get("signal") == "red":
            reasons.append("市场红灯，仅建议观察")
        if (scores.get("growth_score") or 0) >= 65:
            reasons.append("成长因子有支撑")
        if (scores.get("quality_score") or 0) >= 65:
            reasons.append("质量因子有支撑")
        return reasons or ["综合评分达到候选阈值"]

    @staticmethod
    def _risk_tags(scores: dict[str, float | None], technical: dict, capital: dict, market_result: dict) -> list[str]:
        tags: list[str] = []
        if not technical.get("valid"):
            tags.append("技术数据不足")
        if not capital.get("valid"):
            tags.append("资金流缺失")
        if market_result.get("signal") == "red":
            tags.append("市场红灯")
        if (scores.get("sector_strength") or 0) < 40:
            tags.append("板块偏弱")
        if (scores.get("valuation_score") or 0) < 35:
            tags.append("估值压力")
        if (scores.get("quality_score") or 0) < 35:
            tags.append("质量因子偏弱")
        return tags

    @staticmethod
    def _trade_profile(strategy_type: str, market_signal: str | None) -> dict:
        if strategy_type == "aggressive":
            profile = {
                "style": "进攻候选",
                "holding_period": "1-5 个交易日观察窗口",
                "review_focus": ["情绪强度", "板块持续性", "次日承接"],
            }
        else:
            profile = {
                "style": "短线核心候选",
                "holding_period": "3-15 个交易日观察窗口",
                "review_focus": ["趋势延续", "资金连续性", "基本面风险"],
            }
        profile["market_gate"] = "watch_only" if market_signal == "red" else "reviewable"
        return profile

    def _weighted_score(self, scores: dict[str, float | None]) -> tuple[float, dict[str, float]]:
        configured = settings.strategy.factor_weights
        valid = {k: v for k, v in scores.items() if v is not None and not pd.isna(v)}
        if not valid:
            return 0.0, {}
        weight_sum = sum(configured[k] for k in valid if k in configured)
        used = {k: configured[k] / weight_sum for k in valid if k in configured}
        total = sum(float(valid[k]) * used[k] for k in used)
        return total, used

    def _latest_daily_basic(self, ts_code: str, end_date: str) -> dict:
        df = self.client.get_daily_basic(ts_code=ts_code, end_date=end_date, start_date=days_ago_yyyymmdd(20))
        if df.empty:
            return {}
        df = df.sort_values("trade_date", ascending=False)
        return df.iloc[0].to_dict()

    def _latest_financial(self, ts_code: str, end_date: str) -> dict:
        df = self.client.get_fina_indicator(ts_code=ts_code, end_date=end_date, start_date=days_ago_yyyymmdd(550))
        if df.empty:
            return {}
        date_col = "end_date" if "end_date" in df.columns else df.columns[0]
        df = df.sort_values(date_col, ascending=False)
        return df.iloc[0].to_dict()

    @staticmethod
    def _market_fit_score(market_result: dict, technical: float) -> float:
        signal = market_result.get("signal", "yellow")
        if signal == "green":
            return min(100.0, 60.0 + technical * 0.4)
        if signal == "red":
            return max(0.0, technical * 0.5)
        return 60.0

    @staticmethod
    def _valuation_score(basic: dict) -> float:
        pe = basic.get("pe_ttm") or basic.get("pe")
        pb = basic.get("pb")
        pe_score = normalize_score(float(pe), 10, 80, reverse=True) if pe not in (None, "") and pd.notna(pe) and float(pe) > 0 else 50.0
        pb_score = normalize_score(float(pb), 0.8, 8, reverse=True) if pb not in (None, "") and pd.notna(pb) and float(pb) > 0 else 50.0
        return pe_score * 0.7 + pb_score * 0.3

    @staticmethod
    def _growth_score(fina: dict) -> float:
        growth = fina.get("dt_netprofit_yoy") or fina.get("netprofit_yoy") or fina.get("or_yoy")
        return normalize_score(float(growth), -20, 80) if growth not in (None, "") and pd.notna(growth) else 50.0

    @staticmethod
    def _quality_score(fina: dict) -> float:
        roe = fina.get("roe_dt") or fina.get("roe")
        margin = fina.get("grossprofit_margin")
        roe_score = normalize_score(float(roe), 0, 25) if roe not in (None, "") and pd.notna(roe) else 50.0
        margin_score = normalize_score(float(margin), 5, 60) if margin not in (None, "") and pd.notna(margin) else 50.0
        return roe_score * 0.7 + margin_score * 0.3
