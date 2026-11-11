from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.tushare_client import TushareClient
from ..utils.dates import days_ago_yyyymmdd, today_yyyymmdd
from .indicators import ensure_ascending, moving_average, normalize_score


@dataclass
class MarketAnalyzer:
    client: TushareClient

    main_indices: tuple[str, ...] = ("000001.SH", "399001.SZ", "399006.SZ")

    def analyze(self, end_date: str | None = None) -> dict:
        end = end_date or today_yyyymmdd()
        start = days_ago_yyyymmdd(120)
        trend_score = self._trend_score(start, end)
        breadth_score = self._breadth_score(end)
        volume_score = self._volume_score(start, end)
        limit_score = self._limit_score(end)
        volatility_score = self._volatility_score(start, end)
        north_score = self._north_money_score(days_ago_yyyymmdd(20), end)

        total = (
            trend_score * 0.25
            + breadth_score * 0.20
            + volume_score * 0.15
            + limit_score * 0.15
            + volatility_score * 0.10
            + north_score * 0.15
        )
        signal = "green" if total >= 70 else "yellow" if total >= 40 else "red"
        return {
            "trade_date": end,
            "score": round(float(total), 4),
            "signal": signal,
            "signal_text": {"green": "绿灯", "yellow": "黄灯", "red": "红灯"}[signal],
            "trading_advice": self._trading_advice(signal),
            "details": {
                "trend_score": trend_score,
                "breadth_score": breadth_score,
                "volume_score": volume_score,
                "limit_score": limit_score,
                "volatility_score": volatility_score,
                "north_score": north_score,
            },
        }

    @staticmethod
    def _trading_advice(signal: str) -> dict:
        if signal == "green":
            return {"candidate_status": "active", "text": "市场环境支持积极筛选，候选股可进入人工复核。"}
        if signal == "yellow":
            return {"candidate_status": "selective", "text": "市场环境一般，候选股需提高复核标准。"}
        return {"candidate_status": "watch_only", "text": "系统性风险高，候选股降级为观察，不输出强买入语义。"}

    def _trend_score(self, start: str, end: str) -> float:
        scores: list[float] = []
        for index_code in self.main_indices:
            df = ensure_ascending(self.client.get_index_daily(index_code, start, end))
            if len(df) < 60:
                continue
            close = df["close"].astype(float)
            ma5 = moving_average(close, 5).iloc[-1]
            ma10 = moving_average(close, 10).iloc[-1]
            ma20 = moving_average(close, 20).iloc[-1]
            ma60 = moving_average(close, 60).iloc[-1]
            if ma5 > ma10 > ma20 > ma60:
                scores.append(100)
            elif ma5 > ma10 > ma20:
                scores.append(80)
            elif close.iloc[-1] > ma20:
                scores.append(60)
            else:
                scores.append(25)
        return float(pd.Series(scores).mean()) if scores else 50.0

    def _breadth_score(self, trade_date: str) -> float:
        df = self.client.get_daily_quotes(trade_date=trade_date)
        if df.empty or "pct_chg" not in df.columns:
            return 50.0
        up_ratio = (df["pct_chg"] > 0).sum() / len(df)
        return normalize_score(up_ratio, 0.3, 0.7)

    def _volume_score(self, start: str, end: str) -> float:
        df = ensure_ascending(self.client.get_index_daily("000001.SH", start, end))
        if len(df) < 20 or "vol" not in df.columns:
            return 50.0
        recent = df["vol"].tail(5).mean()
        base = df["vol"].tail(20).mean()
        return normalize_score(recent / base if base else 1.0, 0.6, 1.6)

    def _limit_score(self, trade_date: str) -> float:
        up = self.client.get_limit_list(trade_date=trade_date, limit_type="U")
        down = self.client.get_limit_list(trade_date=trade_date, limit_type="D")
        up_count = len(up)
        down_count = len(down)
        raw = 50 + min(40, up_count * 0.8) - min(40, down_count * 1.5)
        return max(0.0, min(100.0, float(raw)))

    def _volatility_score(self, start: str, end: str) -> float:
        df = ensure_ascending(self.client.get_index_daily("000001.SH", start, end))
        if len(df) < 20:
            return 50.0
        returns = df["close"].astype(float).pct_change().tail(20)
        volatility = returns.std()
        return normalize_score(float(volatility), 0.005, 0.03, reverse=True)

    def _north_money_score(self, start: str, end: str) -> float:
        df = self.client.get_moneyflow_hsgt(start_date=start, end_date=end)
        if df.empty:
            return 50.0
        amount_col = "north_money" if "north_money" in df.columns else None
        if not amount_col:
            for col in df.columns:
                if "north" in col.lower() or "net" in col.lower():
                    amount_col = col
                    break
        if not amount_col:
            return 50.0
        net = float(pd.to_numeric(df[amount_col], errors="coerce").fillna(0).sum())
        return normalize_score(net, -1000000, 1000000)
