from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.tushare_client import TushareClient
from ..utils.dates import days_ago_yyyymmdd, today_yyyymmdd
from .indicators import ensure_ascending, normalize_score


@dataclass
class SectorAnalyzer:
    client: TushareClient

    def analyze(self, end_date: str | None = None, top_n: int = 30) -> dict:
        end = end_date or today_yyyymmdd()
        start = days_ago_yyyymmdd(90)
        index_df = self.client.get_index_basic(market="SW")
        if index_df.empty:
            return {"trade_date": end, "sectors": [], "hot_sectors": [], "weak_sectors": []}

        sectors: list[dict] = []
        for row in index_df.head(top_n).to_dict("records"):
            code = row.get("ts_code") or row.get("index_code")
            if not code:
                continue
            daily = ensure_ascending(self.client.get_index_daily(code, start, end))
            if len(daily) < 20:
                continue
            score_data = self._score_sector(code, row.get("name"), daily)
            sectors.append(score_data)

        sectors.sort(key=lambda item: item["score"], reverse=True)
        return {
            "trade_date": end,
            "sectors": sectors,
            "hot_sectors": sectors[:10],
            "weak_sectors": sorted(sectors, key=lambda item: item["score"])[:5],
        }

    @staticmethod
    def _score_sector(code: str, name: str | None, daily: pd.DataFrame) -> dict:
        close = daily["close"].astype(float)
        vol = daily["vol"].astype(float) if "vol" in daily.columns else pd.Series(dtype=float)
        ret_5d = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 and close.iloc[-6] else 0
        ret_20d = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 and close.iloc[-21] else 0
        vol_ratio = vol.tail(5).mean() / vol.tail(20).mean() if len(vol) >= 20 and vol.tail(20).mean() else 1
        volatility = close.pct_change().tail(20).std()

        score = (
            normalize_score(float(ret_5d), -0.08, 0.12) * 0.35
            + normalize_score(float(ret_20d), -0.15, 0.25) * 0.25
            + normalize_score(float(vol_ratio), 0.6, 1.8) * 0.25
            + normalize_score(float(volatility), 0.005, 0.04, reverse=True) * 0.15
        )
        return {
            "code": code,
            "name": name or code,
            "score": round(float(score), 4),
            "return_5d": round(float(ret_5d * 100), 4),
            "return_20d": round(float(ret_20d * 100), 4),
            "volume_ratio": round(float(vol_ratio), 4),
            "volatility": round(float(volatility), 6) if pd.notna(volatility) else None,
        }

    def stock_sector_strength(self, industry: str | None, sector_result: dict) -> float:
        if not industry or not sector_result.get("sectors"):
            return 50.0
        for sector in sector_result["sectors"]:
            name = str(sector.get("name", ""))
            if industry in name or name in industry:
                return float(sector["score"])
        return 50.0
