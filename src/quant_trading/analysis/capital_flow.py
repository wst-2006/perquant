from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.tushare_client import TushareClient
from ..utils.dates import days_ago_yyyymmdd, today_yyyymmdd


@dataclass
class CapitalFlowAnalyzer:
    client: TushareClient

    def score_stock(self, ts_code: str, end_date: str | None = None, days: int = 5) -> dict:
        end = end_date or today_yyyymmdd()
        start = days_ago_yyyymmdd(max(days * 3, 15))

        moneyflow = self.client.get_moneyflow(ts_code=ts_code, start_date=start, end_date=end)
        main_score = self._main_capital_score(moneyflow)
        dragon_score = self._dragon_tiger_score(ts_code, end)
        north_score = self._north_bound_score(ts_code, start, end)

        available = {
            "main_capital_score": main_score,
            "dragon_tiger_score": dragon_score,
            "north_bound_score": north_score,
        }
        valid_scores = {k: v for k, v in available.items() if v is not None}
        if not valid_scores:
            return {"score": None, "valid": False, "reason": "no_real_capital_flow_data", "details": available}

        weights = {
            "main_capital_score": 0.60,
            "north_bound_score": 0.25,
            "dragon_tiger_score": 0.15,
        }
        weight_sum = sum(weights[k] for k in valid_scores)
        score = sum(valid_scores[k] * weights[k] / weight_sum for k in valid_scores)
        return {"score": round(float(score), 4), "valid": True, "details": available}

    @staticmethod
    def _amount_to_score(amount_wan: float | None) -> float:
        if amount_wan is None or pd.isna(amount_wan):
            return 0.0
        if amount_wan >= 100000:
            return 100.0
        if amount_wan >= 50000:
            return 90.0
        if amount_wan >= 10000:
            return 85.0
        if amount_wan >= 5000:
            return 80.0
        if amount_wan >= 1000:
            return 70.0
        if amount_wan >= 0:
            return 60.0
        return 0.0

    def _main_capital_score(self, moneyflow: pd.DataFrame) -> float | None:
        if moneyflow.empty:
            return None
        df = moneyflow.copy()
        # Tushare moneyflow amount fields are in ten thousand yuan in common Pro outputs.
        buy_cols = [c for c in ["buy_lg_amount", "buy_elg_amount"] if c in df.columns]
        sell_cols = [c for c in ["sell_lg_amount", "sell_elg_amount"] if c in df.columns]
        if not buy_cols or not sell_cols:
            return None
        main_net = df[buy_cols].sum(axis=1) - df[sell_cols].sum(axis=1)
        recent_total = float(main_net.head(5).sum() if "trade_date" in df.columns else main_net.tail(5).sum())
        return self._amount_to_score(recent_total)

    def _dragon_tiger_score(self, ts_code: str, trade_date: str) -> float | None:
        top = self.client.get_top_list(trade_date=trade_date)
        if top.empty or "ts_code" not in top.columns:
            return None
        rows = top[top["ts_code"] == ts_code]
        if rows.empty:
            return 0.0
        score = min(40.0, len(rows) * 10.0)
        if "net_amount" in rows.columns:
            net = float(rows["net_amount"].fillna(0).sum())
            if net >= 50000:
                score += 40
            elif net >= 20000:
                score += 30
            elif net >= 10000:
                score += 20
            elif net >= 0:
                score += 10
        else:
            score += 10
        return min(100.0, score)

    def _north_bound_score(self, ts_code: str, start_date: str, end_date: str) -> float | None:
        top10 = self.client.get_hsgt_top10(start_date=start_date, end_date=end_date)
        if top10.empty or "ts_code" not in top10.columns:
            return None
        rows = top10[top10["ts_code"] == ts_code]
        if rows.empty:
            return 0.0
        amount_col = "net_amount" if "net_amount" in rows.columns else None
        if not amount_col:
            return 60.0
        net = float(rows[amount_col].fillna(0).sum())
        if net >= 50000:
            return 100.0
        if net >= 10000:
            return 90.0
        if net >= 5000:
            return 80.0
        if net >= 1000:
            return 70.0
        if net >= 0:
            return 60.0
        return 0.0
