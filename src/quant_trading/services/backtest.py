from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.tushare_client import TushareClient
from ..db import BacktestRun, session_scope


@dataclass
class BacktestService:
    client: TushareClient

    @classmethod
    def create(cls) -> "BacktestService":
        return cls(TushareClient())

    def run_buy_and_hold_backtest(self, ts_code: str, start_date: str, end_date: str, initial_capital: float = 100000.0, fee_rate: float = 0.0003, slippage: float = 0.001) -> dict:
        df = self.client.get_daily_quotes(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError("No real daily data returned from Tushare for backtest.")
        df = df.sort_values("trade_date").reset_index(drop=True)
        buy_price = float(df.iloc[0]["close"]) * (1 + slippage)
        shares = int((initial_capital * (1 - fee_rate)) / buy_price / 100) * 100
        if shares <= 0:
            raise ValueError("Initial capital is insufficient to buy one board lot.")
        cash = initial_capital - shares * buy_price * (1 + fee_rate)
        equity = df["close"].astype(float) * shares + cash
        final_value = float(equity.iloc[-1])
        total_return = final_value / initial_capital - 1
        drawdown = equity / equity.cummax() - 1
        max_drawdown = float(drawdown.min())
        daily_ret = equity.pct_change().dropna()
        sharpe = float((daily_ret.mean() / daily_ret.std()) * (252**0.5)) if not daily_ret.empty and daily_ret.std() else None

        result = {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 6),
            "max_drawdown": round(max_drawdown, 6),
            "sharpe_ratio": None if sharpe is None else round(sharpe, 6),
            "shares": shares,
            "cash": round(cash, 2),
            "equity_curve": [
                {"trade_date": str(trade_date), "value": round(float(value), 2)}
                for trade_date, value in zip(df["trade_date"].astype(str), equity)
            ],
        }
        self._persist(result)
        return result

    def _persist(self, result: dict) -> None:
        with session_scope() as session:
            session.add(
                BacktestRun(
                    strategy_type="buy_and_hold_baseline",
                    start_date=result["start_date"],
                    end_date=result["end_date"],
                    initial_capital=result["initial_capital"],
                    final_value=result["final_value"],
                    total_return=result["total_return"],
                    max_drawdown=result["max_drawdown"],
                    sharpe_ratio=result["sharpe_ratio"],
                    metadata_json=result,
                )
            )

    def run(self, *args, **kwargs) -> dict:
        """Compatibility wrapper for older callers."""
        return self.run_buy_and_hold_backtest(*args, **kwargs)
