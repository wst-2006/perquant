from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ..data.tushare_client import TushareClient
from ..db import Position, TradeLog, session_scope
from ..utils.dates import today_yyyymmdd


@dataclass
class PortfolioService:
    client: TushareClient | None = None

    @classmethod
    def create(cls) -> "PortfolioService":
        return cls(TushareClient())

    def add_position(
        self,
        ts_code: str,
        name: str,
        strategy_type: str,
        buy_price: float,
        quantity: int,
        buy_date: date | None = None,
        notes: str | None = None,
    ) -> int:
        if quantity <= 0 or quantity % 100 != 0:
            raise ValueError("A-share quantity must be a positive multiple of 100.")
        if buy_price <= 0:
            raise ValueError("buy_price must be positive.")
        with session_scope() as session:
            position = Position(
                ts_code=ts_code,
                name=name,
                strategy_type=strategy_type,
                buy_date=buy_date or date.today(),
                buy_price=buy_price,
                quantity=quantity,
                current_price=buy_price,
                highest_price=buy_price,
                lowest_price=buy_price,
                notes=notes,
            )
            session.add(position)
            session.flush()
            return int(position.id)

    def import_positions_csv(self, csv_path: str | Path, clear_existing: bool = False) -> dict:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        imported = 0
        failed: list[dict] = []
        with session_scope() as session:
            if clear_existing:
                session.query(Position).filter(Position.status == "holding").delete()
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required = {"ts_code", "name", "cost", "quantity"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"CSV missing required fields: {sorted(missing)}")
                for line_no, row in enumerate(reader, start=2):
                    try:
                        ts_code = row["ts_code"].strip()
                        quantity = int(float(row["quantity"]))
                        if quantity <= 0 or quantity % 100 != 0:
                            raise ValueError("quantity must be a positive multiple of 100")
                        buy_date = self._parse_buy_date(row.get("buy_date"))
                        price = float(row["cost"])
                        if price <= 0:
                            raise ValueError("cost must be positive")
                        session.add(
                            Position(
                                ts_code=ts_code,
                                name=row["name"].strip(),
                                strategy_type=(row.get("strategy") or "manual").strip() or "manual",
                                buy_date=buy_date,
                                buy_price=price,
                                quantity=quantity,
                                current_price=price,
                                highest_price=price,
                                lowest_price=price,
                                notes=row.get("notes"),
                            )
                        )
                        imported += 1
                    except Exception as exc:
                        failed.append({"line": line_no, "row": row, "error": str(exc)})
        return {"imported": imported, "failed": failed}

    def evaluate_positions(self) -> list[dict]:
        with session_scope() as session:
            positions = session.query(Position).filter(Position.status == "holding").order_by(Position.buy_date.asc()).all()
            return [self._position_payload(pos) for pos in positions]

    def portfolio_summary(self) -> dict:
        positions = self.evaluate_positions()
        total_cost = sum(item["cost_value"] for item in positions)
        total_market_value = sum(item["market_value"] for item in positions)
        total_profit = total_market_value - total_cost
        risk_counts: dict[str, int] = {}
        for item in positions:
            for tag in item["risk_tags"]:
                risk_counts[tag] = risk_counts.get(tag, 0) + 1
        return {
            "mode": "observation_only",
            "message": "资金分配由用户人工决定；系统只提供持仓风险和操作倾向。",
            "holding_count": len(positions),
            "total_cost": round(total_cost, 4),
            "total_market_value": round(total_market_value, 4),
            "total_profit": round(total_profit, 4),
            "total_profit_pct": round(total_profit / total_cost * 100, 4) if total_cost else 0.0,
            "risk_counts": risk_counts,
            "positions": positions,
        }

    def update_prices(self) -> int:
        if self.client is None:
            self.client = TushareClient()
        updated = 0
        with session_scope() as session:
            positions = session.query(Position).filter(Position.status == "holding").all()
            for pos in positions:
                daily = self.client.get_daily_quotes(ts_code=pos.ts_code, end_date=today_yyyymmdd())
                if daily.empty or "close" not in daily.columns:
                    continue
                latest = daily.sort_values("trade_date", ascending=False).iloc[0]
                close = float(latest["close"])
                pos.current_price = close
                pos.highest_price = max(pos.highest_price or close, close)
                pos.lowest_price = min(pos.lowest_price or close, close)
                pos.updated_at = datetime.utcnow()
                updated += 1
        return updated

    def trade_logs(self, limit: int = 100) -> list[dict]:
        with session_scope() as session:
            logs = session.query(TradeLog).order_by(TradeLog.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": item.id,
                    "ts_code": item.ts_code,
                    "name": item.name,
                    "action": item.action,
                    "trade_date": item.trade_date.isoformat(),
                    "price": item.price,
                    "quantity": item.quantity,
                    "reason": item.reason,
                    "created_at": item.created_at.isoformat(),
                }
                for item in logs
            ]

    def exit_position(self, position_id: int, exit_price: float, reason: str) -> None:
        with session_scope() as session:
            pos = session.get(Position, position_id)
            if not pos:
                raise ValueError(f"Position not found: {position_id}")
            pos.status = "exited"
            pos.current_price = exit_price
            session.add(
                TradeLog(
                    ts_code=pos.ts_code,
                    name=pos.name,
                    action="sell",
                    trade_date=date.today(),
                    price=exit_price,
                    quantity=pos.quantity,
                    reason=reason,
                )
            )

    @staticmethod
    def _parse_buy_date(value: str | None) -> date:
        if not value:
            return date.today()
        value = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        raise ValueError("buy_date must be YYYY-MM-DD or YYYYMMDD")

    @staticmethod
    def _target_position_pct(action: str, profit_pct: float) -> float:
        """Legacy compatibility only. New position guidance is observation-only."""
        if action == "sell":
            return 0.0
        if action == "reduce":
            return 0.5
        return 1.0

    @staticmethod
    def _position_payload(pos: Position) -> dict:
        current_price = pos.current_price if pos.current_price is not None else pos.buy_price
        cost_value = pos.buy_price * pos.quantity
        market_value = current_price * pos.quantity
        profit = market_value - cost_value
        profit_pct = profit / cost_value * 100 if cost_value else 0.0
        hold_days = max(0, (date.today() - pos.buy_date).days)
        risk_tags = PortfolioService._risk_tags(profit_pct, hold_days, pos.highest_price, current_price)
        return {
            "id": pos.id,
            "ts_code": pos.ts_code,
            "name": pos.name,
            "strategy_type": pos.strategy_type,
            "buy_date": pos.buy_date.isoformat(),
            "buy_price": pos.buy_price,
            "current_price": current_price,
            "quantity": pos.quantity,
            "cost_value": round(cost_value, 4),
            "market_value": round(market_value, 4),
            "profit": round(profit, 4),
            "profit_pct": round(profit_pct, 4),
            "hold_days": hold_days,
            "highest_price": pos.highest_price,
            "lowest_price": pos.lowest_price,
            "risk_tags": risk_tags,
            "operation_tendency": PortfolioService._operation_tendency(risk_tags, profit_pct, hold_days),
            "notes": pos.notes,
        }

    @staticmethod
    def _risk_tags(profit_pct: float, hold_days: int, highest_price: float | None, current_price: float) -> list[str]:
        tags: list[str] = []
        if profit_pct <= -8:
            tags.append("强风险")
        elif profit_pct <= -5:
            tags.append("亏损扩大")
        if hold_days >= 20:
            tags.append("持仓时间偏长")
        if highest_price and highest_price > 0:
            drawdown = (current_price - highest_price) / highest_price * 100
            if drawdown <= -8:
                tags.append("高点回撤较大")
        return tags

    @staticmethod
    def _operation_tendency(risk_tags: list[str], profit_pct: float, hold_days: int) -> str:
        if "强风险" in risk_tags:
            return "强风险"
        if "亏损扩大" in risk_tags or "高点回撤较大" in risk_tags:
            return "退出关注"
        if hold_days >= 20 and profit_pct < 2:
            return "减弱"
        if profit_pct >= 10:
            return "持有并跟踪止盈"
        return "观察"
