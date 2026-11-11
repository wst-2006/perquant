from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv

from sqlalchemy.dialects.postgresql import insert

from ..analysis.capital_flow import CapitalFlowAnalyzer
from ..analysis.factors import FactorEngine
from ..analysis.market import MarketAnalyzer
from ..analysis.sector import SectorAnalyzer
from ..config import settings
from ..data.tushare_client import TushareClient
from ..db import FactorScore, SelectionResult, SelectionRun, Stock, session_scope
from ..utils.dates import today_yyyymmdd


class SelectionRuleError(RuntimeError):
    """Raised when selection would violate project rules."""


SUPPORTED_STRATEGY_TYPES = {"short_term_core", "aggressive"}


@dataclass
class SelectionService:
    client: TushareClient

    @classmethod
    def create(cls) -> "SelectionService":
        return cls(TushareClient())

    def run(self, strategy_type: str = "short_term_core", end_date: str | None = None) -> dict:
        """Compatibility wrapper for older scripts and UI callers."""
        return self.run_full_market_selection(strategy_type=strategy_type, end_date=end_date)

    def run_full_market_selection(self, strategy_type: str = "short_term_core", end_date: str | None = None) -> dict:
        if strategy_type not in SUPPORTED_STRATEGY_TYPES:
            raise SelectionRuleError(f"Unsupported strategy_type: {strategy_type}")

        trade_date = end_date or today_yyyymmdd()
        stock_pool = self.client.get_all_a_shares(with_filter=False)
        if stock_pool.empty:
            raise SelectionRuleError("Full-market stock pool is empty from Tushare. Selection stopped.")

        self._persist_stock_pool(stock_pool)

        market_analyzer = MarketAnalyzer(self.client)
        sector_analyzer = SectorAnalyzer(self.client)
        capital_analyzer = CapitalFlowAnalyzer(self.client)
        factor_engine = FactorEngine(self.client, sector_analyzer, capital_analyzer)

        market_result = market_analyzer.analyze(trade_date)
        sector_result = sector_analyzer.analyze(trade_date)

        scored: list[dict] = []
        for stock in stock_pool.to_dict("records"):
            score = factor_engine.score_stock(
                stock=stock,
                market_result=market_result,
                sector_result=sector_result,
                end_date=trade_date,
                strategy_type=strategy_type,
            )
            if score["total_score"] >= settings.strategy.min_score:
                scored.append(score)

        scored.sort(key=lambda item: item["total_score"], reverse=True)
        if strategy_type == "short_term_core":
            top_n = settings.strategy.stable_top_n
        elif strategy_type == "aggressive":
            top_n = settings.strategy.aggressive_top_n
        selected = scored[:top_n]
        run_id = self._persist_selection(trade_date, strategy_type, len(stock_pool), selected, market_result, sector_result)
        return {
            "run_id": run_id,
            "trade_date": trade_date,
            "strategy_type": strategy_type,
            "full_market_count": len(stock_pool),
            "scored_count": len(scored),
            "selected": selected,
            "market": market_result,
            "hot_sectors": sector_result.get("hot_sectors", []),
        }

    def _persist_stock_pool(self, stock_pool) -> None:
        now = datetime.utcnow()
        with session_scope() as session:
            for row in stock_pool.to_dict("records"):
                stmt = insert(Stock).values(
                    ts_code=row.get("ts_code"),
                    symbol=row.get("symbol"),
                    name=row.get("name"),
                    area=row.get("area"),
                    industry=row.get("industry"),
                    market=row.get("market"),
                    list_date=row.get("list_date"),
                    list_status="L",
                    is_hs=row.get("is_hs"),
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[Stock.ts_code],
                    set_={
                        "symbol": stmt.excluded.symbol,
                        "name": stmt.excluded.name,
                        "area": stmt.excluded.area,
                        "industry": stmt.excluded.industry,
                        "market": stmt.excluded.market,
                        "list_date": stmt.excluded.list_date,
                        "list_status": stmt.excluded.list_status,
                        "is_hs": stmt.excluded.is_hs,
                        "updated_at": now,
                    },
                )
                session.execute(stmt)

    def _persist_selection(self, trade_date: str, strategy_type: str, stock_count: int, selected: list[dict], market_result: dict, sector_result: dict) -> int:
        with session_scope() as session:
            run = SelectionRun(
                trade_date=trade_date,
                strategy_type=strategy_type,
                stock_count=stock_count,
                status="completed",
                metadata_json={"market": market_result, "hot_sectors": sector_result.get("hot_sectors", [])},
            )
            session.add(run)
            session.flush()
            for rank, item in enumerate(selected, start=1):
                session.add(
                    SelectionResult(
                        run_id=run.id,
                        rank=rank,
                        ts_code=item["ts_code"],
                        name=item.get("name"),
                        score=item["total_score"],
                        action=item.get("action", "watch"),
                        details=item,
                    )
                )
                session.add(
                    FactorScore(
                        ts_code=item["ts_code"],
                        trade_date=trade_date,
                        strategy_type=strategy_type,
                        total_score=item["total_score"],
                        technical_score=item["scores"].get("technical_score"),
                        capital_score=item["scores"].get("capital_score"),
                        sector_strength=item["scores"].get("sector_strength"),
                        market_fit=item["scores"].get("market_fit"),
                        valuation_score=item["scores"].get("valuation_score"),
                        growth_score=item["scores"].get("growth_score"),
                        quality_score=item["scores"].get("quality_score"),
                        details=item.get("details"),
                    )
                )
            session.flush()
            return int(run.id)

    def list_runs(self, limit: int = 20) -> list[dict]:
        with session_scope() as session:
            runs = (
                session.query(SelectionRun)
                .order_by(SelectionRun.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": run.id,
                    "trade_date": run.trade_date,
                    "strategy_type": run.strategy_type,
                    "stock_count": run.stock_count,
                    "status": run.status,
                    "created_at": run.created_at.isoformat(),
                    "metadata": run.metadata_json,
                }
                for run in runs
            ]

    def get_run_results(self, run_id: int) -> dict:
        with session_scope() as session:
            run = session.get(SelectionRun, run_id)
            if run is None:
                raise ValueError(f"Selection run not found: {run_id}")
            results = (
                session.query(SelectionResult)
                .filter(SelectionResult.run_id == run_id)
                .order_by(SelectionResult.rank.asc())
                .all()
            )
            return {
                "run": {
                    "id": run.id,
                    "trade_date": run.trade_date,
                    "strategy_type": run.strategy_type,
                    "stock_count": run.stock_count,
                    "status": run.status,
                    "created_at": run.created_at.isoformat(),
                    "metadata": run.metadata_json,
                },
                "results": [
                    {
                        "rank": row.rank,
                        "ts_code": row.ts_code,
                        "name": row.name,
                        "score": row.score,
                        "action": row.action,
                        "details": row.details,
                    }
                    for row in results
                ],
            }

    def export_run_results_csv(self, run_id: int, output_path: str | Path) -> dict:
        data = self.get_run_results(run_id)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "ts_code",
                    "name",
                    "score",
                    "action",
                    "technical_score",
                    "capital_score",
                    "sector_strength",
                    "market_fit",
                    "valuation_score",
                    "growth_score",
                    "quality_score",
                    "trade_style",
                    "market_gate",
                    "selection_reasons",
                    "risk_tags",
                ],
            )
            writer.writeheader()
            for row in data["results"]:
                scores = (row.get("details") or {}).get("scores") or {}
                details = row.get("details") or {}
                trade_profile = details.get("trade_profile") or {}
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "ts_code": row["ts_code"],
                        "name": row["name"],
                        "score": row["score"],
                        "action": row["action"],
                        "technical_score": scores.get("technical_score"),
                        "capital_score": scores.get("capital_score"),
                        "sector_strength": scores.get("sector_strength"),
                        "market_fit": scores.get("market_fit"),
                        "valuation_score": scores.get("valuation_score"),
                        "growth_score": scores.get("growth_score"),
                        "quality_score": scores.get("quality_score"),
                        "trade_style": trade_profile.get("style"),
                        "market_gate": trade_profile.get("market_gate"),
                        "selection_reasons": ";".join(details.get("selection_reasons") or []),
                        "risk_tags": ";".join(details.get("risk_tags") or []),
                    }
                )
        return {"run_id": run_id, "output_path": str(path), "rows": len(data["results"])}

