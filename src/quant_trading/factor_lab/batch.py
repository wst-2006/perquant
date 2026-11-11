from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Any
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from ..db import DailyBasic, DailyQuote, FactorCandidatePoolItem, FactorCandidatePoolRun, FactorSnapshot, MoneyflowDaily, Stock, session_scope
from .service import DEFAULT_HORIZONS, FactorLabError, FactorLabService


VALIDATION_WINDOWS = [
    {"name": "bull_start", "label": "牛头/启动", "start_date": "20250601", "end_date": "20250731"},
    {"name": "bull_mid", "label": "牛中/跨年强势", "start_date": "20260105", "end_date": "20260227"},
    {"name": "bull_late", "label": "牛尾/高位分歧", "start_date": "20260401", "end_date": "20260520"},
    {"name": "control_choppy", "label": "对照/回撤震荡", "start_date": "20260301", "end_date": "20260415"},
]

VALIDATION_FACTORS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "turnover_rate",
    "volume_ratio",
    "main_net_to_amount",
    "main_net_to_circ_mv",
    "volatility_20d",
    "pe_ttm",
    "pb",
]


@dataclass
class FactorLabBatchRunner:
    service: FactorLabService

    @classmethod
    def create(cls) -> "FactorLabBatchRunner":
        return cls(FactorLabService())

    def data_readiness(self) -> dict:
        windows = []
        for window in VALIDATION_WINDOWS:
            trade_dates = self._trade_dates(window["start_date"], window["end_date"])
            windows.append(
                {
                    **window,
                    "trade_date_count": len(trade_dates),
                    "first_trade_date": trade_dates[0] if trade_dates else None,
                    "last_trade_date": trade_dates[-1] if trade_dates else None,
                    "quote_rows": self._count_rows(DailyQuote, window["start_date"], window["end_date"]),
                    "daily_basic_rows": self._count_rows(DailyBasic, window["start_date"], window["end_date"]),
                    "moneyflow_rows": self._count_rows(MoneyflowDaily, window["start_date"], window["end_date"]),
                    "snapshot_rows": self._count_snapshot_rows(window["start_date"], window["end_date"]),
                }
            )
        with session_scope() as session:
            stock_count = session.scalar(select(func.count()).select_from(Stock)) or 0
        return {
            "stock_count": stock_count,
            "windows": windows,
            "required_history": {"start_date": "20250201", "end_date": "20260618"},
            "moneyflow_policy": "window_only_with_prior_buffer",
        }

    def build_snapshots_for_windows(self) -> dict:
        results = []
        for trade_date in self._all_trade_dates():
            try:
                results.append(self.service.build_snapshot(trade_date))
            except FactorLabError as exc:
                results.append({"task": "factor_snapshot", "trade_date": trade_date, "error": str(exc)})
        return {"task": "build_validation_snapshots", "results": results}

    def build_forward_returns_for_windows(self, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> dict:
        results = []
        for trade_date in self._all_trade_dates():
            try:
                results.append(self.service.build_forward_returns(trade_date, horizons=horizons))
            except FactorLabError as exc:
                results.append({"task": "factor_forward_returns", "trade_date": trade_date, "error": str(exc)})
        return {"task": "build_validation_forward_returns", "horizons": list(horizons), "results": results}

    def evaluate_validation_suite(
        self,
        factors: list[str] | None = None,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        output_dir: str | Path = "reports/factor_lab",
    ) -> dict:
        factor_names = factors or VALIDATION_FACTORS
        rows: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        for factor in factor_names:
            for horizon in horizons:
                window_summaries = []
                for window in VALIDATION_WINDOWS:
                    result = self._evaluate_one_window(factor, horizon, window)
                    window_summaries.append(result)
                    details.append(result)
                combined = self._combined_verdict(factor, horizon, window_summaries)
                rows.append(combined)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "factor_validation_summary.csv"
        json_path = output_path / "factor_validation_details.json"
        self._write_summary_csv(csv_path, rows)
        json_path.write_text(json.dumps({"summary": rows, "details": details}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pool_version = self._pool_version(factor_names, horizons)
        pool_run_id = self._persist_candidate_pool(pool_version, factor_names, horizons, rows, details)
        registry_path = output_path / "candidate_pool_registry.json"
        registry_payload = {
            "factor_set": self.service.config.factor_set,
            "pool_version": pool_version,
            "pool_run_id": pool_run_id,
            "factor_count": len(factor_names),
            "horizons": list(horizons),
            "summary_path": str(csv_path),
            "details_path": str(json_path),
        }
        registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {
            "task": "evaluate_validation_suite",
            "pool_run_id": pool_run_id,
            "pool_version": pool_version,
            "summary_path": str(csv_path),
            "details_path": str(json_path),
            "registry_path": str(registry_path),
            "rows": len(rows),
            "summary": rows,
        }

    def build_extension_report(
        self,
        factors: list[str] | None = None,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        output_dir: str | Path = "reports/factor_lab",
    ) -> dict:
        factor_names = factors or VALIDATION_FACTORS
        rows = []
        details = []
        for factor in factor_names:
            for horizon in horizons:
                window_rows = []
                for window in VALIDATION_WINDOWS:
                    item = self.service._evaluate_factor_frame(
                        factor_name=factor,
                        start_date=window["start_date"],
                        end_date=window["end_date"],
                        horizon=horizon,
                    )
                    window_item = {
                        "factor": factor,
                        "horizon": horizon,
                        "window": window["name"],
                        "label": window["label"],
                        **item,
                    }
                    window_rows.append(window_item)
                    details.append(window_item)
                rows.append(self._combine_extension_factor_rows(factor, horizon, window_rows))

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "factor_extension_summary.csv"
        details_path = output_path / "factor_extension_details.json"
        self._write_summary_csv(summary_path, rows)
        details_path.write_text(json.dumps({"summary": rows, "details": details}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {
            "task": "build_extension_report",
            "rows": rows,
            "details": details,
            "row_count": len(rows),
            "detail_count": len(details),
            "summary_path": str(summary_path),
            "details_path": str(details_path),
        }

    def walk_forward_validation(
        self,
        factor_name: str,
        train_window: int = 60,
        validation_window: int = 20,
        step_window: int = 20,
        start_date: str = "20250201",
        end_date: str = "20260618",
        horizon: int = 5,
        output_dir: str | Path = "reports/factor_lab",
    ) -> dict:
        result = self.service.walk_forward_factor(
            factor_name=factor_name,
            train_window=train_window,
            validation_window=validation_window,
            step_window=step_window,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / f"walk_forward_{factor_name}_summary.json"
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {
            "task": "walk_forward_validation",
            "run_id": result["run_id"],
            "summary": result["summary"],
            "folds": result["folds"],
            "fold_count": len(result["folds"]),
            "summary_path": str(summary_path),
        }

    def _evaluate_one_window(self, factor: str, horizon: int, window: dict) -> dict:
        try:
            result = self.service.evaluate_factor(
                factor_name=factor,
                start_date=window["start_date"],
                end_date=window["end_date"],
                horizon=horizon,
            )
            summary = result["summary"]
            return {
                "factor": factor,
                "horizon": horizon,
                "window": window["name"],
                "label": window["label"],
                "verdict": summary.get("verdict"),
                "rank_ic_mean": summary.get("rank_ic_mean"),
                "rank_ic_ir": summary.get("rank_ic_ir"),
                "top50_return": (summary.get("top_returns") or {}).get("top50", {}).get("mean_return"),
                "sample_size": summary.get("sample_size"),
                "date_count": summary.get("date_count"),
                "status": "evaluated",
                "error": None,
            }
        except FactorLabError as exc:
            return {
                "factor": factor,
                "horizon": horizon,
                "window": window["name"],
                "label": window["label"],
                "verdict": "不可评估",
                "rank_ic_mean": None,
                "rank_ic_ir": None,
                "top50_return": None,
                "sample_size": 0,
                "date_count": 0,
                "status": "error",
                "error": str(exc),
            }

    @staticmethod
    def _combined_verdict(factor: str, horizon: int, window_summaries: list[dict]) -> dict:
        usable_count = sum(1 for item in window_summaries if item.get("verdict") == "可用")
        observe_count = sum(1 for item in window_summaries if item.get("verdict") == "观察")
        failed_windows = [item["window"] for item in window_summaries if item.get("verdict") in {"弃用", "不可评估"}]
        control = next((item for item in window_summaries if item["window"] == "control_choppy"), None)
        if usable_count >= 3 and control and control.get("verdict") in {"可用", "观察"}:
            overall = "score_v1候选"
        elif usable_count + observe_count >= 2:
            overall = "继续观察"
        else:
            overall = "暂不采用"
        return {
            "factor": factor,
            "horizon": horizon,
            "overall_verdict": overall,
            "usable_windows": usable_count,
            "observe_windows": observe_count,
            "failed_windows": ",".join(failed_windows),
            "score_v1_candidate": overall == "score_v1候选",
            "needs_more_tracking": overall == "继续观察",
        }

    def _all_trade_dates(self) -> list[str]:
        dates: list[str] = []
        for window in VALIDATION_WINDOWS:
            dates.extend(self._trade_dates(window["start_date"], window["end_date"]))
        return sorted(set(dates))

    @staticmethod
    def _trade_dates(start_date: str, end_date: str) -> list[str]:
        with session_scope() as session:
            rows = (
                session.query(DailyQuote.trade_date)
                .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)
                .group_by(DailyQuote.trade_date)
                .order_by(DailyQuote.trade_date.asc())
                .all()
            )
        return [row[0] for row in rows]

    @staticmethod
    def _count_rows(model, start_date: str, end_date: str) -> int:
        with session_scope() as session:
            return int(
                session.scalar(
                    select(func.count()).select_from(model).where(model.trade_date >= start_date, model.trade_date <= end_date)
                )
                or 0
            )

    def _count_snapshot_rows(self, start_date: str, end_date: str) -> int:
        with session_scope() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(FactorSnapshot)
                    .where(
                        FactorSnapshot.trade_date >= start_date,
                        FactorSnapshot.trade_date <= end_date,
                        FactorSnapshot.factor_set == self.service.config.factor_set,
                    )
                )
                or 0
            )

    @staticmethod
    def _write_summary_csv(path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "factor",
            "horizon",
            "overall_verdict",
            "usable_windows",
            "observe_windows",
            "failed_windows",
            "score_v1_candidate",
            "needs_more_tracking",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _pool_version(factor_names: list[str], horizons: tuple[int, ...]) -> str:
        factor_count = len(factor_names)
        horizon_part = "-".join(str(item) for item in horizons)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"candidate_{timestamp}_{factor_count}f_{horizon_part}h"

    def _persist_candidate_pool(self, pool_version: str, factor_names: list[str], horizons: tuple[int, ...], rows: list[dict], details: list[dict]) -> int:
        summary = {
            "factor_count": len(factor_names),
            "horizons": list(horizons),
            "summary_rows": len(rows),
            "detail_rows": len(details),
        }
        with session_scope() as session:
            run = FactorCandidatePoolRun(
                factor_set=self.service.config.factor_set,
                pool_version=pool_version,
                source="validation_suite",
                summary=summary,
            )
            session.add(run)
            session.flush()
            detail_by_key = {(item["factor"], item["horizon"], item["window"]): item for item in details}
            for item in rows:
                window_details = [detail_by_key.get((item["factor"], item["horizon"], window["name"])) for window in VALIDATION_WINDOWS]
                session.add(
                    FactorCandidatePoolItem(
                        pool_run_id=run.id,
                        factor_name=item["factor"],
                        horizon=item["horizon"],
                        verdict=item["overall_verdict"],
                        usable_windows=item["usable_windows"],
                        observe_windows=item["observe_windows"],
                        failed_windows=item["failed_windows"],
                        rank_ic_mean=None,
                        rank_ic_ir=None,
                        top50_return=None,
                        score_v1_candidate=bool(item["score_v1_candidate"]),
                        needs_more_tracking=bool(item["needs_more_tracking"]),
                        details={"summary": item, "windows": window_details},
                    )
                )
            session.flush()
            return int(run.id)

    @staticmethod
    def _combine_extension_factor_rows(factor: str, horizon: int, window_rows: list[dict]) -> dict:
        usable = sum(1 for item in window_rows if item.get("verdict") == "可用")
        observe = sum(1 for item in window_rows if item.get("verdict") == "观察")
        failed = [item["window"] for item in window_rows if item.get("verdict") in {"弃用", "不可评估"}]
        control = next((item for item in window_rows if item["window"] == "control_choppy"), None)
        if usable >= 3 and control and control.get("verdict") in {"可用", "观察"}:
            overall = "score_v1候选"
        elif usable + observe >= 2:
            overall = "继续观察"
        else:
            overall = "暂不采用"
        return {
            "factor": factor,
            "horizon": horizon,
            "overall_verdict": overall,
            "usable_windows": usable,
            "observe_windows": observe,
            "failed_windows": ",".join(failed),
            "score_v1_candidate": overall == "score_v1候选",
            "needs_more_tracking": overall == "继续观察",
        }
