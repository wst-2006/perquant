"""Audited one-command daily observation pipeline for v8."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..factor_lab.service import FactorLabConfig, FactorLabService
from ..services.data_sync import DataSyncService
from .v8_service import V8InferenceService


V8_DAILY_REPORT_CONTRACT = "ml_v8_daily_observation_v1"
V8_DAILY_DATASETS = ("daily", "daily_basic", "moneyflow", "adj_factor", "stk_limit", "index_daily")


class V8DailyPipelineError(RuntimeError):
    """Raised after a blocked daily run has been recorded."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


class V8DailyPipeline:
    def __init__(
        self,
        *,
        artifact_path: str | Path = "reports/ml_lab/v8_artifact/artifact.json",
        report_dir: str | Path = "reports/ml_lab/v8_daily",
        foundation_start_date: str = "20240401",
        sync_service: DataSyncService | None = None,
        factor_service: FactorLabService | None = None,
        inference_service: V8InferenceService | None = None,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.report_dir = Path(report_dir)
        self.foundation_start_date = foundation_start_date
        self._sync_service = sync_service
        self._factor_service = factor_service
        self._inference_service = inference_service

    def run(
        self,
        trade_date: str,
        *,
        sync: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._validate_date(trade_date)
        self._progress_callback = progress_callback
        self._progress_current = 0
        self._progress_total = 9
        self._notify_progress("preflight", "running", "正在准备每日研究任务")
        started_at = datetime.now(timezone.utc)
        report: dict[str, Any] = {
            "task": "v8_daily_observation",
            "report_contract": V8_DAILY_REPORT_CONTRACT,
            "trade_date": trade_date,
            "run_id": started_at.strftime("%Y%m%dT%H%M%S%fZ"),
            "started_at_utc": started_at.isoformat(),
            "status": "running",
            "data_source": "Tushare Pro",
            "universe_policy": "dynamic_full_market_a_share",
            "capital_policy": "research_only_no_positions_no_amounts_no_orders",
            "sync_enabled": bool(sync),
            "stages": {},
        }

        try:
            service = self._sync_service or (DataSyncService.create() if sync else DataSyncService(client=None))
            if sync:
                self._run_stage(report, "stock_universe_sync", service.sync_stock_universe)
                self._run_stage(
                    report,
                    "daily_data_sync",
                    lambda: service.sync_calendar_batch(trade_date, trade_date, datasets=list(V8_DAILY_DATASETS)),
                    validator=lambda value: not value.get("failed_count"),
                    blocking_message="one or more real-data endpoints failed to synchronize",
                )
            else:
                report["stages"]["stock_universe_sync"] = {"status": "skipped", "reason": "--skip-sync"}
                self._progress_current += 1
                self._notify_progress("stock_universe_sync", "skipped", "使用已同步股票池")
                report["stages"]["daily_data_sync"] = {"status": "skipped", "reason": "--skip-sync"}
                self._progress_current += 1
                self._notify_progress("daily_data_sync", "skipped", "使用已同步交易日数据")

            self._run_stage(
                report,
                "stock_universe_audit",
                service.stock_universe_audit,
                validator=lambda value: value.get("status") == "pass",
                blocking_message="dynamic A-share universe audit did not pass",
            )
            self._run_stage(
                report,
                "trade_date_snapshot_gate",
                lambda: service.trade_date_snapshot_status(trade_date),
                validator=lambda value: value.get("status") == "pass",
                blocking_message="signal-date Tushare snapshot is incomplete",
            )
            self._run_stage(
                report,
                "foundation_gate",
                lambda: service.foundation_status(self.foundation_start_date, trade_date),
                validator=lambda value: value.get("status") == "pass",
                blocking_message="production foundation gate did not pass",
            )

            factor_service = self._factor_service or FactorLabService(FactorLabConfig(factor_set="factor_lab_v2"))
            self._run_stage(
                report,
                "factor_snapshot",
                lambda: factor_service.build_snapshot(trade_date),
                validator=lambda value: value.get("factor_set") == "factor_lab_v2" and int(value.get("rows") or 0) >= 1_000,
                blocking_message="factor_lab_v2 full-market snapshot is incomplete",
            )

            inference = self._inference_service or V8InferenceService(self.artifact_path)
            self._run_stage(report, "settlement", lambda: inference.settle(trade_date))
            score = self._run_stage(report, "scoring", lambda: inference.score_date(trade_date, persist=True))
            status = self._run_stage(report, "observation_status", inference.status)
            report["result"] = {
                "model_version": score.get("model_version"),
                "decision": score.get("decision"),
                "paper_eligible": bool(score.get("paper_eligible", False)),
                "diagnostic_only": bool(score.get("diagnostic_only", True)),
                "full_market_rows": int(score.get("full_market_rows") or 0),
                "diagnostic_candidate_count": len(score.get("diagnostic_candidates") or []),
                "paper_candidate_count": len(score.get("candidates") or []),
                "paper_ledger": score.get("paper_ledger") or {},
                "market_gate_decision": (score.get("market_gate") or {}).get("decision"),
                "research_gate_status": (score.get("research_gate") or {}).get("status"),
                "data_drift_status": (score.get("data_drift") or {}).get("status"),
                "paper_eligibility_checks": score.get("paper_eligibility_checks") or {},
                "observed_signal_days": int(status.get("observed_signal_days") or 0),
                "paper_signal_days": int(status.get("paper_signal_days") or 0),
                "promotion_allowed": False,
            }
            report["status"] = "pass"
        except Exception as exc:
            report["status"] = "blocked"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            artifact = self._persist_report(report)
            report["report_artifact"] = artifact
            raise V8DailyPipelineError(
                f"v8 daily run blocked for {trade_date}: {exc}; report={artifact['report_path']}",
                report,
            ) from exc

        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        artifact = self._persist_report(report)
        report["report_artifact"] = artifact
        self._notify_progress("completed", "pass", "每日研究任务完成", current=self._progress_total)
        return report

    @staticmethod
    def _validate_date(value: str) -> None:
        try:
            parsed = datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"trade_date must be YYYYMMDD, got {value!r}") from exc
        if parsed.strftime("%Y%m%d") != value:
            raise ValueError(f"trade_date must be YYYYMMDD, got {value!r}")

    def _run_stage(
        self,
        report: dict[str, Any],
        name: str,
        action,
        *,
        validator=None,
        blocking_message: str | None = None,
    ) -> dict[str, Any]:
        stage = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
        report["stages"][name] = stage
        self._notify_progress(name, "running", f"正在执行 {name}")
        try:
            result = action()
            if validator is not None and not validator(result):
                raise RuntimeError(f"{blocking_message or name}: {result.get('blocking') or result.get('failed_preview') or 'blocked'}")
        except Exception as exc:
            stage.update(
                {
                    "status": "blocked",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            self._notify_progress(name, "blocked", str(exc))
            raise
        stage.update(
            {
                "status": "pass",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "result": result,
            }
        )
        self._progress_current += 1
        self._notify_progress(name, "pass", f"{name} 已完成")
        return result

    def _notify_progress(
        self,
        stage: str,
        status: str,
        message: str,
        *,
        current: int | None = None,
    ) -> None:
        callback = getattr(self, "_progress_callback", None)
        if callback is None:
            return
        callback(
            {
                "stage": stage,
                "status": status,
                "message": message,
                "current": int(self._progress_current if current is None else current),
                "total": int(self._progress_total),
            }
        )

    def _persist_report(self, report: dict[str, Any]) -> dict[str, str]:
        run_dir = self.report_dir / report["trade_date"]
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / f"v8_daily_{report['run_id']}.json"
        payload = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        temp_path = report_path.with_suffix(".json.tmp")
        temp_path.write_bytes(payload)
        temp_path.replace(report_path)
        sha256 = hashlib.sha256(payload).hexdigest()
        sha_path = report_path.with_suffix(".sha256")
        sha_path.write_text(f"{sha256}  {report_path.name}\n", encoding="ascii")
        latest = {
            "report_contract": V8_DAILY_REPORT_CONTRACT,
            "trade_date": report["trade_date"],
            "status": report["status"],
            "report_path": str(report_path),
            "report_sha256": sha256,
        }
        latest_path = self.report_dir / "latest.json"
        latest_temp = latest_path.with_suffix(".json.tmp")
        latest_temp.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_temp.replace(latest_path)
        return {
            "report_path": str(report_path),
            "report_sha256": sha256,
            "sha256_path": str(sha_path),
            "latest_path": str(latest_path),
        }
