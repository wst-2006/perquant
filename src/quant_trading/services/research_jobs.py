from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import threading
import time
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from ..config import settings
from ..db import (
    CandidateOutcome,
    DailyBasic,
    DailyQuote,
    FactorSnapshot,
    IndexDaily,
    ManualSelection,
    MlV8CandidateSignal,
    MlV8CandidateOutcome,
    MoneyflowDaily,
    ResearchCandidate,
    ResearchRun,
    Stock,
    StockAdjFactor,
    StockLimitDaily,
    WebJob,
    init_db,
    session_scope,
)
from ..factor_lab.batch import FactorLabBatchRunner
from ..ml_lab.v8_daily import V8DailyPipeline, V8DailyPipelineError
from .data_sync import DataSyncService


ACTIVE_JOB_STATUSES = ("queued", "running", "cancel_requested")
FINAL_JOB_STATUSES = ("completed", "failed", "cancelled", "interrupted")


class ResearchJobError(RuntimeError):
    """Raised when a Web research job cannot be created or resolved."""


class JobCancelled(RuntimeError):
    """Raised between audited stages when cancellation was requested."""


def _utcnow() -> datetime:
    return datetime.utcnow()


def _date_string(value: Any, field: str = "trade_date") -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ResearchJobError(f"{field} must use YYYYMMDD, for example 20260716") from exc
    if parsed.strftime("%Y%m%d") != text:
        raise ResearchJobError(f"{field} must use YYYYMMDD, for example 20260716")
    if parsed.date() > datetime.now().date():
        raise ResearchJobError(f"{field} cannot be in the future: {text}")
    return text


@dataclass
class ResearchJobService:
    artifact_path: str = settings.research.v8_artifact_path
    report_dir: str = "reports/ml_lab/daily_v8"

    @classmethod
    def create(cls) -> "ResearchJobService":
        init_db()
        return cls()

    def submit_daily_research(
        self,
        trade_date: str,
        *,
        execution_mode: str = "skip_sync",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        trade_date = _date_string(trade_date)
        if execution_mode not in {"full", "skip_sync"}:
            raise ResearchJobError("execution_mode must be full or skip_sync")
        availability = self.calendar_availability(trade_date)
        if execution_mode == "skip_sync" and not availability["is_local_trade_day"]:
            raise ResearchJobError(f"No synchronized market snapshot is available for {trade_date}")

        dedupe_key = f"daily_research:{trade_date}:{execution_mode}"
        with session_scope() as session:
            active = (
                session.query(WebJob)
                .filter(WebJob.dedupe_key == dedupe_key, WebJob.status.in_(ACTIVE_JOB_STATUSES))
                .order_by(WebJob.created_at.desc())
                .first()
            )
            if active is not None:
                run = session.query(ResearchRun).filter(ResearchRun.job_id == active.id).one()
                return {"created": False, "job": self._job_dict(active), "run": self._run_dict(run)}

            version = int(
                session.query(func.coalesce(func.max(ResearchRun.version), 0))
                .filter(ResearchRun.trade_date == trade_date, ResearchRun.pipeline == "v11_top5_daily")
                .scalar()
                or 0
            ) + 1
            job = WebJob(
                job_type="daily_research",
                dedupe_key=dedupe_key,
                trade_date=trade_date,
                status="queued",
                stage="queued",
                progress_current=0,
                progress_total=9,
                message="已进入每日研究队列",
                params={"trade_date": trade_date, "execution_mode": execution_mode},
                request_id=request_id,
            )
            session.add(job)
            session.flush()
            run = ResearchRun(
                job_id=job.id,
                trade_date=trade_date,
                pipeline="v11_top5_daily",
                execution_mode=execution_mode,
                version=version,
                status="queued",
                diagnostic_only=True,
                promotion_allowed=False,
                data_version=availability,
            )
            session.add(run)
            session.flush()
            return {"created": True, "job": self._job_dict(job), "run": self._run_dict(run)}

    def submit_factor_evaluation(
        self,
        horizons: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        clean = sorted({int(value) for value in horizons if int(value) in {1, 3, 5, 10, 20}})
        if not clean:
            raise ResearchJobError("At least one horizon from 1,3,5,10,20 is required")
        return self._submit_generic_job(
            "factor_evaluate",
            {"horizons": clean},
            dedupe_key=f"factor_evaluate:{','.join(map(str, clean))}",
            request_id=request_id,
        )

    def submit_data_sync(
        self,
        action: str,
        *,
        trade_date: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        allowed = {"stock_universe", "daily_snapshot", "moneyflow", "daily_basic"}
        if action not in allowed:
            raise ResearchJobError(f"Unsupported data sync action: {action}")
        date_value = _date_string(trade_date) if trade_date else None
        if action != "stock_universe" and not date_value:
            raise ResearchJobError("trade_date is required for this data sync action")
        return self._submit_generic_job(
            "data_sync",
            {"action": action, "trade_date": date_value},
            trade_date=date_value,
            dedupe_key=f"data_sync:{action}:{date_value or 'all'}",
            request_id=request_id,
        )

    def _submit_generic_job(
        self,
        job_type: str,
        params: dict[str, Any],
        *,
        trade_date: str | None = None,
        dedupe_key: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        with session_scope() as session:
            if dedupe_key:
                active = (
                    session.query(WebJob)
                    .filter(WebJob.dedupe_key == dedupe_key, WebJob.status.in_(ACTIVE_JOB_STATUSES))
                    .order_by(WebJob.created_at.desc())
                    .first()
                )
                if active is not None:
                    return {"created": False, "job": self._job_dict(active)}
            job = WebJob(
                job_type=job_type,
                dedupe_key=dedupe_key,
                trade_date=trade_date,
                status="queued",
                stage="queued",
                progress_current=0,
                progress_total=1,
                message="任务已进入队列",
                params=params,
                request_id=request_id,
            )
            session.add(job)
            session.flush()
            return {"created": True, "job": self._job_dict(job)}

    def calendar_availability(self, trade_date: str | None = None, *, recent_limit: int = 40) -> dict[str, Any]:
        date_value = _date_string(trade_date) if trade_date else None
        with session_scope() as session:
            latest = {
                "daily": session.query(func.max(DailyQuote.trade_date)).scalar(),
                "daily_basic": session.query(func.max(DailyBasic.trade_date)).scalar(),
                "moneyflow": session.query(func.max(MoneyflowDaily.trade_date)).scalar(),
                "adj_factor": session.query(func.max(StockAdjFactor.trade_date)).scalar(),
                "stk_limit": session.query(func.max(StockLimitDaily.trade_date)).scalar(),
                "index_daily": session.query(func.max(IndexDaily.trade_date)).scalar(),
                "factor_snapshot": session.query(func.max(FactorSnapshot.trade_date)).scalar(),
            }
            listed_count = int(session.query(func.count()).select_from(Stock).filter(Stock.list_status == "L").scalar() or 0)
            historical_count = int(session.query(func.count()).select_from(Stock).scalar() or 0)
            recent_dates = [
                row[0]
                for row in session.query(DailyQuote.trade_date)
                .group_by(DailyQuote.trade_date)
                .order_by(DailyQuote.trade_date.desc())
                .limit(max(1, min(int(recent_limit), 120)))
                .all()
            ]
            counts: dict[str, int] = {}
            if date_value:
                for name, model in (
                    ("daily", DailyQuote),
                    ("daily_basic", DailyBasic),
                    ("moneyflow", MoneyflowDaily),
                    ("adj_factor", StockAdjFactor),
                    ("stk_limit", StockLimitDaily),
                    ("index_daily", IndexDaily),
                ):
                    counts[name] = int(session.query(func.count()).select_from(model).filter(model.trade_date == date_value).scalar() or 0)
                counts["factor_snapshot_v2"] = int(
                    session.query(func.count())
                    .select_from(FactorSnapshot)
                    .filter(FactorSnapshot.trade_date == date_value, FactorSnapshot.factor_set == "factor_lab_v2")
                    .scalar()
                    or 0
                )
        complete_components = [name for name in ("daily", "daily_basic", "moneyflow", "adj_factor", "stk_limit") if counts.get(name, 0) >= 1000]
        return {
            "trade_date": date_value,
            "is_local_trade_day": bool(counts.get("daily", 0)) if date_value else None,
            "snapshot_ready": len(complete_components) == 5 if date_value else None,
            "component_counts": counts,
            "complete_components": complete_components,
            "latest_dates": latest,
            "latest_ready_date": min((value for value in latest.values() if value), default=None),
            "recent_trade_dates": recent_dates,
            "listed_stock_count": listed_count,
            "historical_stock_count": historical_count,
            "stock_count_policy": "listed_stock_count is the active universe; historical_stock_count includes delisted and paused records",
        }

    def list_jobs(self, *, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        with session_scope() as session:
            query = session.query(WebJob)
            if status:
                query = query.filter(WebJob.status == status)
            rows = query.order_by(WebJob.created_at.desc()).limit(max(1, min(int(limit), 200))).all()
            return [self._job_dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with session_scope() as session:
            row = session.get(WebJob, job_id)
            if row is None:
                raise ResearchJobError(f"Job not found: {job_id}")
            result = self._job_dict(row)
            run = session.query(ResearchRun).filter(ResearchRun.job_id == row.id).one_or_none()
            if run is not None:
                result["research_run_id"] = run.id
                result["research_run_status"] = run.status
            return result

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with session_scope() as session:
            row = session.get(WebJob, job_id)
            if row is None:
                raise ResearchJobError(f"Job not found: {job_id}")
            if row.status in FINAL_JOB_STATUSES:
                return self._job_dict(row)
            row.cancel_requested = True
            if row.status == "queued":
                row.status = "cancelled"
                row.stage = "cancelled"
                row.completed_at = _utcnow()
                run = session.query(ResearchRun).filter(ResearchRun.job_id == row.id).one_or_none()
                if run is not None:
                    run.status = "cancelled"
                    run.completed_at = row.completed_at
            else:
                row.status = "cancel_requested"
                row.message = "将在当前审计阶段完成后取消"
            session.flush()
            return self._job_dict(row)

    def list_runs(
        self,
        *,
        trade_date: str | None = None,
        limit: int = 50,
        pipeline: str | None = None,
    ) -> list[dict[str, Any]]:
        date_value = _date_string(trade_date) if trade_date else None
        with session_scope() as session:
            query = session.query(ResearchRun)
            if date_value:
                query = query.filter(ResearchRun.trade_date == date_value)
            if pipeline:
                query = query.filter(ResearchRun.pipeline == pipeline)
            rows = query.order_by(ResearchRun.trade_date.desc(), ResearchRun.version.desc()).limit(max(1, min(int(limit), 200))).all()
            return [self._run_dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with session_scope() as session:
            run = session.get(ResearchRun, run_id)
            if run is None:
                raise ResearchJobError(f"Research run not found: {run_id}")
            candidates = (
                session.query(ResearchCandidate)
                .filter(ResearchCandidate.research_run_id == run.id)
                .order_by(ResearchCandidate.rank.asc())
                .all()
            )
            candidate_ids = [row.id for row in candidates]
            selections = {
                row.research_candidate_id: row
                for row in session.query(ManualSelection).filter(ManualSelection.research_candidate_id.in_(candidate_ids)).all()
            } if candidate_ids else {}
            outcomes = (
                session.query(CandidateOutcome)
                .filter(CandidateOutcome.research_candidate_id.in_(candidate_ids))
                .order_by(CandidateOutcome.research_candidate_id, CandidateOutcome.horizon)
                .all()
            ) if candidate_ids else []
            outcome_map: dict[int, list[dict[str, Any]]] = {}
            for row in outcomes:
                outcome_map.setdefault(row.research_candidate_id, []).append(self._outcome_dict(row))
            payload = self._run_dict(run)
            payload["candidates"] = [
                {
                    **self._candidate_dict(row),
                    "manual_selection": self._manual_dict(selections.get(row.id)),
                    "outcomes": outcome_map.get(row.id, []),
                }
                for row in candidates
            ]
            return payload

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        unique = list(dict.fromkeys(run_ids))
        if len(unique) < 2 or len(unique) > 4:
            raise ResearchJobError("Compare requires two to four research run IDs")
        return {"runs": [self.get_run(run_id) for run_id in unique]}

    def record_manual_selection(
        self,
        candidate_id: int,
        *,
        selected: bool,
        reason: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        with session_scope() as session:
            candidate = session.get(ResearchCandidate, int(candidate_id))
            if candidate is None:
                raise ResearchJobError(f"Research candidate not found: {candidate_id}")
            stmt = insert(ManualSelection).values(
                research_candidate_id=candidate.id,
                selected=bool(selected),
                reason=(reason or "").strip() or None,
                notes=(notes or "").strip() or None,
                created_at=_utcnow(),
                updated_at=_utcnow(),
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_manual_selection_candidate",
                set_={
                    "selected": stmt.excluded.selected,
                    "reason": stmt.excluded.reason,
                    "notes": stmt.excluded.notes,
                    "updated_at": stmt.excluded.updated_at,
                },
            ).returning(ManualSelection.id)
            selection_id = int(session.execute(stmt).scalar_one())
            row = session.get(ManualSelection, selection_id)
            return self._manual_dict(row) or {}

    def process_next_job(self) -> dict[str, Any] | None:
        with session_scope() as session:
            job = (
                session.query(WebJob)
                .filter(WebJob.status == "queued")
                .order_by(WebJob.created_at.asc())
                .with_for_update(skip_locked=True)
                .first()
            )
            if job is None:
                return None
            job.status = "running"
            job.stage = "starting"
            job.message = "Worker 已接收任务"
            job.started_at = _utcnow()
            job.heartbeat_at = job.started_at
            job_id = job.id
            session.flush()

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, heartbeat_stop),
            name=f"web-job-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            result = self._execute_job(job_id)
        except JobCancelled as exc:
            self._finish_job(job_id, "cancelled", error={"type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            if isinstance(cause, JobCancelled):
                error = {"type": type(cause).__name__, "message": str(cause)}
                self._finish_research_run(job_id, "cancelled", error=error)
                self._finish_job(job_id, "cancelled", error=error)
                return self.get_job(job_id)
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, V8DailyPipelineError):
                error["report_artifact"] = exc.report.get("report_artifact")
                self._finish_research_run(job_id, "failed", report=exc.report, error=error)
            else:
                self._finish_research_run(job_id, "failed", error=error)
            self._finish_job(job_id, "failed", error=error)
        else:
            self._finish_job(job_id, "completed", result=result)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
        return self.get_job(job_id)

    @staticmethod
    def _heartbeat_loop(job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(15.0):
            try:
                with session_scope() as session:
                    row = session.get(WebJob, job_id)
                    if row is None or row.status not in ACTIVE_JOB_STATUSES:
                        return
                    row.heartbeat_at = _utcnow()
            except Exception:
                # The main Worker path owns job failure reporting; heartbeat retries quietly.
                continue

    def _execute_job(self, job_id: str) -> dict[str, Any]:
        with session_scope() as session:
            job = session.get(WebJob, job_id)
            if job is None:
                raise ResearchJobError(f"Job not found: {job_id}")
            job_type = job.job_type
            params = dict(job.params or {})
        if job_type == "daily_research":
            return self._execute_daily_research(job_id, params)
        if job_type == "factor_evaluate":
            return self._execute_factor(job_id, params)
        if job_type == "data_sync":
            return self._execute_data_sync(job_id, params)
        raise ResearchJobError(f"Unsupported job type: {job_type}")

    def _execute_daily_research(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        trade_date = params["trade_date"]
        mode = params.get("execution_mode", "skip_sync")
        with session_scope() as session:
            run = session.query(ResearchRun).filter(ResearchRun.job_id == job_id).one()
            run.status = "running"
            run.started_at = _utcnow()

        def progress(event: dict[str, Any]) -> None:
            with session_scope() as session:
                job = session.get(WebJob, job_id)
                if job is None:
                    raise JobCancelled("Job record was removed")
                if job.cancel_requested:
                    raise JobCancelled("任务已按用户请求取消")
                job.stage = str(event.get("stage") or "running")
                job.progress_current = int(event.get("current") or 0)
                job.progress_total = int(event.get("total") or 9)
                job.message = str(event.get("message") or "")
                job.heartbeat_at = _utcnow()

        report = V8DailyPipeline(
            artifact_path=self.artifact_path,
            report_dir=self.report_dir,
        ).run(trade_date, sync=mode == "full", progress_callback=progress)
        self._persist_research_snapshot(job_id, report)
        self.sync_candidate_outcomes()
        run = self._run_for_job(job_id)
        return {
            "research_run_id": run.id,
            "trade_date": run.trade_date,
            "version": run.version,
            "decision": run.decision,
            "candidate_count": len((run.paper_ledger or {}).get("research_top5") or []),
            "report_artifact": run.report_artifact,
        }

    def _execute_factor(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        self._update_job(job_id, stage="factor_evaluate", current=0, total=1, message="正在执行 Factor Lab 验证")
        result = FactorLabBatchRunner.create().evaluate_validation_suite(horizons=tuple(params["horizons"]))
        self._update_job(job_id, stage="factor_evaluate", current=1, total=1, message="Factor Lab 验证完成")
        return result

    def _execute_data_sync(self, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
        service = DataSyncService.create()
        action = params["action"]
        trade_date = params.get("trade_date")
        self._update_job(job_id, stage=action, current=0, total=1, message=f"正在同步 {action}")
        if action == "stock_universe":
            result = service.sync_stock_universe()
        elif action == "daily_snapshot":
            result = service.sync_calendar_batch(
                trade_date,
                trade_date,
                datasets=["daily", "daily_basic", "moneyflow", "adj_factor", "stk_limit", "index_daily"],
            )
        elif action == "moneyflow":
            result = service.sync_moneyflow_full_market(trade_date, trade_date)
        else:
            result = service.sync_daily_basic(trade_date)
        self._update_job(job_id, stage=action, current=1, total=1, message=f"{action} 同步完成")
        return result

    def _persist_research_snapshot(self, job_id: str, report: dict[str, Any]) -> None:
        scoring = ((report.get("stages") or {}).get("scoring") or {}).get("result") or {}
        result = report.get("result") or {}
        ledger = result.get("paper_ledger") or scoring.get("paper_ledger") or {}
        candidates = ledger.get("research_top5") or scoring.get("diagnostic_candidates") or scoring.get("candidates") or []
        artifact_path = Path(self.artifact_path)
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest() if artifact_path.exists() else None
        availability = self.calendar_availability(report["trade_date"])
        audit = {
            name: (report.get("stages") or {}).get(name)
            for name in ("stock_universe_audit", "trade_date_snapshot_gate", "foundation_gate", "factor_snapshot")
        }
        now = _utcnow()
        with session_scope() as session:
            run = session.query(ResearchRun).filter(ResearchRun.job_id == job_id).one()
            run.status = "completed"
            run.decision = str(result.get("decision") or scoring.get("decision") or "unknown")
            run.model_version = result.get("model_version") or scoring.get("model_version")
            run.artifact_sha256 = artifact_sha
            run.diagnostic_only = bool(result.get("diagnostic_only", True))
            run.promotion_allowed = bool(result.get("promotion_allowed", False))
            run.data_version = availability
            run.audit = audit
            run.market_gate = scoring.get("market_gate") or {}
            run.paper_ledger = ledger
            run.result_summary = result
            run.report_artifact = report.get("report_artifact") or {}
            run.completed_at = now
            for row in candidates:
                session.add(
                    ResearchCandidate(
                        research_run_id=run.id,
                        rank=int(row.get("rank") or 0),
                        ts_code=str(row.get("ts_code") or ""),
                        name=row.get("name"),
                        industry=row.get("industry"),
                        candidate_type=str(row.get("candidate_type") or "stable"),
                        decision=str(row.get("decision") or run.decision or "unknown"),
                        overall_score=row.get("overall_score"),
                        stable_score=row.get("stable_score"),
                        aggressive_score=row.get("aggressive_score"),
                        p_hit_5pct_10d=row.get("p_hit_5pct_10d"),
                        expected_net_return_5d=row.get("expected_net_return_5d"),
                        expected_mfe_10d=row.get("expected_mfe_10d"),
                        expected_mae_10d=row.get("expected_mae_10d"),
                        model_disagreement=row.get("model_disagreement"),
                        evidence=row.get("evidence") or {},
                        risk_tags=row.get("risk_tags") or [],
                    )
                )

    def import_legacy_v8_runs(self) -> dict[str, Any]:
        imported = 0
        candidates_imported = 0
        with session_scope() as session:
            groups = (
                session.query(MlV8CandidateSignal.signal_date, MlV8CandidateSignal.model_version)
                .group_by(MlV8CandidateSignal.signal_date, MlV8CandidateSignal.model_version)
                .order_by(MlV8CandidateSignal.signal_date.asc())
                .all()
            )
        for signal_date, model_version in groups:
            with session_scope() as session:
                exists = (
                    session.query(ResearchRun)
                    .filter(
                        ResearchRun.trade_date == signal_date,
                        ResearchRun.pipeline == "legacy_import",
                        ResearchRun.model_version == model_version,
                    )
                    .first()
                )
                if exists is not None:
                    continue
                rows = (
                    session.query(MlV8CandidateSignal)
                    .filter(MlV8CandidateSignal.signal_date == signal_date, MlV8CandidateSignal.model_version == model_version)
                    .order_by(MlV8CandidateSignal.rank.asc())
                    .all()
                )
                if not rows:
                    continue
                version = int(
                    session.query(func.coalesce(func.max(ResearchRun.version), 0))
                    .filter(ResearchRun.trade_date == signal_date, ResearchRun.pipeline == "legacy_import")
                    .scalar()
                    or 0
                ) + 1
                run = ResearchRun(
                    trade_date=signal_date,
                    pipeline="legacy_import",
                    execution_mode="legacy_import",
                    version=version,
                    status="completed",
                    decision=rows[0].decision,
                    model_version=model_version,
                    diagnostic_only=True,
                    promotion_allowed=False,
                    data_version=self.calendar_availability(signal_date),
                    audit={"status": "legacy_import", "source_table": "ml_v8_candidate_signals"},
                    market_gate={},
                    paper_ledger={
                        "contract": "ml_v12_paper_ledger_v1",
                        "research_top5": [],
                        "default_70_30_portfolio": [],
                        "manual_selection": {"status": "awaiting_user_review"},
                        "diagnostic_only": True,
                    },
                    result_summary={"source": "legacy_import", "candidate_count": len(rows)},
                    created_at=min(row.created_at for row in rows),
                    completed_at=max(row.created_at for row in rows),
                )
                session.add(run)
                session.flush()
                ledger_rows = []
                for row in rows:
                    payload = {
                        "rank": row.rank,
                        "ts_code": row.ts_code,
                        "name": row.name,
                        "industry": row.industry,
                        "candidate_type": (row.evidence or {}).get("candidate_type") or "stable",
                        "decision": row.decision,
                        "overall_score": row.overall_score,
                        "stable_score": (row.evidence or {}).get("stable_score"),
                        "aggressive_score": (row.evidence or {}).get("aggressive_score"),
                        "p_hit_5pct_10d": row.p_hit_5pct_10d,
                        "expected_net_return_5d": row.expected_net_return_5d,
                        "expected_mfe_10d": row.expected_mfe_10d,
                        "expected_mae_10d": row.expected_mae_10d,
                        "model_disagreement": row.model_disagreement,
                        "evidence": row.evidence or {},
                        "risk_tags": row.risk_tags or [],
                    }
                    ledger_rows.append(payload)
                    session.add(ResearchCandidate(research_run_id=run.id, **payload))
                run.paper_ledger = {**(run.paper_ledger or {}), "research_top5": ledger_rows}
                imported += 1
                candidates_imported += len(rows)
        outcomes_imported = self.sync_candidate_outcomes()
        return {
            "runs_imported": imported,
            "candidates_imported": candidates_imported,
            "outcomes_imported": outcomes_imported,
        }

    def sync_candidate_outcomes(self) -> int:
        """Mirror audited v8 settlements into immutable website run candidates."""
        with session_scope() as session:
            rows = (
                session.query(MlV8CandidateOutcome, MlV8CandidateSignal)
                .join(MlV8CandidateSignal, MlV8CandidateSignal.id == MlV8CandidateOutcome.signal_id)
                .all()
            )
            candidates = (
                session.query(ResearchCandidate, ResearchRun)
                .join(ResearchRun, ResearchRun.id == ResearchCandidate.research_run_id)
                .all()
            )
            candidate_by_key = {
                (run.trade_date, candidate.ts_code, run.model_version): candidate.id
                for candidate, run in candidates
            }
            candidate_by_date_code = {
                (run.trade_date, candidate.ts_code): candidate.id
                for candidate, run in candidates
                if run.pipeline == "legacy_import"
            }
            persisted = 0
            for outcome, signal in rows:
                candidate_id = candidate_by_key.get((signal.signal_date, signal.ts_code, signal.model_version))
                if candidate_id is None:
                    candidate_id = candidate_by_date_code.get((signal.signal_date, signal.ts_code))
                if candidate_id is None:
                    continue
                stmt = insert(CandidateOutcome).values(
                    research_candidate_id=candidate_id,
                    horizon=outcome.horizon,
                    settlement_date=outcome.settlement_date,
                    fixed_tradable=outcome.fixed_tradable,
                    net_return=outcome.net_return,
                    mfe_close=outcome.mfe_close,
                    mae_close=outcome.mae_close,
                    hit_5pct=outcome.hit_5pct,
                    hit_8pct=outcome.hit_8pct,
                    hit_12pct=outcome.hit_12pct,
                    details={**(outcome.details or {}), "source": "ml_v8_candidate_outcomes"},
                    created_at=outcome.created_at or _utcnow(),
                ).on_conflict_do_update(
                    constraint="uq_candidate_outcome_horizon",
                    set_={
                        "settlement_date": outcome.settlement_date,
                        "fixed_tradable": outcome.fixed_tradable,
                        "net_return": outcome.net_return,
                        "mfe_close": outcome.mfe_close,
                        "mae_close": outcome.mae_close,
                        "hit_5pct": outcome.hit_5pct,
                        "hit_8pct": outcome.hit_8pct,
                        "hit_12pct": outcome.hit_12pct,
                        "details": {**(outcome.details or {}), "source": "ml_v8_candidate_outcomes"},
                    },
                )
                session.execute(stmt)
                persisted += 1
            return persisted

    def research_metrics(self, *, pipeline: str = "v11_top5_daily", horizon: int = 5) -> dict[str, Any]:
        if horizon not in {1, 3, 5, 10, 20}:
            raise ResearchJobError("horizon must be one of 1,3,5,10,20")
        with session_scope() as session:
            runs = (
                session.query(ResearchRun)
                .filter(ResearchRun.pipeline == pipeline, ResearchRun.status == "completed")
                .order_by(ResearchRun.trade_date.asc(), ResearchRun.version.desc())
                .all()
            )
            latest_by_date: dict[str, ResearchRun] = {}
            for run in runs:
                latest_by_date.setdefault(run.trade_date, run)
            run_ids = [run.id for run in latest_by_date.values()]
            candidates = (
                session.query(ResearchCandidate)
                .filter(ResearchCandidate.research_run_id.in_(run_ids))
                .all()
            ) if run_ids else []
            candidate_ids = [row.id for row in candidates]
            outcomes = (
                session.query(CandidateOutcome)
                .filter(CandidateOutcome.research_candidate_id.in_(candidate_ids), CandidateOutcome.horizon == horizon)
                .all()
            ) if candidate_ids else []
        candidates_by_run: dict[str, list[ResearchCandidate]] = {}
        for row in candidates:
            candidates_by_run.setdefault(row.research_run_id, []).append(row)
        outcome_by_candidate = {row.research_candidate_id: row for row in outcomes}
        daily_rows: list[dict[str, Any]] = []
        unsettled_days = 0
        for trade_date, run in sorted(latest_by_date.items()):
            run_candidates = candidates_by_run.get(run.id, [])
            is_no_signal = str(run.decision or "").endswith("no_signal")
            if is_no_signal or not run_candidates:
                daily_rows.append({"trade_date": trade_date, "net_return": 0.0, "all_loss": False, "hit_5pct": False, "signal": False})
                continue
            run_outcomes = [outcome_by_candidate.get(row.id) for row in run_candidates]
            if any(row is None for row in run_outcomes):
                unsettled_days += 1
                continue
            values = [float(row.net_return) for row in run_outcomes if row and row.net_return is not None]
            if len(values) != len(run_outcomes):
                unsettled_days += 1
                continue
            daily_rows.append(
                {
                    "trade_date": trade_date,
                    "net_return": round(sum(values) / len(values), 8),
                    "all_loss": all(value < 0 for value in values),
                    "hit_5pct": any(bool(row.hit_5pct) for row in run_outcomes if row),
                    "signal": True,
                }
            )
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for row in daily_rows:
            equity *= 1.0 + float(row["net_return"])
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
            row["equity"] = round(equity, 8)
        signal_rows = [row for row in daily_rows if row["signal"]]
        return {
            "pipeline": pipeline,
            "horizon": horizon,
            "version_policy": "latest_completed_version_per_trade_date",
            "calendar_days": len(daily_rows),
            "settled_signal_days": len(signal_rows),
            "no_signal_days": sum(not row["signal"] for row in daily_rows),
            "unsettled_signal_days": unsettled_days,
            "mean_net_return": round(sum(row["net_return"] for row in daily_rows) / len(daily_rows), 8) if daily_rows else None,
            "positive_signal_day_rate": round(sum(row["net_return"] > 0 for row in signal_rows) / len(signal_rows), 8) if signal_rows else None,
            "all_loss_rate": round(sum(row["all_loss"] for row in signal_rows) / len(signal_rows), 8) if signal_rows else None,
            "hit_5pct_rate": round(sum(row["hit_5pct"] for row in signal_rows) / len(signal_rows), 8) if signal_rows else None,
            "max_drawdown": round(max_drawdown, 8) if daily_rows else None,
            "daily": daily_rows,
        }

    def recover_orphaned_jobs(self, *, stale_after_seconds: int = 180) -> int:
        cutoff = _utcnow() - timedelta(seconds=max(30, int(stale_after_seconds)))
        with session_scope() as session:
            rows = (
                session.query(WebJob)
                .filter(WebJob.status.in_(("running", "cancel_requested")), WebJob.heartbeat_at < cutoff)
                .all()
            )
            for row in rows:
                row.status = "interrupted"
                row.stage = "interrupted"
                row.message = "Worker 心跳中断；任务已停止，历史记录保留"
                row.error = {"type": "WorkerInterrupted", "message": row.message}
                row.completed_at = _utcnow()
                run = session.query(ResearchRun).filter(ResearchRun.job_id == row.id).one_or_none()
                if run is not None and run.status in {"queued", "running"}:
                    run.status = "interrupted"
                    run.error = row.error
                    run.completed_at = row.completed_at
            return len(rows)

    def run_worker(self, *, poll_seconds: float = 1.0, once: bool = False) -> None:
        self.recover_orphaned_jobs()
        while True:
            result = self.process_next_job()
            if once:
                return
            if result is None:
                time.sleep(max(0.2, float(poll_seconds)))

    def _update_job(self, job_id: str, *, stage: str, current: int, total: int, message: str) -> None:
        with session_scope() as session:
            row = session.get(WebJob, job_id)
            if row is None:
                raise JobCancelled("Job record was removed")
            if row.cancel_requested:
                raise JobCancelled("任务已按用户请求取消")
            row.stage = stage
            row.progress_current = current
            row.progress_total = total
            row.message = message
            row.heartbeat_at = _utcnow()

    def _finish_job(
        self,
        job_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with session_scope() as session:
            row = session.get(WebJob, job_id)
            if row is None:
                return
            row.status = status
            row.stage = status
            row.result = result
            row.error = error
            row.completed_at = _utcnow()
            row.heartbeat_at = row.completed_at
            row.progress_current = row.progress_total if status == "completed" else row.progress_current
            row.message = "任务完成" if status == "completed" else (error or {}).get("message")

    def _finish_research_run(
        self,
        job_id: str,
        status: str,
        *,
        report: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        with session_scope() as session:
            run = session.query(ResearchRun).filter(ResearchRun.job_id == job_id).one_or_none()
            if run is None or run.status == "completed":
                return
            run.status = status
            run.error = error
            run.completed_at = _utcnow()
            if report:
                run.audit = report.get("stages") or {}
                run.result_summary = report.get("result") or {}
                run.report_artifact = report.get("report_artifact") or {}

    def _run_for_job(self, job_id: str) -> ResearchRun:
        with session_scope() as session:
            run = session.query(ResearchRun).filter(ResearchRun.job_id == job_id).one_or_none()
            if run is None:
                raise ResearchJobError(f"Research run not found for job: {job_id}")
            return run

    @staticmethod
    def _job_dict(row: WebJob) -> dict[str, Any]:
        progress = float(row.progress_current or 0) / max(int(row.progress_total or 1), 1)
        return {
            "id": row.id,
            "job_type": row.job_type,
            "trade_date": row.trade_date,
            "status": row.status,
            "stage": row.stage,
            "progress_current": row.progress_current,
            "progress_total": row.progress_total,
            "progress_pct": round(progress * 100, 1),
            "message": row.message,
            "params": row.params or {},
            "result": row.result,
            "error": row.error,
            "cancel_requested": bool(row.cancel_requested),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    @staticmethod
    def _run_dict(row: ResearchRun) -> dict[str, Any]:
        return {
            "id": row.id,
            "job_id": row.job_id,
            "trade_date": row.trade_date,
            "pipeline": row.pipeline,
            "execution_mode": row.execution_mode,
            "version": row.version,
            "version_label": f"v{row.version}",
            "status": row.status,
            "decision": row.decision,
            "model_version": row.model_version,
            "artifact_sha256": row.artifact_sha256,
            "diagnostic_only": bool(row.diagnostic_only),
            "promotion_allowed": bool(row.promotion_allowed),
            "data_version": row.data_version or {},
            "audit": row.audit or {},
            "market_gate": row.market_gate or {},
            "paper_ledger": row.paper_ledger or {},
            "result_summary": row.result_summary or {},
            "report_artifact": row.report_artifact or {},
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    @staticmethod
    def _candidate_dict(row: ResearchCandidate) -> dict[str, Any]:
        return {
            "id": row.id,
            "rank": row.rank,
            "ts_code": row.ts_code,
            "name": row.name,
            "industry": row.industry,
            "candidate_type": row.candidate_type,
            "decision": row.decision,
            "overall_score": row.overall_score,
            "stable_score": row.stable_score,
            "aggressive_score": row.aggressive_score,
            "p_hit_5pct_10d": row.p_hit_5pct_10d,
            "expected_net_return_5d": row.expected_net_return_5d,
            "expected_mfe_10d": row.expected_mfe_10d,
            "expected_mae_10d": row.expected_mae_10d,
            "model_disagreement": row.model_disagreement,
            "evidence": row.evidence or {},
            "risk_tags": row.risk_tags or [],
        }

    @staticmethod
    def _manual_dict(row: ManualSelection | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row.id,
            "selected": bool(row.selected),
            "reason": row.reason,
            "notes": row.notes,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    @staticmethod
    def _outcome_dict(row: CandidateOutcome) -> dict[str, Any]:
        return {
            "horizon": row.horizon,
            "settlement_date": row.settlement_date,
            "fixed_tradable": bool(row.fixed_tradable),
            "net_return": row.net_return,
            "mfe_close": row.mfe_close,
            "mae_close": row.mae_close,
            "hit_5pct": row.hit_5pct,
            "hit_8pct": row.hit_8pct,
            "hit_12pct": row.hit_12pct,
            "details": row.details or {},
        }
