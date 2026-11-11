from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import func
from werkzeug.exceptions import HTTPException

from src.quant_trading.db import BacktestRun, Position, ReviewJournal, WebJob, session_scope
from src.quant_trading.ml_lab.v8_service import V8InferenceService
from src.quant_trading.config import settings
from src.quant_trading.services.backtest import BacktestService
from src.quant_trading.services.data_sync import DataSyncService
from src.quant_trading.services.portfolio import PortfolioService
from src.quant_trading.services.research_jobs import ResearchJobError, ResearchJobService
from src.quant_trading.services.review import ReviewService


api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def ok(data=None, status: int = 200):
    return jsonify({"success": True, "data": data, "request_id": g.request_id}), status


def fail(code: str, message: str, status: int = 400, details=None):
    return jsonify(
        {
            "success": False,
            "error": {"code": code, "message": message, "details": details},
            "request_id": g.request_id,
        }
    ), status


@api_v1.before_request
def attach_request_id() -> None:
    supplied = str(request.headers.get("X-Request-ID") or "").strip()
    g.request_id = supplied[:36] if supplied else str(uuid.uuid4())


@api_v1.errorhandler(ResearchJobError)
def handle_research_job_error(exc: ResearchJobError):
    return fail("research_job_error", str(exc), 400)


@api_v1.errorhandler(ValueError)
def handle_value_error(exc: ValueError):
    return fail("validation_error", str(exc), 400)


@api_v1.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return fail(exc.name.lower().replace(" ", "_"), str(exc.description), int(exc.code or 500))
    current_app.logger.exception("Unhandled /api/v1 error")
    return fail("internal_error", str(exc), 500)


@api_v1.get("/health")
def health():
    service = ResearchJobService.create()
    availability = service.calendar_availability()
    jobs = service.list_jobs(limit=10)
    running = [job for job in jobs if job["status"] in {"queued", "running", "cancel_requested"}]
    return ok(
        {
            "status": "ok",
            "database": "available",
            "latest_ready_date": availability.get("latest_ready_date"),
            "active_jobs": len(running),
            "worker_heartbeat": next((job.get("heartbeat_at") for job in running if job.get("heartbeat_at")), None),
            "canonical_url": f"http://{settings.flask.host}:{settings.flask.port}",
        }
    )


@api_v1.get("/calendar/availability")
def calendar_availability():
    return ok(ResearchJobService.create().calendar_availability(request.args.get("trade_date")))


@api_v1.post("/research-runs")
def create_research_run():
    payload = request.get_json(silent=True) or {}
    result = ResearchJobService.create().submit_daily_research(
        payload.get("trade_date"),
        execution_mode=str(payload.get("execution_mode") or "skip_sync"),
        request_id=g.request_id,
    )
    return ok(result, 202 if result.get("created") else 200)


@api_v1.get("/research-runs")
def list_research_runs():
    result = ResearchJobService.create().list_runs(
        trade_date=request.args.get("trade_date"),
        pipeline=request.args.get("pipeline"),
        limit=int(request.args.get("limit", "50")),
    )
    return ok(result)


@api_v1.get("/research-metrics")
def research_metrics():
    return ok(
        ResearchJobService.create().research_metrics(
            pipeline=str(request.args.get("pipeline") or "v11_top5_daily"),
            horizon=int(request.args.get("horizon", "5")),
        )
    )


@api_v1.get("/research-runs/<run_id>")
def research_run_detail(run_id: str):
    return ok(ResearchJobService.create().get_run(run_id))


@api_v1.post("/research-runs/compare")
def compare_research_runs():
    payload = request.get_json(silent=True) or {}
    return ok(ResearchJobService.create().compare_runs(list(payload.get("run_ids") or [])))


@api_v1.get("/jobs")
def list_jobs():
    return ok(
        ResearchJobService.create().list_jobs(
            limit=int(request.args.get("limit", "50")),
            status=request.args.get("status"),
        )
    )


@api_v1.get("/jobs/<job_id>")
def job_detail(job_id: str):
    return ok(ResearchJobService.create().get_job(job_id))


@api_v1.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    return ok(ResearchJobService.create().cancel_job(job_id))


@api_v1.post("/manual-selections")
def manual_selection():
    payload = request.get_json(silent=True) or {}
    if "candidate_id" not in payload or "selected" not in payload:
        return fail("validation_error", "candidate_id and selected are required", 400)
    return ok(
        ResearchJobService.create().record_manual_selection(
            int(payload["candidate_id"]),
            selected=payload["selected"] is True or str(payload["selected"]).lower() == "true",
            reason=payload.get("reason"),
            notes=payload.get("notes"),
        )
    )


@api_v1.post("/factor-jobs")
def factor_job():
    payload = request.get_json(silent=True) or {}
    return ok(
        ResearchJobService.create().submit_factor_evaluation(
            list(payload.get("horizons") or [1, 3, 5, 10, 20]),
            request_id=g.request_id,
        ),
        202,
    )


@api_v1.post("/data-jobs")
def data_job():
    payload = request.get_json(silent=True) or {}
    return ok(
        ResearchJobService.create().submit_data_sync(
            str(payload.get("action") or ""),
            trade_date=payload.get("trade_date"),
            request_id=g.request_id,
        ),
        202,
    )


@api_v1.get("/data/status")
def data_status():
    service = ResearchJobService.create()
    return ok(
        {
            "storage": DataSyncService.create().data_status(),
            "availability": service.calendar_availability(request.args.get("trade_date")),
        }
    )


@api_v1.get("/data/availability-audit")
def data_availability_audit():
    path = Path("reports/ml_lab/data_availability_audit_v1/data_availability_audit.json")
    if not path.exists():
        return ok(
            {
                "available": False,
                "path": str(path),
                "message": "尚未生成历史数据可用性审计。运行 scripts/audit_data_availability.py 后再查看。",
            }
        )
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    return ok({"available": True, "path": str(path), "report": report})


@api_v1.get("/dashboard")
def dashboard():
    jobs = ResearchJobService.create()
    inference = V8InferenceService(settings.research.v8_artifact_path)
    return ok(
        {
            "availability": jobs.calendar_availability(),
            "latest_runs": jobs.list_runs(limit=8),
            "active_jobs": [
                item for item in jobs.list_jobs(limit=20) if item["status"] in {"queued", "running", "cancel_requested"}
            ],
            "research_status": inference.status(),
            "portfolio": PortfolioService.create().portfolio_summary(),
        }
    )


@api_v1.get("/positions")
def positions():
    return ok(PortfolioService.create().evaluate_positions())


@api_v1.post("/positions")
def add_position():
    payload = request.get_json(silent=True) or {}
    required = ("ts_code", "name", "buy_price", "quantity")
    missing = [field for field in required if payload.get(field) in (None, "")]
    if missing:
        return fail("validation_error", f"Missing fields: {', '.join(missing)}", 400)
    position_id = PortfolioService.create().add_position(
        ts_code=str(payload["ts_code"]),
        name=str(payload["name"]),
        strategy_type=str(payload.get("strategy_type") or "manual"),
        buy_price=float(payload["buy_price"]),
        quantity=int(payload["quantity"]),
        notes=payload.get("notes"),
    )
    return ok({"id": position_id}, 201)


@api_v1.delete("/positions/<int:position_id>")
def delete_position(position_id: int):
    with session_scope() as session:
        row = session.get(Position, position_id)
        if row is None:
            return fail("not_found", f"Position not found: {position_id}", 404)
        session.delete(row)
    return ok({"deleted": position_id})


@api_v1.get("/backtests")
def backtests():
    with session_scope() as session:
        rows = session.query(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(50).all()
        result = [
            {
                "id": row.id,
                "strategy_type": row.strategy_type,
                "start_date": row.start_date,
                "end_date": row.end_date,
                "initial_capital": row.initial_capital,
                "final_value": row.final_value,
                "total_return": row.total_return,
                "max_drawdown": row.max_drawdown,
                "sharpe_ratio": row.sharpe_ratio,
                "metadata": row.metadata_json or {},
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    return ok(result)


@api_v1.post("/backtests")
def run_backtest():
    payload = request.get_json(silent=True) or {}
    return ok(
        BacktestService.create().run_buy_and_hold_backtest(
            ts_code=str(payload.get("ts_code") or ""),
            start_date=str(payload.get("start_date") or ""),
            end_date=str(payload.get("end_date") or ""),
            initial_capital=float(payload.get("initial_capital") or 100000),
        ),
        201,
    )


@api_v1.get("/reviews")
def reviews():
    return ok(
        ReviewService().list_entries(
            limit=int(request.args.get("limit", "100")),
            ts_code=request.args.get("ts_code"),
        )
    )


@api_v1.post("/reviews")
def add_review():
    payload = request.get_json(silent=True) or {}
    if not str(payload.get("title") or "").strip() or not str(payload.get("content") or "").strip():
        return fail("validation_error", "title and content are required", 400)
    entry_id = ReviewService().add_entry(
        title=str(payload["title"]),
        content=str(payload["content"]),
        ts_code=payload.get("ts_code"),
        name=payload.get("name"),
        category=str(payload.get("category") or "daily"),
        tags=list(payload.get("tags") or []),
    )
    return ok({"id": entry_id}, 201)


@api_v1.put("/reviews/<int:review_id>")
def update_review(review_id: int):
    payload = request.get_json(silent=True) or {}
    with session_scope() as session:
        row = session.get(ReviewJournal, review_id)
        if row is None:
            return fail("not_found", f"Review not found: {review_id}", 404)
        if "title" in payload:
            row.title = str(payload["title"]).strip()
        if "content" in payload:
            row.content = str(payload["content"]).strip()
        if "category" in payload:
            row.category = str(payload["category"])
        if "tags" in payload:
            row.tags = list(payload["tags"] or [])
    return ok({"updated": review_id})


@api_v1.delete("/reviews/<int:review_id>")
def delete_review(review_id: int):
    with session_scope() as session:
        row = session.get(ReviewJournal, review_id)
        if row is None:
            return fail("not_found", f"Review not found: {review_id}", 404)
        session.delete(row)
    return ok({"deleted": review_id})
