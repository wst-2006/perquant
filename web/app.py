from __future__ import annotations

import json
from pathlib import Path
import re
import uuid

from flask import Flask, g, jsonify, render_template, request
try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - dependency is optional for local tests
    CORS = None

from src.quant_trading.analysis.market import MarketAnalyzer
from src.quant_trading.analysis.sector import SectorAnalyzer
from src.quant_trading.config import ConfigError, settings
from src.quant_trading.data.tushare_client import TushareClient, TushareDataError
from src.quant_trading.db import init_db
from src.quant_trading.services.backtest import BacktestService
from src.quant_trading.services.data_sync import DataSyncError, DataSyncService
from src.quant_trading.services.portfolio import PortfolioService
from src.quant_trading.services.review import ReviewService
from src.quant_trading.services.selection import SUPPORTED_STRATEGY_TYPES, SelectionRuleError, SelectionService
from src.quant_trading.services.stock_profile import StockProfileService
from src.quant_trading.ml_lab.service import MlLabService
from src.quant_trading.ml_lab.v8_service import V8InferenceError, V8InferenceService
from src.quant_trading.factor_lab.batch import FactorLabBatchRunner
from sqlalchemy.exc import OperationalError
from werkzeug.exceptions import HTTPException

from web.api_v1 import api_v1


def v8_inference_service() -> V8InferenceService:
    return V8InferenceService(settings.research.v8_artifact_path)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.flask.secret_key
    app.register_blueprint(api_v1)

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        if request.path.startswith("/api/v1"):
            request_id = getattr(g, "request_id", None) or str(uuid.uuid4())
            if isinstance(exc, HTTPException):
                return jsonify(
                    {
                        "success": False,
                        "error": {
                            "code": exc.name.lower().replace(" ", "_"),
                            "message": str(exc.description),
                            "details": None,
                        },
                        "request_id": request_id,
                    }
                ), exc.code
            app.logger.exception("Unhandled /api/v1 request error")
            return jsonify(
                {
                    "success": False,
                    "error": {"code": "internal_error", "message": str(exc), "details": None},
                    "request_id": request_id,
                }
            ), 500
        if isinstance(exc, HTTPException):
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "error": str(exc.description), "error_code": exc.name.lower().replace(" ", "_")}), exc.code
            return exc
        app.logger.exception("Unhandled request error")
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": str(exc), "error_code": "internal_error"}), 500
        return "Internal server error", 500

    @app.errorhandler(ConfigError)
    def handle_config_error(exc: ConfigError):
        return jsonify({"success": False, "error": str(exc)}), 400

    @app.errorhandler(TushareDataError)
    def handle_tushare_error(exc: TushareDataError):
        return jsonify({"success": False, "error": str(exc)}), 502

    @app.errorhandler(DataSyncError)
    def handle_data_sync_error(exc: DataSyncError):
        return jsonify({"success": False, "error": str(exc)}), 502

    @app.errorhandler(SelectionRuleError)
    def handle_selection_rule_error(exc: SelectionRuleError):
        return jsonify({"success": False, "error": str(exc)}), 400

    @app.errorhandler(ValueError)
    def handle_value_error(exc: ValueError):
        return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/config")
    def api_config():
        return jsonify({"success": True, "data": settings.as_public_dict()})

    @app.route("/api/health")
    def health():
        return jsonify({"success": True, "status": "ok"})

    @app.route("/api/init_db", methods=["POST"])
    def api_init_db():
        init_db()
        return jsonify({"success": True})

    @app.route("/api/data/status")
    def api_data_status():
        result = DataSyncService.create().data_status()
        return jsonify({"success": True, "data": result})

    @app.route("/api/data/sync/stock_pool", methods=["POST"])
    def api_sync_stock_pool():
        result = DataSyncService.create().sync_stock_pool()
        return jsonify({"success": True, "data": result})

    @app.route("/api/data/sync/daily", methods=["POST"])
    def api_sync_daily():
        data = request.get_json(silent=True) or {}
        result = DataSyncService.create().sync_daily_quotes(data.get("trade_date"))
        return jsonify({"success": True, "data": result})

    @app.route("/api/data/sync/daily_basic", methods=["POST"])
    def api_sync_daily_basic():
        data = request.get_json(silent=True) or {}
        result = DataSyncService.create().sync_daily_basic(data.get("trade_date"))
        return jsonify({"success": True, "data": result})

    @app.route("/api/stocks/search")
    def api_stock_search():
        keyword = request.args.get("q", "")
        result = DataSyncService.create().search_stocks(keyword)
        return jsonify({"success": True, "data": result})

    @app.route("/api/stocks/<path:ts_code>/profile")
    def api_stock_profile(ts_code: str):
        inference = v8_inference_service()
        result = StockProfileService(
            model_version=str(inference.artifact.get("model_version") or "") or None,
        ).get_profile(ts_code)
        if result is None:
            return jsonify({"success": False, "error": f"stock not found: {ts_code}"}), 404
        result["active_research"] = {
            "model_version": inference.artifact.get("model_version"),
            "research_gate": (inference.artifact.get("research_gate") or {}).get("status"),
            "market_gate": (inference.artifact.get("market_gate") or {}).get("research_status"),
            "diagnostic_only": bool(inference.artifact.get("diagnostic_only", True)),
        }
        return jsonify({"success": True, "data": result})

    @app.route("/api/market")
    def api_market():
        client = TushareClient()
        result = MarketAnalyzer(client).analyze(request.args.get("trade_date"))
        return jsonify({"success": True, "data": result})

    @app.route("/api/sectors")
    def api_sectors():
        client = TushareClient()
        top_n = int(request.args.get("top_n", "30"))
        result = SectorAnalyzer(client).analyze(request.args.get("trade_date"), top_n=top_n)
        return jsonify({"success": True, "data": result})

    @app.route("/api/selection/run", methods=["POST"])
    def api_run_selection():
        data = request.get_json(silent=True) or {}
        strategy_type = data.get("strategy_type") or "short_term_core"
        if strategy_type not in SUPPORTED_STRATEGY_TYPES:
            raise SelectionRuleError(f"Unsupported strategy_type: {strategy_type}")
        trade_date = data.get("trade_date")
        if trade_date is not None:
            trade_date = str(trade_date).strip()
            if not re.fullmatch(r"\d{8}", trade_date):
                raise ValueError("trade_date must use YYYYMMDD, for example 20260716")
        result = SelectionService.create().run_full_market_selection(strategy_type=strategy_type, end_date=trade_date)
        return jsonify({"success": True, "data": result})

    @app.route("/api/selection/runs")
    def api_selection_runs():
        limit = int(request.args.get("limit", "20"))
        result = SelectionService.create().list_runs(limit=limit)
        return jsonify({"success": True, "data": result})

    @app.route("/api/selection/runs/<int:run_id>")
    def api_selection_run_detail(run_id: int):
        result = SelectionService.create().get_run_results(run_id)
        return jsonify({"success": True, "data": result})

    @app.route("/api/selection/runs/<int:run_id>/export", methods=["POST"])
    def api_selection_run_export(run_id: int):
        data = request.get_json() or {}
        output_path = data.get("output_path") or f"reports/selection_run_{run_id}.csv"
        result = SelectionService.create().export_run_results_csv(run_id, output_path)
        return jsonify({"success": True, "data": result})

    @app.route("/api/positions", methods=["GET"])
    def api_positions():
        result = PortfolioService.create().evaluate_positions()
        return jsonify({"success": True, "data": result})

    @app.route("/api/positions/summary")
    def api_positions_summary():
        result = PortfolioService.create().portfolio_summary()
        return jsonify({"success": True, "data": result})

    @app.route("/api/positions", methods=["POST"])
    def api_add_position():
        data = request.get_json() or {}
        position_id = PortfolioService.create().add_position(
            ts_code=data["ts_code"],
            name=data["name"],
            strategy_type=data.get("strategy_type", "manual"),
            buy_price=float(data["buy_price"]),
            quantity=int(data["quantity"]),
            notes=data.get("notes"),
        )
        return jsonify({"success": True, "data": {"id": position_id}})

    @app.route("/api/positions/import_csv", methods=["POST"])
    def api_import_positions_csv():
        data = request.get_json() or {}
        result = PortfolioService.create().import_positions_csv(
            csv_path=data["csv_path"],
            clear_existing=bool(data.get("clear_existing", False)),
        )
        return jsonify({"success": True, "data": result})

    @app.route("/api/positions/update_prices", methods=["POST"])
    def api_update_prices():
        count = PortfolioService.create().update_prices()
        return jsonify({"success": True, "data": {"updated": count}})

    @app.route("/api/trades")
    def api_trade_logs():
        limit = int(request.args.get("limit", "100"))
        result = PortfolioService.create().trade_logs(limit=limit)
        return jsonify({"success": True, "data": result})

    @app.route("/api/reviews", methods=["GET"])
    def api_reviews():
        limit = int(request.args.get("limit", "50"))
        ts_code = request.args.get("ts_code")
        result = ReviewService().list_entries(limit=limit, ts_code=ts_code)
        return jsonify({"success": True, "data": result})

    @app.route("/api/reviews", methods=["POST"])
    def api_add_review():
        data = request.get_json() or {}
        entry_id = ReviewService().add_entry(
            title=data["title"],
            content=data["content"],
            ts_code=data.get("ts_code"),
            name=data.get("name"),
            category=data.get("category", "daily"),
            tags=data.get("tags") or [],
        )
        return jsonify({"success": True, "data": {"id": entry_id}})

    @app.route("/api/backtest/buy_hold", methods=["POST"])
    def api_backtest_buy_hold():
        data = request.get_json() or {}
        result = BacktestService.create().run_buy_and_hold_backtest(
            ts_code=data["ts_code"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_capital=float(data.get("initial_capital", 100000)),
        )
        return jsonify({"success": True, "data": result})

    @app.route("/api/ml/readiness")
    def api_ml_readiness():
        start_date = request.args.get("start_date", "20240101")
        end_date = request.args.get("end_date", "20260601")
        result = MlLabService().readiness(start_date=start_date, end_date=end_date)
        return jsonify({"success": True, "data": result})

    @app.route("/api/ml/top10")
    def api_ml_top10():
        trade_date = request.args.get("trade_date")
        if not trade_date:
            return jsonify({"success": False, "error": "trade_date is required"}), 400
        result = MlLabService().build_top10(trade_date)
        return jsonify({"success": True, "data": result})

    @app.route("/api/ml/model_report")
    def api_ml_model_report():
        candidates = [
            Path("reports/ml_lab/models_versioned_v2/model_comparison_report.json"),
            Path("reports/ml_lab/models_full/model_comparison_report.json"),
            Path("reports/ml_lab/models_versioned/model_comparison_report.json"),
        ]
        report_path = next((path for path in candidates if path.exists()), candidates[0])
        data = {
            "available": report_path.exists(),
            "path": str(report_path),
            "candidate_paths": [str(path) for path in candidates],
        }
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as file:
                data["report"] = json.load(file)

        harness_candidates = [
            Path("reports/ml_lab/research_harness_v5_cpu_full_lightgbm/research_harness_report.json"),
            Path("reports/ml_lab/research_harness_v4_cpu_full_lightgbm/research_harness_report.json"),
            Path("reports/ml_lab/research_harness_v3_cpu_full_lightgbm/research_harness_report.json"),
            Path("reports/ml_lab/research_harness_v2_full/research_harness_report.json"),
            Path("reports/ml_lab/research_harness_v1_full/research_harness_report.json"),
        ]
        harness_path = next((path for path in harness_candidates if path.exists()), harness_candidates[-1])
        if harness_path.exists():
            with harness_path.open("r", encoding="utf-8") as file:
                harness = json.load(file)
            data["latest_research_harness"] = {
                "available": True,
                "path": str(harness_path),
                "fold_count": harness.get("fold_count"),
                "feature_count": harness.get("feature_count"),
                "summary": harness.get("summary", []),
                "verdict_policy": harness.get("verdict_policy"),
            }
        else:
            data["latest_research_harness"] = {"available": False, "path": str(harness_path)}

        audit_candidates = [
            Path("reports/ml_lab/research_audit_v5_cpu_full_lightgbm/research_audit_report.json"),
            Path("reports/ml_lab/research_audit_v4_cpu_full_lightgbm/research_audit_report.json"),
            Path("reports/ml_lab/research_audit_v3_cpu_full_lightgbm/research_audit_report.json"),
            Path("reports/ml_lab/research_audit_v2_full/research_audit_report.json"),
            Path("reports/ml_lab/research_audit_20260708/research_audit_report.json"),
        ]
        audit_path = next((path for path in audit_candidates if path.exists()), audit_candidates[-1])
        if audit_path.exists():
            with audit_path.open("r", encoding="utf-8") as file:
                audit = json.load(file)
            data["latest_research_audit"] = {
                "available": True,
                "path": str(audit_path),
                "status": audit.get("status"),
                "gate": audit.get("gate"),
                "feature_set": audit.get("feature_set"),
                "diagnostic_only": audit.get("diagnostic_only"),
                "promotion_allowed": audit.get("promotion_allowed"),
            }
        else:
            data["latest_research_audit"] = {"available": False, "path": str(audit_path)}

        market_state_path = Path("reports/ml_lab/market_state_v6_cpu_full/market_state_report.json")
        if market_state_path.exists():
            with market_state_path.open("r", encoding="utf-8") as file:
                market_state = json.load(file)
            data["latest_market_state_research"] = {
                "available": True,
                "path": str(market_state_path),
                "gate": market_state.get("gate"),
                "summary": market_state.get("summary", []),
                "diagnostic_only": market_state.get("diagnostic_only"),
                "promotion_allowed": market_state.get("promotion_allowed"),
            }
        else:
            data["latest_market_state_research"] = {"available": False, "path": str(market_state_path)}

        short_research_path = Path("reports/ml_lab/short_execution_research_v1/short_execution_research_report.json")
        if short_research_path.exists():
            with short_research_path.open("r", encoding="utf-8") as file:
                short_research = json.load(file)
            data["latest_short_execution_research"] = {
                "available": True,
                "path": str(short_research_path),
                "execution_contract": short_research.get("execution_contract"),
                "feature_count": short_research.get("feature_count"),
                "heads": short_research.get("heads"),
                "split": short_research.get("split"),
                "final_audit": short_research.get("final_audit"),
                "baselines": short_research.get("final_audit_baselines"),
                "gate": short_research.get("gate"),
                "diagnostic_only": short_research.get("diagnostic_only"),
                "promotion_allowed": short_research.get("promotion_allowed"),
            }
            reproducibility_path = short_research_path.parent / "reproducibility_audit.json"
            if reproducibility_path.exists():
                with reproducibility_path.open("r", encoding="utf-8") as file:
                    data["latest_short_execution_research"]["reproducibility_audit"] = json.load(file)
        else:
            data["latest_short_execution_research"] = {"available": False, "path": str(short_research_path)}

        short_oos_path = Path("reports/ml_lab/short_execution_oos_audit_v1/short_execution_oos_audit.json")
        if short_oos_path.exists():
            with short_oos_path.open("r", encoding="utf-8") as file:
                short_oos = json.load(file)
            data["latest_short_execution_oos_audit"] = {
                "available": True,
                "path": str(short_oos_path),
                "existing_v11_top5": short_oos.get("existing_v11_top5"),
                "v10_market_gate": short_oos.get("v10_market_gate"),
                "conclusion": short_oos.get("conclusion"),
            }
        else:
            data["latest_short_execution_oos_audit"] = {"available": False, "path": str(short_oos_path)}

        window_study_path = Path("reports/ml_lab/short_execution_window_study_cpu_v3/short_execution_research_report.json")
        if window_study_path.exists():
            with window_study_path.open("r", encoding="utf-8") as file:
                window_study = json.load(file)
            data["recent_window_study"] = {
                "available": True,
                "path": str(window_study_path),
                "common_validation_starts": window_study.get("common_validation_starts"),
                "split": window_study.get("split"),
                "best_model": window_study.get("best_model"),
                "final_audit": window_study.get("final_audit"),
                "window_final_audits": window_study.get("window_final_audits"),
                "gate": window_study.get("gate"),
            }
        else:
            data["recent_window_study"] = {"available": False, "path": str(window_study_path)}

        failure_window_path = Path("reports/ml_lab/short_execution_failure_window_audit_v1/failure_window_audit.json")
        if failure_window_path.exists():
            with failure_window_path.open("r", encoding="utf-8") as file:
                failure_window = json.load(file)
            data["failure_window_audit"] = {
                "available": True,
                "path": str(failure_window_path),
                "oos_date_range": failure_window.get("oos_date_range"),
                "overall": failure_window.get("overall"),
                "baselines": failure_window.get("baselines"),
                "classifications": failure_window.get("classifications"),
                "market_state_data": failure_window.get("market_state_data"),
                "conclusion": failure_window.get("conclusion"),
            }
        else:
            data["failure_window_audit"] = {"available": False, "path": str(failure_window_path)}

        blind_path = Path("reports/ml_lab/short_execution_blind_20260602_20260709_v3/blind_report.json")
        if blind_path.exists():
            with blind_path.open("r", encoding="utf-8") as file:
                blind = json.load(file)
            data["short_execution_blind"] = {
                "available": True,
                "path": str(blind_path),
                "status": blind.get("status"),
                "split": blind.get("split"),
                "evaluation": blind.get("blind_evaluation"),
                "baselines": blind.get("baselines"),
                "feature_distribution_audit": blind.get("feature_distribution_audit"),
                "label_snapshot_audit": blind.get("label_snapshot_audit"),
                "promotion_allowed": blind.get("promotion_allowed"),
            }
        else:
            data["short_execution_blind"] = {"available": False, "path": str(blind_path)}
        return jsonify({"success": True, "data": data})

    @app.route("/api/ml/v8/status")
    def api_ml_v8_status():
        try:
            return jsonify({"success": True, "data": v8_inference_service().status()})
        except V8InferenceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/ml/v8/candidates")
    def api_ml_v8_candidates():
        signal_date = request.args.get("signal_date")
        if not signal_date:
            return jsonify({"success": False, "error": "signal_date is required"}), 400
        try:
            result = v8_inference_service().get_candidates(signal_date)
            return jsonify({"success": True, "data": result})
        except V8InferenceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/ml/v8/score", methods=["POST"])
    def api_ml_v8_score():
        data = request.get_json(silent=True) or {}
        trade_date = data.get("trade_date")
        if not trade_date:
            return jsonify({"success": False, "error": "trade_date is required"}), 400
        try:
            result = v8_inference_service().score_date(str(trade_date), persist=True)
            return jsonify({"success": True, "data": result})
        except (V8InferenceError, ValueError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/ml/v8/settle", methods=["POST"])
    def api_ml_v8_settle():
        data = request.get_json(silent=True) or {}
        as_of_date = data.get("as_of_date")
        if not as_of_date:
            return jsonify({"success": False, "error": "as_of_date is required"}), 400
        try:
            return jsonify({"success": True, "data": v8_inference_service().settle(str(as_of_date))})
        except V8InferenceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/ml/v8/manual-review", methods=["POST"])
    def api_ml_v8_manual_review():
        data = request.get_json(silent=True) or {}
        if "signal_id" not in data or "selected" not in data:
            return jsonify({"success": False, "error": "signal_id and selected are required"}), 400
        try:
            result = v8_inference_service().record_manual_review(
                int(data["signal_id"]),
                data["selected"] is True or str(data["selected"]).lower() == "true",
                reason=data.get("reason"),
                notes=data.get("notes"),
            )
            return jsonify({"success": True, "data": result})
        except V8InferenceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/factor/readiness")
    def api_factor_readiness():
        try:
            result = FactorLabBatchRunner.create().data_readiness()
            return jsonify({"success": True, "data": {**result, "status": "ready"}})
        except OperationalError as exc:
            return jsonify({"success": False, "error": f"database unavailable: {exc}"}), 503

    @app.route("/api/factor/evaluate", methods=["POST"])
    def api_factor_evaluate():
        data = request.get_json(silent=True) or {}
        factors = data.get("factors")
        horizons = data.get("horizons")
        output_dir = data.get("output_dir", "reports/factor_lab")
        runner = FactorLabBatchRunner.create()
        result = runner.evaluate_validation_suite(factors=factors, horizons=tuple(horizons) if horizons else None, output_dir=output_dir)
        return jsonify({"success": True, "data": result})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=settings.flask.host, port=settings.flask.port, debug=settings.flask.debug)
