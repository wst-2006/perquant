from pathlib import Path

from web.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_endpoint():
    response = create_app().test_client().get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_factor_readiness_endpoint_available():
    response = create_app().test_client().get("/api/factor/readiness")
    assert response.status_code in (200, 400, 502, 503)


def test_index_promotes_v11_top5_daily_research():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "baseline_v0" not in body
    assert "v11 Top5 每日研究" in body
    assert "开始每日研究" in body
    assert 'id="dailyTradeDate"' in body
    assert 'id="dailyExecutionMode"' in body
    assert 'id="dailyJobPanel"' in body
    assert "系统 Top5" in body
    assert "默认 70/30 纸面组合" in body
    assert "我的人工选择" in body


def test_selection_rejects_unknown_strategy_type():
    response = create_app().test_client().post("/api/selection/run", json={"strategy_type": "mystery_mode"})
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_selection_rejects_malformed_trade_date_before_running_full_market_selection():
    response = create_app().test_client().post(
        "/api/selection/run",
        json={"strategy_type": "short_term_core", "trade_date": "2026-07-16"},
    )
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_model_report_exposes_latest_research_harness_and_audit():
    response = create_app().test_client().get("/api/ml/model_report")
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["path"].replace("\\", "/") == "reports/ml_lab/models_versioned_v2/model_comparison_report.json"
    assert "candidate_paths" in payload
    assert payload["latest_research_harness"]["path"].replace("\\", "/") == "reports/ml_lab/research_harness_v5_cpu_full_lightgbm/research_harness_report.json"
    assert payload["latest_research_audit"]["path"].replace("\\", "/") == "reports/ml_lab/research_audit_v5_cpu_full_lightgbm/research_audit_report.json"
    assert payload["latest_research_audit"]["status"] == "observe_only"
    assert payload["latest_market_state_research"]["available"] is True
    assert payload["latest_market_state_research"]["gate"]["status"] == "observe_only"
    assert payload["latest_short_execution_research"]["available"] is True
    assert payload["latest_short_execution_research"]["gate"]["status"] == "observe_only"
    assert payload["latest_short_execution_research"]["promotion_allowed"] is False
    assert payload["latest_short_execution_research"]["reproducibility_audit"]["status"] == "blocked"
    assert payload["latest_short_execution_oos_audit"]["available"] is True
    assert payload["latest_short_execution_oos_audit"]["conclusion"]["status"] == "observe_only"
    assert "failure_window_audit" in payload
    assert payload["failure_window_audit"]["market_state_data"]["status"] == "pass"
    assert payload["short_execution_blind"]["available"] is True
    assert payload["short_execution_blind"]["status"] == "observe_only"
    assert payload["short_execution_blind"]["feature_distribution_audit"]["status"] == "pass"


def test_v8_candidates_requires_signal_date():
    response = create_app().test_client().get("/api/ml/v8/candidates")
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_v8_score_requires_trade_date():
    response = create_app().test_client().post("/api/ml/v8/score", json={})
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_v8_status_reports_missing_artifact_without_fabricating_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = create_app().test_client().get("/api/ml/v8/status")
    assert response.status_code == 404
    assert response.get_json()["success"] is False


def test_portfolio_target_ratio_is_legacy_compatibility_only():
    from src.quant_trading.services.portfolio import PortfolioService

    assert PortfolioService._target_position_pct("hold", 0) == 1.0
    assert PortfolioService._target_position_pct("reduce", 0) == 0.5
    assert PortfolioService._target_position_pct("sell", 0) == 0.0
    assert callable(PortfolioService().portfolio_summary)


def test_index_is_capital_neutral():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "资金分配与买卖数量由人工决定" in body
    assert "持仓管理" not in body


def test_index_stock_search_has_profile_workspace_and_radar():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "输入 601899 / 601899.SH / 紫金矿业" in body
    assert 'id="stockProfile"' in body
    assert 'id="stockRadar"' in body
    assert 'id="stockQuality"' in body
    javascript = (PROJECT_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "scores.radar" in javascript
    assert "evidence.missing_factors" in javascript


def test_index_navigation_has_product_pages():
    body = create_app().test_client().get("/").get_data(as_text=True)
    for route in ("dashboard", "daily", "history", "stocks", "data", "models", "factors", "portfolio", "backtests", "reviews", "jobs"):
        assert f'data-route="{route}"' in body
        assert f'data-page="{route}"' in body


def test_frontend_recovers_active_daily_job_after_refresh():
    javascript = (PROJECT_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "recoverable=active.find(job=>job.job_type==='daily_research')" in javascript
    assert "window.setInterval(refreshJobs,2000)" in javascript
    assert "clearDailyResult" in javascript


def test_index_does_not_show_static_fake_sector_examples():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "半导体 <small>中芯国际" not in body
    assert "人工智能</span><b class=\"pos\">+2.35%" not in body
    assert "真实 Tushare 记录" in body


def test_index_uses_local_versioned_chart_and_icon_assets():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "static/vendor/echarts.min.js" in body
    assert "static/vendor/lucide.min.js" in body
    assert "cdn." not in body


def test_index_exposes_formal_history_and_result_views():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert 'id="historyMetrics"' in body
    assert 'id="historyEquityChart"' in body
    assert 'id="backtestChart"' in body
    assert 'id="reviewTimeline"' in body
    assert 'id="jobRows"' in body


def test_candidate_inspector_exposes_audited_d2_execution_without_fake_prediction():
    javascript = (PROJECT_ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")

    assert "D+2 独立预测头" in javascript
    assert "short_execution_prediction" in javascript
    assert "details?.short_execution" in javascript
    assert "不使用 5 日预测代替" in javascript
    assert "09:35 分钟成交价目前不可审计" in javascript


def test_index_exposes_formal_failure_window_audit_panel():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert 'id="failureWindowAudit"' in body
