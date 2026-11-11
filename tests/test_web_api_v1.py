from web.app import create_app


def test_v1_health_and_history_are_json_contracts():
    client = create_app().test_client()
    for url in ("/api/v1/health", "/api/v1/research-runs?limit=10", "/api/v1/jobs?limit=10"):
        response = client.get(url)
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["request_id"]
        assert "data" in payload


def test_v1_invalid_date_has_structured_error_and_request_id():
    response = create_app().test_client().get("/api/v1/calendar/availability?trade_date=2026-07-16")
    payload = response.get_json()
    assert response.status_code == 400
    assert payload["success"] is False
    assert payload["error"]["code"] == "research_job_error"
    assert "YYYYMMDD" in payload["error"]["message"]
    assert payload["request_id"]


def test_v1_unknown_route_never_returns_html_500():
    response = create_app().test_client().get("/api/v1/not-a-real-route")
    payload = response.get_json()
    assert response.status_code == 404
    assert response.content_type.startswith("application/json")
    assert payload["success"] is False
    assert payload["error"]["code"] == "not_found"
    assert payload["request_id"]


def test_v1_calendar_separates_listed_and_historical_stock_counts():
    response = create_app().test_client().get("/api/v1/calendar/availability?trade_date=20260716")
    data = response.get_json()["data"]
    assert response.status_code == 200
    assert data["listed_stock_count"] <= data["historical_stock_count"]
    assert "active universe" in data["stock_count_policy"]
    assert data["component_counts"]["daily"] > 1000


def test_v1_research_metrics_include_no_signal_calendar_policy():
    response = create_app().test_client().get("/api/v1/research-metrics?horizon=5")
    data = response.get_json()["data"]
    assert response.status_code == 200
    assert data["version_policy"] == "latest_completed_version_per_trade_date"
    assert data["horizon"] == 5
    assert "all_loss_rate" in data
    assert "max_drawdown" in data
    assert "unsettled_signal_days" in data


def test_v1_research_metrics_reject_unknown_horizon():
    response = create_app().test_client().get("/api/v1/research-metrics?horizon=7")
    payload = response.get_json()
    assert response.status_code == 400
    assert payload["error"]["code"] == "research_job_error"


def test_stock_profile_contract_exposes_auditable_date_radar_and_missing_factors():
    response = create_app().test_client().get("/api/stocks/601899.SH/profile")
    data = response.get_json()["data"]
    assert response.status_code == 200
    assert data["stock"]["name"] == "紫金矿业"
    assert data["as_of_date"]
    assert len(data["scores"]["radar"]) == 6
    assert data["evidence"]["missing_factor_count"] == len(data["evidence"]["missing_factors"])
    assert data["active_research"]["diagnostic_only"] is True
