from __future__ import annotations

import web.api_v1 as api_module
import web.app as app_module


class FakeResearch:
    def calendar_availability(self, trade_date=None):
        return {"trade_date": trade_date, "latest_ready_date": "20260716", "listed_stock_count": 5000}

    def list_jobs(self, **kwargs):
        return []

    def list_runs(self, **kwargs):
        return []

    def submit_daily_research(self, trade_date, **kwargs):
        return {"created": True, "job": {"id": "job-1"}, "run": {"id": "run-1"}}

    def get_run(self, run_id):
        return {"id": run_id, "candidates": []}

    def compare_runs(self, run_ids):
        return {"runs": [{"id": value} for value in run_ids]}

    def research_metrics(self, **kwargs):
        return {"horizon": kwargs["horizon"], "daily": []}

    def get_job(self, job_id):
        return {"id": job_id, "status": "queued"}

    def cancel_job(self, job_id):
        return {"id": job_id, "status": "cancelled"}

    def record_manual_selection(self, candidate_id, **kwargs):
        return {"candidate_id": candidate_id, **kwargs}

    def submit_factor_evaluation(self, horizons, **kwargs):
        return {"created": True, "job": {"id": "factor-1", "horizons": horizons}}

    def submit_data_sync(self, action, **kwargs):
        return {"created": True, "job": {"id": "data-1", "action": action}}


class FakePortfolio:
    def evaluate_positions(self):
        return []

    def portfolio_summary(self):
        return {"holding_count": 0}

    def add_position(self, **kwargs):
        return 42

    def import_positions_csv(self, **kwargs):
        return {"imported": 0, "failed": []}

    def update_prices(self):
        return 0

    def trade_logs(self, limit=100):
        return []


class FakeData:
    def data_status(self):
        return {"daily_quote_count": 1}

    def sync_stock_pool(self):
        return {"rows": 1}

    def sync_daily_quotes(self, trade_date):
        return {"trade_date": trade_date}

    def sync_daily_basic(self, trade_date):
        return {"trade_date": trade_date}

    def search_stocks(self, keyword):
        return [{"ts_code": "000001.SZ", "name": keyword}]


class FakeV8:
    artifact = {
        "model_version": "test-model",
        "research_gate": {"status": "observe_only"},
        "market_gate": {"research_status": "observe_only"},
        "diagnostic_only": True,
    }

    def status(self):
        return {"diagnostic_only": True}

    def get_candidates(self, signal_date):
        return {"signal_date": signal_date, "candidates": []}

    def score_date(self, trade_date, persist=True):
        return {"trade_date": trade_date, "persist": persist}

    def settle(self, as_of_date):
        return {"as_of_date": as_of_date}

    def record_manual_review(self, signal_id, selected, **kwargs):
        return {"signal_id": signal_id, "selected": selected}


def _provider(value):
    return type("Provider", (), {"create": staticmethod(lambda: value)})


def test_v1_route_surface_with_isolated_services(monkeypatch):
    research = FakeResearch()
    monkeypatch.setattr(api_module, "ResearchJobService", _provider(research))
    monkeypatch.setattr(api_module, "PortfolioService", _provider(FakePortfolio()))
    monkeypatch.setattr(api_module, "DataSyncService", _provider(FakeData()))
    monkeypatch.setattr(api_module, "V8InferenceService", lambda *args, **kwargs: FakeV8())
    monkeypatch.setattr(api_module, "BacktestService", _provider(type("B", (), {"run_buy_and_hold_backtest": lambda self, **kwargs: kwargs})()))
    monkeypatch.setattr(api_module, "ReviewService", lambda: type("R", (), {"list_entries": lambda self, **kwargs: [], "add_entry": lambda self, **kwargs: 9})())
    client = app_module.create_app().test_client()

    calls = [
        ("post", "/api/v1/research-runs", {"trade_date": "20260716", "execution_mode": "skip_sync"}, 202),
        ("get", "/api/v1/research-runs/run-1", None, 200),
        ("post", "/api/v1/research-runs/compare", {"run_ids": ["a", "b"]}, 200),
        ("get", "/api/v1/jobs/job-1", None, 200),
        ("post", "/api/v1/jobs/job-1/cancel", {}, 200),
        ("post", "/api/v1/manual-selections", {"candidate_id": 1, "selected": True}, 200),
        ("post", "/api/v1/factor-jobs", {"horizons": [1, 5]}, 202),
        ("post", "/api/v1/data-jobs", {"action": "daily_snapshot", "trade_date": "20260716"}, 202),
        ("get", "/api/v1/data/status?trade_date=20260716", None, 200),
        ("get", "/api/v1/data/availability-audit", None, 200),
        ("get", "/api/v1/dashboard", None, 200),
        ("get", "/api/v1/positions", None, 200),
        ("post", "/api/v1/positions", {"ts_code": "000001.SZ", "name": "平安银行", "buy_price": 10, "quantity": 100}, 201),
        ("get", "/api/v1/backtests", None, 200),
        ("post", "/api/v1/backtests", {"ts_code": "000001.SZ", "start_date": "20260101", "end_date": "20260201"}, 201),
        ("get", "/api/v1/reviews", None, 200),
        ("post", "/api/v1/reviews", {"title": "复盘", "content": "内容"}, 201),
    ]
    for method, url, payload, status in calls:
        response = getattr(client, method)(url, json=payload) if payload is not None else getattr(client, method)(url)
        assert response.status_code == status, (method, url, response.get_json())
        assert response.get_json()["success"] is True

    assert client.post("/api/v1/manual-selections", json={}).status_code == 400
    assert client.post("/api/v1/positions", json={}).status_code == 400
    assert client.post("/api/v1/reviews", json={}).status_code == 400
    assert client.delete("/api/v1/positions/987654321").status_code == 404
    assert client.put("/api/v1/reviews/987654321", json={"title": "x"}).status_code == 404
    assert client.delete("/api/v1/reviews/987654321").status_code == 404


def test_legacy_compatibility_routes_with_isolated_services(monkeypatch):
    data = FakeData()
    portfolio = FakePortfolio()
    v8 = FakeV8()
    monkeypatch.setattr(app_module, "init_db", lambda: None)
    monkeypatch.setattr(app_module, "DataSyncService", _provider(data))
    monkeypatch.setattr(app_module, "PortfolioService", _provider(portfolio))
    monkeypatch.setattr(app_module, "v8_inference_service", lambda: v8)
    monkeypatch.setattr(app_module, "StockProfileService", lambda **kwargs: type("S", (), {"get_profile": lambda self, key: {"stock": {"ts_code": key}}})())
    monkeypatch.setattr(app_module, "TushareClient", lambda: object())
    monkeypatch.setattr(app_module, "MarketAnalyzer", lambda client: type("M", (), {"analyze": lambda self, date: {"date": date}})())
    monkeypatch.setattr(app_module, "SectorAnalyzer", lambda client: type("S", (), {"analyze": lambda self, date, top_n: {"top_n": top_n}})())
    selection = type("Selection", (), {
        "run_full_market_selection": lambda self, **kwargs: kwargs,
        "list_runs": lambda self, **kwargs: [],
        "get_run_results": lambda self, run_id: {"id": run_id},
        "export_run_results_csv": lambda self, run_id, path: {"id": run_id, "path": path},
    })()
    monkeypatch.setattr(app_module, "SelectionService", _provider(selection))
    monkeypatch.setattr(app_module, "ReviewService", lambda: type("R", (), {"list_entries": lambda self, **kwargs: [], "add_entry": lambda self, **kwargs: 1})())
    monkeypatch.setattr(app_module, "BacktestService", _provider(type("B", (), {"run_buy_and_hold_backtest": lambda self, **kwargs: kwargs})()))
    monkeypatch.setattr(app_module, "MlLabService", lambda: type("ML", (), {"readiness": lambda self, **kwargs: kwargs, "build_top10": lambda self, date: {"trade_date": date}})())
    monkeypatch.setattr(app_module, "FactorLabBatchRunner", _provider(type("F", (), {"data_readiness": lambda self: {"rows": 1}, "evaluate_validation_suite": lambda self, **kwargs: kwargs})()))
    client = app_module.create_app().test_client()

    calls = [
        ("get", "/api/config", None), ("post", "/api/init_db", {}), ("get", "/api/data/status", None),
        ("post", "/api/data/sync/stock_pool", {}), ("post", "/api/data/sync/daily", {"trade_date": "20260716"}),
        ("post", "/api/data/sync/daily_basic", {"trade_date": "20260716"}), ("get", "/api/stocks/search?q=平安", None),
        ("get", "/api/stocks/000001.SZ/profile", None), ("get", "/api/market?trade_date=20260716", None),
        ("get", "/api/sectors?top_n=5", None), ("post", "/api/selection/run", {"strategy_type": "short_term_core", "trade_date": "20260716"}),
        ("get", "/api/selection/runs", None), ("get", "/api/selection/runs/1", None),
        ("post", "/api/selection/runs/1/export", {"output_path": "reports/test.csv"}),
        ("get", "/api/positions", None), ("get", "/api/positions/summary", None),
        ("post", "/api/positions", {"ts_code": "000001.SZ", "name": "平安银行", "buy_price": 10, "quantity": 100}),
        ("post", "/api/positions/import_csv", {"csv_path": "x.csv"}), ("post", "/api/positions/update_prices", {}),
        ("get", "/api/trades", None), ("get", "/api/reviews", None),
        ("post", "/api/reviews", {"title": "t", "content": "c"}),
        ("post", "/api/backtest/buy_hold", {"ts_code": "000001.SZ", "start_date": "20260101", "end_date": "20260201"}),
        ("get", "/api/ml/readiness", None), ("get", "/api/ml/top10?trade_date=20260716", None),
        ("get", "/api/ml/v8/status", None), ("get", "/api/ml/v8/candidates?signal_date=20260716", None),
        ("post", "/api/ml/v8/score", {"trade_date": "20260716"}), ("post", "/api/ml/v8/settle", {"as_of_date": "20260716"}),
        ("post", "/api/ml/v8/manual-review", {"signal_id": 1, "selected": True}),
        ("get", "/api/factor/readiness", None), ("post", "/api/factor/evaluate", {"horizons": [5]}),
    ]
    for method, url, payload in calls:
        response = getattr(client, method)(url, json=payload) if payload is not None else getattr(client, method)(url)
        assert response.status_code < 400, (method, url, response.get_json())
        assert response.get_json()["success"] is True

    assert client.get("/api/ml/top10").status_code == 400
    assert client.post("/api/ml/v8/settle", json={}).status_code == 400
    assert client.post("/api/ml/v8/manual-review", json={}).status_code == 400
