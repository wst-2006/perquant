import zipfile

import pytest

from scripts import package_gpu_experiment
from src.quant_trading.data.tushare_client import FINA_INDICATOR_GATE_FIELDS
from src.quant_trading.services import foundation_gate
from src.quant_trading.services.foundation_gate import FoundationGateError, assert_diagnostic_foundation_ready, assert_foundation_ready


class _BlockedDataSyncService:
    def __init__(self, client=None):
        self.client = client

    def foundation_status(self, start_date: str, end_date: str) -> dict:
        return {
            "task": "foundation_status",
            "start_date": start_date,
            "end_date": end_date,
            "coverage": {
                "daily_quotes": 1.0,
                "daily_basic": 1.0,
                "moneyflow": 0.70,
                "index_daily": 1.0,
                "stock_limit": 0.08,
                "adj_factor": 0.09,
            },
            "blocking": ["stock_limit", "adj_factor"],
            "missing_financial_endpoints": ["income", "fina_mainbz_P"],
            "status": "blocked",
        }

    def stock_universe_audit(self, preview_limit: int = 20) -> dict:
        return {"status": "pass", "blocking": []}


class _PollutedStockDataSyncService(_BlockedDataSyncService):
    def foundation_status(self, start_date: str, end_date: str) -> dict:
        status = super().foundation_status(start_date, end_date)
        status["coverage"] = {key: 1.0 for key in status["coverage"]}
        status["blocking"] = []
        status["missing_financial_endpoints"] = []
        status["status"] = "pass"
        return status

    def stock_universe_audit(self, preview_limit: int = 20) -> dict:
        return {"status": "blocked", "blocking": ["unknown_stock_placeholders"]}


class _ProductionBlockedDiagnosticPassDataSyncService(_BlockedDataSyncService):
    def foundation_status(self, start_date: str, end_date: str) -> dict:
        return {
            "task": "foundation_status",
            "start_date": start_date,
            "end_date": end_date,
            "coverage": {
                "daily_quotes": 1.0,
                "daily_basic": 1.0,
                "moneyflow": 1.0,
                "index_daily": 1.0,
                "stock_limit": 1.0,
                "adj_factor": 1.0,
            },
            "raw_endpoint_rows": {
                "fina_indicator": 100,
                "fina_audit": 10,
                "fina_mainbz_P": 10,
                "fina_mainbz_D": 10,
                "fina_mainbz_I": 10,
            },
            "raw_endpoint_stock_coverage": {
                "fina_indicator": 0.96,
                "fina_audit": 0.1,
                "fina_mainbz_P": 0.1,
                "fina_mainbz_D": 0.1,
                "fina_mainbz_I": 0.1,
            },
            "min_raw_stock_coverage": 0.9,
            "financial_field_stock_coverage": {name: 1.0 for name in FINA_INDICATOR_GATE_FIELDS},
            "min_financial_field_stock_coverage": 0.9,
            "missing_financial_endpoints": [],
            "low_raw_stock_coverage": ["fina_audit", "fina_mainbz_P", "fina_mainbz_D", "fina_mainbz_I"],
            "blocking": ["financial_raw_records"],
            "status": "blocked",
        }


class _FinancialFieldBlockedDataSyncService(_ProductionBlockedDiagnosticPassDataSyncService):
    def foundation_status(self, start_date: str, end_date: str) -> dict:
        status = super().foundation_status(start_date, end_date)
        status["financial_field_stock_coverage"]["roa"] = 0.0
        return status


def test_foundation_gate_blocks_training_when_coverage_is_incomplete(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _BlockedDataSyncService)

    with pytest.raises(FoundationGateError) as exc:
        assert_foundation_ready()

    message = str(exc.value)
    assert "moneyflow_coverage_below_0.95" in message
    assert "financial_raw_records" in message


def test_foundation_gate_explicit_override_is_marked(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _BlockedDataSyncService)

    result = assert_foundation_ready(allow_blocked=True)

    assert result["gate_status"] == "blocked"
    assert result["override_used"] is True
    assert "must not be promoted" in result["override_warning"]


def test_foundation_gate_blocks_polluted_stock_universe(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _PollutedStockDataSyncService)

    with pytest.raises(FoundationGateError) as exc:
        assert_foundation_ready()

    assert "stock_universe_audit" in str(exc.value)


def test_diagnostic_gate_uses_current_feature_dependencies(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _ProductionBlockedDiagnosticPassDataSyncService)

    result = assert_diagnostic_foundation_ready(factor_set="factor_lab_v2")

    assert result["gate_status"] == "pass"
    assert result["diagnostic_only"] is True
    assert result["promotion_allowed"] is False
    assert result["production_gate_status"] == "blocked"
    assert result["data_dependency_profile"]["required_financial_raw"] == ["fina_indicator"]


def test_diagnostic_gate_blocks_when_required_financial_fields_are_empty(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _FinancialFieldBlockedDataSyncService)

    result = foundation_gate.inspect_diagnostic_foundation(factor_set="factor_lab_v2")

    assert result["gate_status"] == "blocked"
    assert "required_financial_field_coverage_below_threshold" in result["gate_blocking"]
    assert "roa" in result["data_dependency_profile"]["low_required_financial_fields"]


def test_diagnostic_gate_knows_v3_dependencies(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _ProductionBlockedDiagnosticPassDataSyncService)

    result = assert_diagnostic_foundation_ready(factor_set="factor_lab_v3")

    assert result["gate_status"] == "pass"
    assert result["data_dependency_profile"]["required_financial_raw"] == ["fina_indicator"]
    assert "Known 100%-missing v2 fields" in result["data_dependency_profile"]["policy"]


def test_diagnostic_gate_knows_v4_dependencies(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _ProductionBlockedDiagnosticPassDataSyncService)

    result = assert_diagnostic_foundation_ready(factor_set="factor_lab_v4")

    assert result["gate_status"] == "pass"
    assert result["data_dependency_profile"]["required_financial_raw"] == ["fina_indicator"]
    assert "derived interaction/meta factors" in result["data_dependency_profile"]["policy"]


def test_diagnostic_gate_knows_v5_cross_sectional_dependencies(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _ProductionBlockedDiagnosticPassDataSyncService)

    result = assert_diagnostic_foundation_ready(factor_set="factor_lab_v5")

    assert result["gate_status"] == "pass"
    assert result["data_dependency_profile"]["required_financial_raw"] == ["fina_indicator"]
    assert "full-market ranks" in result["data_dependency_profile"]["policy"]


def test_diagnostic_gate_knows_v7_raw_state_dependencies(monkeypatch):
    monkeypatch.setattr(foundation_gate, "DataSyncService", _ProductionBlockedDiagnosticPassDataSyncService)

    result = assert_diagnostic_foundation_ready(factor_set="factor_lab_v7")

    assert result["gate_status"] == "pass"
    assert "raw, signal-date Tushare state features" in result["data_dependency_profile"]["policy"]


def test_gpu_package_manifest_records_foundation_gate(tmp_path, monkeypatch):
    training_frame = tmp_path / "training_frame.csv"
    manifest = tmp_path / "experiment_manifest.json"
    output_path = tmp_path / "gpu_package.zip"
    training_frame.write_text(
        "trade_date,ts_code,label_tradable,label_net_return,label_stable_positive,label_top10,feature_x\n"
        "20250101,000001.SZ,1,0.01,1,1,0.5\n",
        encoding="utf-8",
    )
    manifest.write_text('{"kind":"experiment"}', encoding="utf-8")

    monkeypatch.setattr(
        package_gpu_experiment,
        "assert_foundation_ready",
        lambda *args, **kwargs: {
            "gate_status": "pass",
            "gate_blocking": [],
        },
    )

    package_manifest = package_gpu_experiment.build_package(
        output_path,
        training_frame_path=training_frame,
        manifest_path=manifest,
        require_foundation_ready=True,
    )

    assert package_manifest["foundation_gate_status"] == "pass"
    assert package_manifest["foundation_gate_blocking"] == []
    with zipfile.ZipFile(output_path) as archive:
        assert training_frame.name in set(archive.namelist())
