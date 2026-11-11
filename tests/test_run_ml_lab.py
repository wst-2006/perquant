from types import SimpleNamespace

from scripts import run_ml_lab
from src.quant_trading.ml_lab.research import WalkForwardConfig


def test_gate_factor_set_prefers_feature_set():
    args = SimpleNamespace(feature_set="factor_lab_v2", factor_set="factor_lab_v1", gate_factor_set=None)

    assert run_ml_lab.gate_factor_set_from_args(args) == "factor_lab_v2"


def test_preflight_report_includes_feature_missingness_gate(monkeypatch):
    def fake_preflight(training_frame_path, split_config=None):
        return {
            "universe_audit": {"status": "pass", "issues": []},
            "label_audit": {"status": "pass", "issues": []},
            "split_audit": {"status": "pass", "issues": []},
            "feature_missingness_gate": {"status": "pass", "blocking_features": []},
        }

    monkeypatch.setattr(run_ml_lab, "training_frame_preflight", fake_preflight)

    report = run_ml_lab.build_preflight_report(
        "reports/ml_lab/training_frame_v2_diagnostic_smoke.csv",
        {"gate_status": "pass", "diagnostic_only": True, "promotion_allowed": False},
        WalkForwardConfig(train_days=8, validation_days=4, step_days=4, top_n=(10, 20)),
    )

    assert report["training_frame_preflight"]["feature_missingness_gate"]["status"] == "pass"


def test_build_preflight_report_keeps_report_context(monkeypatch):
    def fake_preflight(training_frame_path, split_config=None):
        return {
            "universe_audit": {"status": "pass", "issues": []},
            "label_audit": {"status": "pass", "issues": []},
            "split_audit": {"status": "pass", "issues": []},
        }

    monkeypatch.setattr(run_ml_lab, "training_frame_preflight", fake_preflight)

    report = run_ml_lab.build_preflight_report(
        "reports/ml_lab/training_frame_v2_diagnostic_smoke.csv",
        {"gate_status": "pass", "diagnostic_only": True, "promotion_allowed": False},
        WalkForwardConfig(train_days=8, validation_days=4, step_days=4, top_n=(10, 20)),
        report_context={"feature_set": "factor_lab_v2", "report_contract_version": "ml_research_contract_v2"},
    )

    assert report["feature_set"] == "factor_lab_v2"
    assert report["report_contract_version"] == "ml_research_contract_v2"
    assert report["status"] == "pass"


def test_audit_research_parser_exposes_foundation_gate_args():
    parser = run_ml_lab.build_parser()
    args = parser.parse_args(
        [
            "audit-research",
            "--training-frame-path",
            "frame.csv",
            "--model-report-path",
            "model.json",
            "--feature-set",
            "factor_lab_v3",
        ]
    )

    assert args.gate_mode == "diagnostic"
    assert run_ml_lab.gate_factor_set_from_args(args) == "factor_lab_v3"


def test_market_state_parser_requires_research_artifacts():
    parser = run_ml_lab.build_parser()
    args = parser.parse_args(
        [
            "market-state-research",
            "--ranking-report-path",
            "ranking.json",
            "--baseline-audit-path",
            "audit.json",
        ]
    )

    assert args.feature_set == "factor_lab_v5"
    assert args.target_coverage == 0.25
    assert args.raw_state_factor_set == "factor_lab_v2"
    assert args.gate_mode == "diagnostic"


def test_paper_observation_parser_requires_signal_and_observation_dates():
    parser = run_ml_lab.build_parser()
    args = parser.parse_args(
        [
            "paper-observation",
            "--signal-date",
            "20260710",
            "--observation-date",
            "20260713",
        ]
    )

    assert args.factor_set == "factor_lab_v1"
    assert args.output_dir == "reports/ml_lab/paper_observation"
