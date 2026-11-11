import numpy as np
import pandas as pd

from src.quant_trading.ml_lab.v8_service import V8InferenceService, _apply_calibration


def test_ranker_calibration_maps_scores_to_target_units():
    values = _apply_calibration(np.array([1.0, 2.0]), {"slope": 0.1, "intercept": -0.05})

    assert np.allclose(values, [0.05, 0.15])


def test_feature_drift_blocks_only_when_severe_features_exceed_ten_percent():
    service = object.__new__(V8InferenceService)
    service.artifact = {
        "training_distribution": {
            "feature_a": {"median": 0.0, "q25": -1.0, "q75": 1.0, "missing_rate": 0.0},
            "feature_b": {"median": 0.0, "q25": -1.0, "q75": 1.0, "missing_rate": 0.0},
        }
    }
    frame = pd.DataFrame({"feature_a": [10.0, 11.0], "feature_b": [0.0, 0.1]})

    result = service._feature_drift(frame, ["feature_a", "feature_b"])

    assert result["status"] == "blocked"
    assert result["severe_feature_count"] == 1


def test_feature_drift_blocks_on_single_critical_limit_feature():
    service = object.__new__(V8InferenceService)
    features = ["feature_limit_down_count_5d", *[f"feature_{index}" for index in range(19)]]
    service.artifact = {
        "training_distribution": {
            name: {"median": 0.0, "q25": -1.0, "q75": 1.0, "missing_rate": 0.0}
            for name in features
        }
    }
    frame = pd.DataFrame({name: [0.0, 0.1] for name in features})
    frame["feature_limit_down_count_5d"] = [10.0, 11.0]

    result = service._feature_drift(frame, features)

    assert result["severe_feature_rate"] == 0.05
    assert result["status"] == "blocked"
    assert result["critical_severe_features"] == ["feature_limit_down_count_5d"]


def test_feature_drift_allows_single_noncritical_shift_below_rate_threshold():
    service = object.__new__(V8InferenceService)
    features = [f"feature_{index}" for index in range(20)]
    service.artifact = {
        "training_distribution": {
            name: {"median": 0.0, "q25": -1.0, "q75": 1.0, "missing_rate": 0.0}
            for name in features
        }
    }
    frame = pd.DataFrame({name: [0.0, 0.1] for name in features})
    frame["feature_0"] = [10.0, 11.0]

    result = service._feature_drift(frame, features)

    assert result["severe_feature_rate"] == 0.05
    assert result["status"] == "pass"


def test_paper_eligibility_requires_candidate_market_and_drift_to_all_pass():
    checks = V8InferenceService._paper_eligibility(
        {"status": "candidate"},
        {"decision": "eligible"},
        {"status": "blocked"},
    )

    assert checks == {
        "research_gate_candidate": True,
        "market_gate_eligible": True,
        "data_drift_passed": False,
    }
    assert all(checks.values()) is False


def test_paper_eligibility_accepts_v10_stable_and_aggressive_states():
    stable = V8InferenceService._paper_eligibility({"status": "candidate"}, {"decision": "eligible_stable"}, {"status": "pass"})
    aggressive = V8InferenceService._paper_eligibility({"status": "candidate"}, {"decision": "eligible_aggressive"}, {"status": "pass"})
    diagnostic = V8InferenceService._paper_eligibility({"status": "candidate"}, {"decision": "diagnostic_eligible_stable"}, {"status": "pass"})

    assert all(stable.values())
    assert all(aggressive.values())
    assert diagnostic["market_gate_eligible"] is False


def test_signal_day_risk_tags_distinguish_limit_stress_from_large_decline():
    near_limit = V8InferenceService._signal_day_risk_tags(
        {"pct_chg": -8.5, "at_down_limit": False, "down_limit_room_used": 0.85}
    )
    at_limit = V8InferenceService._signal_day_risk_tags(
        {"pct_chg": -10.0, "at_down_limit": True, "down_limit_room_used": 1.0}
    )

    assert near_limit == ["信号日接近跌停", "信号日大幅下跌"]
    assert at_limit == ["信号日触及跌停", "信号日大幅下跌"]


def test_v12_paper_ledger_keeps_research_top5_separate_from_default_portfolio():
    candidates = [
        {"rank": 1, "ts_code": "000001.SZ", "candidate_type": "stable"},
        {"rank": 2, "ts_code": "000002.SZ", "candidate_type": "stable"},
        {"rank": 3, "ts_code": "000003.SZ", "candidate_type": "stable"},
        {"rank": 4, "ts_code": "000004.SZ", "candidate_type": "aggressive"},
        {"rank": 5, "ts_code": "000005.SZ", "candidate_type": "aggressive"},
    ]

    ledger = V8InferenceService._build_v12_paper_ledger(candidates, paper_eligible=True, market_gate={"decision": "eligible_stable"})

    assert len(ledger["research_top5"]) == 5
    assert [row["paper_weight"] for row in ledger["default_70_30_portfolio"]] == [0.233333, 0.233333, 0.233333, 0.15, 0.15]
    assert ledger["manual_selection"]["status"] == "awaiting_user_review"
    assert ledger["capital_policy"].startswith("percentage weights only")
