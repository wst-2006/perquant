from src.quant_trading.ml_lab.data_availability_audit import _availability_status, _model_membership


def test_availability_status_requires_predeclared_coverage_threshold():
    assert _availability_status(None) == "not_derivable"
    assert _availability_status(0.8999) == "historically_insufficient"
    assert _availability_status(0.90) == "historically_sufficient"


def test_model_membership_does_not_infer_feature_usage_from_source_presence():
    result = _model_membership({"feature_v9_top_list_count_20d"}, ("feature_roe", "feature_v9_top_list_count_20d"))

    assert result == {
        "currently_used_by_active_model": True,
        "used_feature_count": 1,
        "used_features": ["feature_v9_top_list_count_20d"],
    }
