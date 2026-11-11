import pandas as pd
import pytest

from src.quant_trading.ml_lab import v8_features
from src.quant_trading.ml_lab.v8_features import V8_LIVE_FEATURE_CONTRACT, build_live_v8_feature_frame, derive_v5_features


def test_v8_live_feature_contract_is_versioned():
    assert V8_LIVE_FEATURE_CONTRACT == "ml_v8_v5_feature_derivation_v2"


def test_v8_live_derivation_matches_v5_cross_section_contract():
    frame = pd.DataFrame(
        [
            {"trade_date": "20260101", "feature_return_5d": 0.1, "feature_return_20d": 0.2, "feature_amount_avg_5d": 20, "feature_amount_avg_20d": 10},
            {"trade_date": "20260101", "feature_return_5d": -0.1, "feature_return_20d": 0.0, "feature_amount_avg_5d": 10, "feature_amount_avg_20d": 10},
        ]
    )
    requested = [
        "feature_short_vs_medium_return",
        "feature_amount_trend_5d_20d",
        "feature_cs_rank_return_5d",
        "feature_market_breadth_return_5d",
    ]

    result = derive_v5_features(frame, requested)

    assert result["feature_short_vs_medium_return"].tolist() == [-0.1, -0.1]
    assert result["feature_amount_trend_5d_20d"].tolist() == [1.0, 0.0]
    assert result["feature_cs_rank_return_5d"].tolist() == [1.0, 0.5]
    assert result["feature_market_breadth_return_5d"].tolist() == [0.5, 0.5]


def test_v8_live_derivation_canonicalizes_precomputed_features():
    frame = pd.DataFrame(
        [
            {
                "trade_date": "20260101",
                "feature_main_net_amount_sum_5d": 10.0,
                "feature_amount": 20.0,
                "feature_volatility_20d": 2.0,
                "feature_main_net_to_amount_sum_5d": None,
                "feature_flow_risk_adjusted_5d": None,
                "feature_cs_rank_main_net_to_amount_sum_5d": None,
            },
            {
                "trade_date": "20260101",
                "feature_main_net_amount_sum_5d": 6.0,
                "feature_amount": 10.0,
                "feature_volatility_20d": 2.0,
                "feature_main_net_to_amount_sum_5d": 50.0,
                "feature_flow_risk_adjusted_5d": 25.0,
                "feature_cs_rank_main_net_to_amount_sum_5d": None,
            },
        ]
    )
    requested = [
        "feature_main_net_to_amount_sum_5d",
        "feature_flow_risk_adjusted_5d",
        "feature_cs_rank_main_net_to_amount_sum_5d",
    ]

    result = derive_v5_features(frame, requested)

    assert result["feature_main_net_to_amount_sum_5d"].tolist() == pytest.approx([0.5, 0.6])
    assert result["feature_flow_risk_adjusted_5d"].tolist() == pytest.approx([0.25, 0.3])
    assert result["feature_cs_rank_main_net_to_amount_sum_5d"].tolist() == pytest.approx([0.5, 1.0])


def test_live_feature_frame_reuses_injected_v9_event_store(monkeypatch):
    class FakeService:
        def __init__(self, *_args, **_kwargs):
            pass

        def _score_frame(self, _trade_date):
            return pd.DataFrame([{"ts_code": "000001.SZ"}])

        def _rank_scores(self, _source):
            return pd.DataFrame(
                [{"ts_code": "000001.SZ", "name": "A", "industry": "I", "stable_score": 0.1, "aggressive_score": 0.2, "stable_rank": 1, "aggressive_rank": 1, "features": {"return_5d": 0.03}, "raw_features": {}, "data_quality": {}}]
            )

        def _risk_tags(self, *_args):
            return []

    class FakeV9Store:
        def __init__(self):
            self.calls = 0

        def features_for_rows(self, rows):
            self.calls += 1
            return pd.DataFrame({"feature_v9_top_list_count_20d": [2.0] * len(rows)}, index=rows.index)

    store = FakeV9Store()
    monkeypatch.setattr(v8_features, "MlLabService", FakeService)
    result = build_live_v8_feature_frame(
        "20260602",
        ["feature_return_5d", "feature_v9_top_list_count_20d"],
        v9_event_store=store,
    )

    assert store.calls == 1
    assert result["feature_v9_top_list_count_20d"].tolist() == [2.0]
