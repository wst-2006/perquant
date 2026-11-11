from src.quant_trading.ml_lab.paper_observation import build_mark_row, summarize_marks


def test_paper_mark_is_observable_but_not_a_settled_return():
    row = build_mark_row(
        {"list_type": "stable", "rank": 1, "ts_code": "000001.SZ", "name": "test", "stable_score": 80, "aggressive_score": 70},
        {"open": 10.0, "close": 10.5, "pct_chg": 5.0},
        {"up_limit": 11.0, "down_limit": 9.0},
    )

    assert row["entry_status"] == "paper_entry_observable"
    assert row["same_day_open_to_close_mark"] == 0.05
    assert row["settlement_status"] == "unsettled_t_plus_1"
    assert "action" not in row


def test_paper_mark_blocks_limit_up_open_and_excludes_it_from_summary():
    blocked = build_mark_row(
        {"list_type": "stable", "rank": 1, "ts_code": "000001.SZ", "name": "test", "stable_score": 80, "aggressive_score": 70},
        {"open": 11.0, "close": 11.0, "pct_chg": 10.0},
        {"up_limit": 11.0, "down_limit": 9.0},
    )
    observable = build_mark_row(
        {"list_type": "stable", "rank": 2, "ts_code": "000002.SZ", "name": "test2", "stable_score": 79, "aggressive_score": 69},
        {"open": 10.0, "close": 10.2, "pct_chg": 2.0},
        {"up_limit": 11.0, "down_limit": 9.0},
    )

    summary = summarize_marks([blocked, observable])

    assert blocked["entry_status"] == "blocked_limit_up_at_open"
    assert summary["same_day_mark_observation_count"] == 1
    assert summary["same_day_open_to_close_mark_mean"] == 0.02
    assert summary["settled_return_available"] is False
