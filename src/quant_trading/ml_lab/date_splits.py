from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DateWindow:
    name: str
    start_date: str
    end_date: str
    role: str
    rationale: str


REPRESENTATIVE_A_SHARE_20240401_20260601_480 = "representative_a_share_20240401_20260601_480"

SPLIT_PRESETS: dict[str, dict] = {
    REPRESENTATIVE_A_SHARE_20240401_20260601_480: {
        "name": REPRESENTATIVE_A_SHARE_20240401_20260601_480,
        "start_date": "20240401",
        "end_date": "20260601",
        "target_train_days": 480,
        "policy": "Use the full domestic A-share span, remove three out-of-sample windows, and train only on the remaining dates.",
        "holdout_windows": [
            DateWindow(
                name="post_2024_national_day_surge",
                start_date="20241008",
                end_date="20241025",
                role="backtest_only",
                rationale="High-volatility post-holiday surge and reversal window reserved for pure out-of-sample testing.",
            ),
            DateWindow(
                name="mid_2025_transition",
                start_date="20250603",
                end_date="20250623",
                role="backtest_only",
                rationale="Mid-2025 transition window reserved to test regime sensitivity after training.",
            ),
            DateWindow(
                name="early_2026_drawdown_transition",
                start_date="20260112",
                end_date="20260129",
                role="backtest_only",
                rationale="Early-2026 stress and transition window reserved for pure out-of-sample testing.",
            ),
        ],
    }
}


def split_trade_dates(trade_dates: list[str], preset_name: str) -> dict:
    if preset_name not in SPLIT_PRESETS:
        raise ValueError(f"Unknown split preset: {preset_name}")
    preset = SPLIT_PRESETS[preset_name]
    start_date = preset["start_date"]
    end_date = preset["end_date"]
    universe_dates = sorted(date for date in trade_dates if start_date <= date <= end_date)
    holdout_windows: list[DateWindow] = preset["holdout_windows"]
    holdout_dates = [
        date
        for date in universe_dates
        if any(window.start_date <= date <= window.end_date for window in holdout_windows)
    ]
    holdout_set = set(holdout_dates)
    label_blocked_dates = []
    for idx, date in enumerate(universe_dates):
        if date in holdout_set:
            continue
        future_label_dates = universe_dates[idx + 1 : idx + 3]
        if any(item in holdout_set for item in future_label_dates):
            label_blocked_dates.append(date)
    label_blocked_set = set(label_blocked_dates)
    train_dates = [date for date in universe_dates if date not in holdout_set and date not in label_blocked_set]
    return {
        "preset": preset_name,
        "policy": preset["policy"],
        "start_date": start_date,
        "end_date": end_date,
        "target_train_days": preset["target_train_days"],
        "total_trade_days": len(universe_dates),
        "train_trade_days": len(train_dates),
        "holdout_trade_days": len(holdout_dates),
        "train_dates": train_dates,
        "holdout_dates": holdout_dates,
        "label_blocked_train_dates": label_blocked_dates,
        "holdout_windows": [asdict(window) for window in holdout_windows],
        "leakage_policy": "Holdout signal dates and any train signal dates whose buy/sell labels touch holdout dates must not be exported into the GPU training frame or used for model selection.",
    }
