from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_ascending(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "trade_date" not in df.columns:
        return df.copy()
    return df.sort_values("trade_date").reset_index(drop=True)


def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    dif = ema(close, 12) - ema(close, 26)
    dea = ema(dif, 9)
    hist = (dif - dea) * 2
    return dif, dea, hist


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def normalize_score(value: float, low: float, high: float, reverse: bool = False) -> float:
    if pd.isna(value):
        return 50.0
    if high == low:
        return 50.0
    score = (value - low) / (high - low) * 100
    score = max(0.0, min(100.0, score))
    return 100.0 - score if reverse else score


def technical_score(daily_df: pd.DataFrame) -> dict:
    df = ensure_ascending(daily_df)
    if df.empty or len(df) < 60:
        return {"score": 0.0, "valid": False, "reason": "insufficient_daily_data"}

    close = df["close"].astype(float)
    vol = df["vol"].astype(float) if "vol" in df.columns else pd.Series(dtype=float)

    ma5 = moving_average(close, 5)
    ma10 = moving_average(close, 10)
    ma20 = moving_average(close, 20)
    ma60 = moving_average(close, 60)
    dif, dea, hist = macd(close)
    rsi14 = rsi(close, 14)

    latest_close = float(close.iloc[-1])
    trend = 20.0
    if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
        trend = 100.0
    elif ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
        trend = 80.0
    elif ma5.iloc[-1] > ma10.iloc[-1]:
        trend = 60.0
    elif latest_close > ma20.iloc[-1]:
        trend = 50.0

    momentum = 50.0
    if dif.iloc[-1] > dea.iloc[-1] and hist.iloc[-1] > hist.iloc[-2]:
        momentum += 25.0
    elif dif.iloc[-1] < dea.iloc[-1]:
        momentum -= 20.0
    if 45 <= rsi14.iloc[-1] <= 70:
        momentum += 20.0
    elif rsi14.iloc[-1] > 80:
        momentum -= 15.0
    momentum = max(0.0, min(100.0, momentum))

    rolling_low = close.tail(120).min()
    rolling_high = close.tail(120).max()
    pos = (latest_close - rolling_low) / (rolling_high - rolling_low) if rolling_high > rolling_low else 0.5
    if 0.25 <= pos <= 0.75:
        position = 90.0
    elif pos < 0.25:
        position = 70.0
    elif pos <= 0.9:
        position = 55.0
    else:
        position = 35.0

    volume_score = 50.0
    if len(vol) >= 20:
        recent_vol = vol.tail(5).mean()
        base_vol = vol.tail(20).mean()
        if base_vol > 0:
            ratio = recent_vol / base_vol
            volume_score = normalize_score(ratio, 0.6, 1.8)

    score = trend * 0.35 + momentum * 0.30 + position * 0.20 + volume_score * 0.15
    return {
        "score": round(float(score), 4),
        "valid": True,
        "trend": round(float(trend), 4),
        "momentum": round(float(momentum), 4),
        "position": round(float(position), 4),
        "volume": round(float(volume_score), 4),
        "latest_close": latest_close,
        "ma20": None if pd.isna(ma20.iloc[-1]) else float(ma20.iloc[-1]),
        "rsi14": None if pd.isna(rsi14.iloc[-1]) else float(rsi14.iloc[-1]),
    }
