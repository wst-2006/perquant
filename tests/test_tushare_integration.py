import os

import pytest

from src.quant_trading.data.tushare_client import TushareClient, TushareDataError


TOKEN = os.getenv("TUSHARE_TOKEN")
PLACEHOLDER_TOKENS = {"replace_with_your_tushare_token", "your_token", "your_tushare_token"}

pytestmark = pytest.mark.skipif(
    not TOKEN or TOKEN in PLACEHOLDER_TOKENS,
    reason="A real TUSHARE_TOKEN is required for integration tests",
)


def test_real_full_market_stock_pool_not_empty():
    client = TushareClient()
    try:
        df = client.get_all_a_shares(with_filter=True)
    except TushareDataError as exc:
        if "token" in str(exc).lower() or "token" in str(exc):
            pytest.skip(f"Tushare token is not accepted by server: {exc}")
        raise
    assert not df.empty
    assert "ts_code" in df.columns
    assert len(df) > 1000
