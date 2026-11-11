import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.portfolio import PortfolioService


if __name__ == "__main__":
    service = PortfolioService.create()
    result = {
        "summary": service.portfolio_summary(),
        "positions": service.evaluate_positions(),
        "trades": service.trade_logs(limit=50),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
