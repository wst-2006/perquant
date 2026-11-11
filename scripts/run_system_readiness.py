from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.db import init_db
from src.quant_trading.factor_lab.batch import FactorLabBatchRunner
from src.quant_trading.ml_lab.service import MlLabService
from src.quant_trading.services.data_sync import DataSyncService
from sqlalchemy.exc import OperationalError


def main() -> None:
    readiness = {}
    try:
        init_db()
        readiness["db"] = {"status": "initialized"}
    except OperationalError as exc:
        readiness["db"] = {"status": "unavailable", "error": str(exc)}
        readiness["data_sync"] = {"error": "database unavailable"}
        readiness["factor_lab"] = {"error": "database unavailable"}
        readiness["ml_lab"] = {"error": "database unavailable"}
        print(json.dumps(readiness, ensure_ascii=False, indent=2, default=str))
        return

    readiness["data_sync"] = DataSyncService.create().data_status()
    readiness["factor_lab"] = FactorLabBatchRunner.create().data_readiness()
    readiness["ml_lab"] = MlLabService().readiness()
    print(json.dumps(readiness, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
