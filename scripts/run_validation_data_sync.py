import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.data_sync import DataSyncService


TASKS = [
    ("stock-universe", "20250201", "20260618", None),
    ("calendar-batch", "20250201", "20260618", ["daily", "daily_basic"]),
    ("moneyflow-window-1", "20250501", "20250731", ["moneyflow"]),
    ("moneyflow-window-2", "20251201", "20260227", ["moneyflow"]),
    ("moneyflow-window-3", "20260201", "20260520", ["moneyflow"]),
]


def main() -> None:
    service = DataSyncService.create()
    results = []
    for name, start_date, end_date, datasets in TASKS:
        print(f"[validation_sync] start {name} {start_date}-{end_date}", flush=True)
        if name == "stock-universe":
            result = service.sync_stock_universe()
        else:
            result = service.sync_calendar_batch(start_date=start_date, end_date=end_date, datasets=datasets)
        results.append({"name": name, "result": result})
        print(f"[validation_sync] done {name}: {json.dumps(result, ensure_ascii=False, default=str)}", flush=True)
    print(json.dumps({"task": "validation_data_sync", "results": results}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
