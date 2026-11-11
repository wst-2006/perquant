from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.services.research_jobs import ResearchJobService  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(ResearchJobService.create().import_legacy_v8_runs(), ensure_ascii=False, indent=2))
