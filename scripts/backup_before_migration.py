from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from sqlalchemy import inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_trading.db import get_engine  # noqa: E402


KEY_TABLES = (
    "positions",
    "trade_logs",
    "review_journals",
    "selection_runs",
    "selection_results",
    "ml_v8_candidate_signals",
    "ml_v8_candidate_outcomes",
    "ml_v8_manual_reviews",
)


def main() -> None:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output = PROJECT_ROOT / "reports" / "db_backups" / stamp
    output.mkdir(parents=True, exist_ok=False)
    engine = get_engine()
    inspector = inspect(engine)
    schema = {}
    for table in inspector.get_table_names():
        schema[table] = {
            "columns": [
                {"name": col["name"], "type": str(col["type"]), "nullable": col["nullable"]}
                for col in inspector.get_columns(table)
            ],
            "primary_key": inspector.get_pk_constraint(table),
            "unique_constraints": inspector.get_unique_constraints(table),
        }
    (output / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    manifest = {"created_at_utc": stamp, "tables": {}, "files": {}}
    with engine.connect() as conn:
        available = set(inspector.get_table_names())
        for table in KEY_TABLES:
            if table not in available:
                continue
            rows = [dict(row._mapping) for row in conn.execute(text(f'SELECT * FROM "{table}"'))]
            path = output / f"{table}.json"
            payload = json.dumps(rows, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            path.write_bytes(payload)
            manifest["tables"][table] = len(rows)
            manifest["files"][path.name] = hashlib.sha256(payload).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"backup_dir": str(output), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
