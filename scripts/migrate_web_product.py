from __future__ import annotations

import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "backup_before_migration.py")], cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
