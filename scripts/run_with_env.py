import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/run_with_env.py <command> [args...]")
    env = os.environ.copy()
    subprocess.run(sys.argv[1:], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
