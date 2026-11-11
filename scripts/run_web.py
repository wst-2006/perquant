from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waitress import serve

from web.app import app
from src.quant_trading.config import settings


if __name__ == "__main__":
    serve(app, host=settings.flask.host, port=settings.flask.port, threads=12)
