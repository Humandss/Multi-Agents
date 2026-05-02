"""WebSocket 추론 서버 실행.

uv run python scripts/run_server.py
uv run python scripts/run_server.py --port 8000 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "src.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(ROOT),
        log_level="info",
    )


if __name__ == "__main__":
    main()
