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
    parser.add_argument("--no-reset", action="store_true",
                        help="기존 ChromaDB 메모리 유지 (기본은 매번 reset + 시드 재적재)")
    args = parser.parse_args()

    # 기본 동작: 매번 reset + 시드 재적재. --no-reset 시 보존.
    import os
    if not args.no_reset:
        os.environ["NPC_RESET_ON_START"] = "1"
        print("[run_server] 시작 시 자동 reset (기본 동작). 메모리 유지하려면 --no-reset 사용.")
    else:
        print("[run_server] --no-reset — 기존 ChromaDB 메모리 유지 (시드 재적재 안 함)")

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
