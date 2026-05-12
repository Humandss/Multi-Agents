"""NPC-NPC 자율 대화 demo (Phase 2 검증 스크립트).

사용:
    uv run python scripts/test_npc_conversation.py                              # mathilda <-> finn 기본 3턴
    uv run python scripts/test_npc_conversation.py --a hermann --b bernhardt    # 페어 지정
    uv run python scripts/test_npc_conversation.py --random                     # 그래프 무작위 페어
    uv run python scripts/test_npc_conversation.py --num_turns 4 --topic "린덴브뤽 광산"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


COLOR = {
    "elias": "\033[36m",      # cyan
    "hermann": "\033[91m",    # red
    "mathilda": "\033[93m",   # yellow
    "finn": "\033[95m",       # magenta
    "bernhardt": "\033[92m",  # green
}
RESET = "\033[0m"


def call_simulate(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_conversation(result: dict):
    if "error" in result:
        print(f"[error] {result['error']}")
        return
    a = result["npc_a"]
    b = result["npc_b"]
    topic = result.get("topic", "")
    day = result.get("day", 0)
    turns = result.get("turns", [])

    print(f"\n{'=' * 70}")
    print(f"[day {day}] {a} ↔ {b}")
    print(f"화제: {topic}")
    print('=' * 70)
    for t in turns:
        spk = t["speaker"]
        ko = t.get("speaker_ko", spk)
        c = COLOR.get(spk, "")
        print(f"  {c}{ko:<6}{RESET}: {t['text']}")
    print('=' * 70)
    if result.get("memory_saved"):
        print(f"→ {a}, {b} 양쪽 메모리에 저장됨\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--a", default="mathilda", help="NPC A (대화 시작자)")
    parser.add_argument("--b", default="finn", help="NPC B (응답자)")
    parser.add_argument("--num_turns", type=int, default=3, help="각 NPC 발화 횟수")
    parser.add_argument("--random", action="store_true", help="그래프 무작위 페어")
    parser.add_argument("--topic", default=None, help="(미지원) topic 시드 - 현재는 자동")
    args = parser.parse_args()

    if args.random:
        url = f"http://{args.host}:{args.port}/simulate_random?num_turns={args.num_turns}"
    else:
        url = (
            f"http://{args.host}:{args.port}/simulate/"
            f"{urllib.parse.quote(args.a)}/{urllib.parse.quote(args.b)}"
            f"?num_turns={args.num_turns}"
        )

    print(f"[simulate] {url}")
    print(f"[simulate] 대화 생성 중... (모델 추론 ≈ {args.num_turns * 2 * 5}초 예상)")

    try:
        result = call_simulate(url)
        print_conversation(result)
    except urllib.error.URLError as e:
        print(f"[error] 서버 연결 실패: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
