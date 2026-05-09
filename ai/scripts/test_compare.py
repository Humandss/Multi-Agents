"""페르소나 비교 demo 테스트 — /compare endpoint 동작 확인.

같은 텍스트를 5종 NPC에 보내고 각자 응답 비교.

사용:
    uv run python scripts/test_compare.py                          # 기본 인사 테스트
    uv run python scripts/test_compare.py --text "마을에 무슨 일?"
    uv run python scripts/test_compare.py --interactive            # 대화형
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Windows cp949 콘솔 한국어 외 문자 print 실패 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def call_compare(url: str, text: str) -> dict:
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_responses(result: dict):
    print(f"\n{'=' * 70}")
    print(f"플레이어: {result['input']}")
    print('=' * 70)
    for r in result["responses"]:
        npc = r["npc"]
        text = r.get("text", "").strip()
        latency = r.get("latency_ms", 0)
        if "error" in r:
            print(f"  ❌ {npc:>10}: [error] {r['error']}")
        else:
            print(f"  {npc:>10} ({latency:>5}ms): {text}")
            # 회상된 메모리 표시 (use_memory=True 때만)
            mems = r.get("memories_used", [])
            for m in mems:
                src = m.get("source", "?")
                imp = m.get("importance", "?")
                mtxt = m.get("text", "")[:80]
                print(f"           ↳ [{src} imp={imp}] {mtxt}")
            # quest 표시
            quest = r.get("quest")
            if quest:
                title = quest.get("title", "")
                desc = quest.get("description", "")
                reward = quest.get("reward", "")
                print(f"           ★ QUEST: {title}")
                print(f"             - {desc}")
                if reward:
                    print(f"             - 보상: {reward}")
    print('=' * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--text", default="안녕하세요!")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/compare"

    if args.interactive:
        print(f"[compare] 대화형 모드. 텍스트 입력 (Ctrl+C로 종료)")
        print(f"[compare] URL: {url}\n")
        try:
            while True:
                text = input("플레이어> ").strip()
                if not text:
                    continue
                try:
                    result = call_compare(url, text)
                    print_responses(result)
                except urllib.error.URLError as e:
                    print(f"[error] 서버 연결 실패: {e}")
                    print(f"        서버가 켜져 있는지 확인: python -m uv run python scripts/run_server.py")
                    return 1
        except KeyboardInterrupt:
            print("\n종료")
    else:
        try:
            result = call_compare(url, args.text)
            print_responses(result)
        except urllib.error.URLError as e:
            print(f"[error] 서버 연결 실패: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
