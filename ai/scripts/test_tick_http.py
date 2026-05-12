"""HTTP /tick 엔드포인트 검증: propagation + NPC-NPC 자율 대화 통합.

사용:
    uv run python scripts/test_tick_http.py                              # 기본 (전파 + 대화 1쌍)
    uv run python scripts/test_tick_http.py --no_conversation            # propagation만
    uv run python scripts/test_tick_http.py --num_turns 3
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


COLOR = {
    "elias": "\033[36m",
    "hermann": "\033[91m",
    "mathilda": "\033[93m",
    "finn": "\033[95m",
    "bernhardt": "\033[92m",
}
RESET = "\033[0m"


def call_tick(url: str) -> dict:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_tick(result: dict):
    if "error" in result:
        print(f"[error] {result['error']}")
        return
    day = result.get("day", 0)
    events = result.get("events", [])
    # 평탄화된 형식: turns + npc_a/npc_b 직접 들어옴 (WebSocket과 동일)
    turns = result.get("turns") or []
    npc_a = result.get("npc_a", "")
    npc_b = result.get("npc_b", "")
    topic = result.get("topic", "")
    convo = {
        "npc_a": npc_a, "npc_b": npc_b, "topic": topic, "turns": turns,
    } if turns else None
    counts = result.get("memory_counts_dict") or result.get("memory_counts", {})

    print(f"\n{'#' * 70}")
    print(f"[ day {day} ]")
    print('#' * 70)

    # 1. propagation events
    print(f"\n[전파] {len(events)}개 이벤트")
    for ev in events:
        fr = ev.get("from", "?")
        to = ev.get("to", "?")
        imp_b = ev.get("importance_before", 0)
        imp_a = ev.get("importance_after", 0)
        orig = ev.get("original", "")[:60]
        trans = ev.get("transformed", "")[:60]
        c_from = COLOR.get(fr, "")
        c_to = COLOR.get(to, "")
        print(f"  {c_from}{fr}{RESET} → {c_to}{to}{RESET}  imp {imp_b}→{imp_a}")
        print(f"    원: {orig}")
        print(f"    변: {trans}")

    # 2. NPC-NPC conversation
    if convo:
        a = convo.get("npc_a", "?")
        b = convo.get("npc_b", "?")
        topic = convo.get("topic", "")
        turns = convo.get("turns", [])
        print(f"\n[자율대화] {a} ↔ {b}")
        print(f"  화제: {topic}")
        for t in turns:
            spk = t.get("speaker", "?")
            ko = t.get("speaker_ko", spk)
            c = COLOR.get(spk, "")
            print(f"  {c}{ko:<6}{RESET}: {t.get('text', '')}")

    # 3. memory counts
    if counts:
        print(f"\n[메모리 카운트]")
        for npc, n in counts.items():
            print(f"  {npc:>10}: {n}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument("--no_conversation", action="store_true")
    args = parser.parse_args()

    params = []
    if args.no_conversation:
        params.append("npc_conversation=false")
    params.append(f"num_turns={args.num_turns}")
    url = f"http://{args.host}:{args.port}/tick?{'&'.join(params)}"

    print(f"[tick] {url}")
    print(f"[tick] 시뮬레이션 실행 중 (≈ {args.num_turns * 2 * 5}초)...")

    try:
        result = call_tick(url)
        print_tick(result)
    except urllib.error.URLError as e:
        print(f"[error] {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
