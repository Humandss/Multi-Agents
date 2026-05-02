"""WebSocket 서버 동작 테스트 (Unity 통합 전 검증).

사용:
    uv run python scripts/test_server.py                    # 5종 NPC 한 번씩
    uv run python scripts/test_server.py --char elias       # 1종 NPC
    uv run python scripts/test_server.py --interactive      # 대화형

서버가 먼저 떠 있어야 함:
    uv run python scripts/run_server.py
"""

import argparse
import asyncio
import json
import sys

import websockets

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
DEFAULT_PROMPTS = {
    "elias": "용 사냥꾼 얘기 들으셨어요?",
    "hermann": "검 한 자루 만들어주세요.",
    "mathilda": "마을에 무슨 일 있어요?",
    "finn": "어떤 영웅 이야기 알아요?",
    "bernhardt": "요즘 마을에 어떤 소문 돌아요?",
}


async def chat_once(host, port, npc, text):
    uri = f"ws://{host}:{port}/ws/{npc}"
    async with websockets.connect(uri) as ws:
        ready = json.loads(await ws.recv())
        assert ready.get("type") == "ready"
        await ws.send(json.dumps({"type": "chat", "text": text}))
        resp = json.loads(await ws.recv())
        return resp


async def chat_interactive(host, port, npc):
    uri = f"ws://{host}:{port}/ws/{npc}"
    async with websockets.connect(uri) as ws:
        ready = json.loads(await ws.recv())
        print(f"[연결] {ready}")
        print("종료: 'quit' 또는 Ctrl+C")
        while True:
            text = input(f"\n플레이어 > ").strip()
            if not text or text.lower() in {"quit", "exit"}:
                break
            await ws.send(json.dumps({"type": "chat", "text": text}))
            resp = json.loads(await ws.recv())
            if resp.get("type") == "response":
                print(f"\n[{npc}] {resp['text']}")
                print(f"  ({resp['latency_ms']}ms, 메모리 {len(resp['memories_used'])}개 활용)")
            else:
                print(f"  ERROR: {resp}")


def show(resp):
    print(f"  text: {resp.get('text', '')}")
    print(f"  latency: {resp.get('latency_ms', '?')}ms")
    mems = resp.get("memories_used", [])
    if mems:
        print(f"  memories ({len(mems)}):")
        for m in mems:
            print(f"    [{m.get('importance', '?'):>2}/{m.get('source', '')}] {m.get('text', '')[:80]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--char", choices=CHARACTERS, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        npc = args.char or "elias"
        await chat_interactive(args.host, args.port, npc)
        return

    targets = [args.char] if args.char else CHARACTERS
    for npc in targets:
        text = args.text or DEFAULT_PROMPTS[npc]
        print(f"\n=== {npc} ===")
        print(f"  q: {text}")
        try:
            resp = await chat_once(args.host, args.port, npc, text)
            show(resp)
        except Exception as e:
            print(f"  FAIL: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
