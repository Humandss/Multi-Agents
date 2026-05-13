"""multi-turn 대화 흐름 테스트 — 같은 세션에서 연속 대화."""

import argparse
import asyncio
import json

import websockets


async def run(host, port, npc, turns):
    uri = f"ws://{host}:{port}/ws/{npc}"
    async with websockets.connect(uri) as ws:
        ready = json.loads(await ws.recv())
        assert ready.get("type") == "ready"
        print(f"[연결] {npc}\n")

        for i, text in enumerate(turns, start=1):
            await ws.send(json.dumps({"type": "chat", "text": text}))
            resp = json.loads(await ws.recv())
            print(f"[{i}] 플레이어: {text}")
            print(f"    {npc}: {resp['text']}")
            mems = resp.get("memories_used", [])
            if mems:
                print(f"    (회상 {len(mems)}개)")
            print(f"    ({resp['latency_ms']}ms)")
            print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--char", default="hermann")
    args = parser.parse_args()

    # hermann 시나리오: 인사 → 잡담 → 본론(검)
    turns = [
        "안녕하세요",
        "잘 지내셨어요?",
        "오늘 날씨 좋네요.",
        "검 한 자루 사고 싶은데요.",
        "용 잡으러 갈 거예요.",
    ]
    await run(args.host, args.port, args.char, turns)


if __name__ == "__main__":
    asyncio.run(main())
