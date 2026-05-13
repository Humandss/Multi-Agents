"""캐주얼 대화 흐름 검증 — 인사 → 잡담 → 본론."""

import argparse
import asyncio
import json

import websockets


SCENARIOS = {
    "hermann": [
        "안녕하세요 아저씨",
        "아녀 ㅋㅋ 아저씨 보러 왔죠",
        "오늘 어떠세요?",
        "검 한 자루 사고 싶은데요",
        "용 잡으러 갈 거예요",
    ],
    "mathilda": [
        "안녕하세요!",
        "여기 분위기 좋네요",
        "마을에 무슨 재미있는 일 있어요?",
        "사과 파이 하나 주세요",
    ],
    "elias": [
        "안녕하세요 마법사님",
        "방해해서 죄송해요",
        "마법에 관심 있어요",
        "용에 대해 알려주세요",
    ],
}


async def run(host, port, npc, turns):
    uri = f"ws://{host}:{port}/ws/{npc}"
    async with websockets.connect(uri) as ws:
        ready = json.loads(await ws.recv())
        print(f"[{npc}] 연결\n")

        for i, text in enumerate(turns, start=1):
            await ws.send(json.dumps({"type": "chat", "text": text}))
            resp = json.loads(await ws.recv())
            print(f"[{i}] 플레이어: {text}")
            print(f"    {npc}: {resp['text']}")
            mems = resp.get("memories_used", [])
            if mems:
                print(f"    (회상 {len(mems)}개)")
            print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--char", default="hermann", choices=list(SCENARIOS.keys()))
    args = parser.parse_args()

    await run(args.host, args.port, args.char, SCENARIOS[args.char])


if __name__ == "__main__":
    asyncio.run(main())
