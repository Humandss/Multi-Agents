"""시간 진행(정보 전파 tick) WebSocket 동작 확인."""

import argparse
import asyncio
import json

import websockets


async def run(host, port):
    # mathilda에 새 정보 주입 → 시간 진행 → elias가 알게 됐는지 확인
    uri_mat = f"ws://{host}:{port}/ws/mathilda"
    async with websockets.connect(uri_mat) as ws:
        await ws.recv()  # ready

        print("[1] mathilda에게 새 사건 알림 (플레이어 발화 → DIALOGUE 메모리)")
        await ws.send(json.dumps({
            "type": "chat",
            "text": "이모! 광장에서 큰 곰이 나타났대요!"
        }))
        resp = json.loads(await ws.recv())
        print(f"   mathilda: {resp['text']}\n")

        print("[2] time_advance — 하루 진행")
        await ws.send(json.dumps({"type": "time_advance"}))
        tick = json.loads(await ws.recv())
        print(f"   day={tick['day']}, 전달 {len(tick['events'])}개")
        for ev in tick["events"][:5]:
            arrow = f"{ev['from']:>10} → {ev['to']:<10}"
            print(f"     {arrow} (imp {ev['importance_before']}→{ev['importance_after']})")
            print(f"       원본: {ev['original'][:60]}")
            print(f"       변형: {ev['transformed'][:60]}")
        print(f"   메모리 수: {tick['memory_counts']}\n")

    # 다른 NPC (elias)가 그 정보를 회상하는지 확인
    print("[3] elias에게 곰 얘기 물어보기")
    uri_eli = f"ws://{host}:{port}/ws/elias"
    async with websockets.connect(uri_eli) as ws:
        await ws.recv()  # ready
        await ws.send(json.dumps({"type": "chat", "text": "마을에 곰 나타났다는 소문 들으셨어요?"}))
        resp = json.loads(await ws.recv())
        print(f"   elias: {resp['text']}")
        if resp.get("memories_used"):
            print(f"   회상 {len(resp['memories_used'])}개:")
            for m in resp["memories_used"]:
                print(f"     [{m['source']}] {m['text'][:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port))


if __name__ == "__main__":
    main()
