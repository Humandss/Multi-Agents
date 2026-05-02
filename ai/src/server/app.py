"""FastAPI WebSocket 서버.

엔드포인트:
  GET  /healthz           서버 상태 + 로딩된 NPC 목록
  GET  /npcs              NPC별 메모리 수 등 메타데이터
  WS   /ws/{npc_name}     NPC와 대화

WebSocket 프로토콜:
  Client -> Server (JSON):
    {"type": "chat", "text": "..."}
  Server -> Client (JSON):
    {"type": "response", "npc": "...", "text": "...",
     "memories_used": [{text, importance, source}, ...],
     "latency_ms": int}
    {"type": "error", "message": "..."}
"""

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .engine import NpcServer

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"


def create_app() -> FastAPI:
    app = FastAPI(title="Korean NPC Dialogue Server")

    print("[app] NpcServer 초기화 중...")
    engine = NpcServer(adapters_dir=ADAPTERS_DIR, chroma_dir=CHROMA_DIR)
    print("[app] 준비 완료")
    app.state.engine = engine

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "npcs": engine.characters}

    @app.get("/npcs")
    def list_npcs():
        return JSONResponse({
            npc: {
                "memory_count": engine.stores[npc].count(),
            }
            for npc in engine.characters
        })

    # 한 세션 = 한 대화. 직전 6쌍(=12 메시지) 유지해서 멀티턴 흐름 살림.
    HISTORY_TURNS = 6

    @app.websocket("/ws/{npc_name}")
    async def chat_ws(ws: WebSocket, npc_name: str):
        if npc_name not in engine.characters:
            await ws.close(code=1008, reason=f"unknown npc: {npc_name}")
            return

        await ws.accept()
        await ws.send_json({"type": "ready", "npc": npc_name})

        history: list[dict] = []

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "invalid JSON"})
                    continue

                msg_type = msg.get("type")
                if msg_type == "reset":
                    history.clear()
                    await ws.send_json({"type": "reset_ok"})
                    continue
                if msg_type != "chat":
                    await ws.send_json({"type": "error", "message": "unsupported type"})
                    continue

                user_text = msg.get("text", "").strip()
                if not user_text:
                    await ws.send_json({"type": "error", "message": "empty text"})
                    continue

                try:
                    result = engine.respond(npc_name, user_text, history=history)
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue

                # 다음 턴을 위해 history 갱신 (원본 user_text + assistant 응답)
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result["text"]})
                # 최근 N쌍만 유지
                if len(history) > HISTORY_TURNS * 2:
                    history = history[-HISTORY_TURNS * 2:]

                await ws.send_json({"type": "response", **result})

        except WebSocketDisconnect:
            pass

    return app


app = create_app()
