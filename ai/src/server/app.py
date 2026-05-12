"""FastAPI WebSocket 서버.

엔드포인트:
  GET  /healthz           서버 상태 + 로딩된 NPC 목록
  GET  /npcs              NPC별 메모리 수
  POST /compare           같은 텍스트를 5종 NPC에 보내고 응답 모음 (페르소나 비교 demo)
  WS   /ws/{npc_name}     NPC와 대화 + 시간 진행 명령

WebSocket 프로토콜:
  Client -> Server (JSON):
    {"type": "chat",         "text": "..."}        대화
    {"type": "reset"}                              세션 history 초기화
    {"type": "time_advance"}                       하루 진행 (정보 전파 tick)

  Server -> Client (JSON):
    {"type": "ready",        "npc": "..."}
    {"type": "response",     "npc": "...", "text": "...", "memories_used": [...], "latency_ms": int}
    {"type": "reset_ok"}
    {"type": "tick_events",  "day": int, "events": [...], "memory_counts": {...}}
    {"type": "error",        "message": "..."}
"""

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import NpcServer


class CompareRequest(BaseModel):
    text: str

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"


def create_app() -> FastAPI:
    app = FastAPI(title="Korean NPC Dialogue Server")

    print("[app] NpcServer 초기화 중...")
    # use_memory=True: ChromaDB 회상 활성화 (단계적 접근 2단계)
    engine = NpcServer(
        adapters_dir=ADAPTERS_DIR,
        chroma_dir=CHROMA_DIR,
        use_memory=True,
    )
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

    @app.post("/compare")
    def compare_npcs(req: CompareRequest):
        """같은 텍스트를 5종 NPC에 보내고 각자 응답을 모아서 반환.

        페르소나 비교 demo용. 발표 시 "같은 사실, 5종 NPC가 어떻게 다르게 표현하는가" 시연.
        history는 사용 안 함 (각 NPC 독립 single-turn 응답).
        """
        text = req.text.strip()
        if not text:
            return JSONResponse({"error": "empty text"}, status_code=400)

        responses = []
        for npc in engine.characters:
            try:
                result = engine.respond(npc, text, history=None)
                responses.append({
                    "npc": npc,
                    "text": result["text"],
                    "latency_ms": result["latency_ms"],
                    "memories_used": result.get("memories_used", []),
                    "quest": result.get("quest"),
                    "trust": result.get("trust"),
                    "trust_label": result.get("trust_label"),
                    "trust_delta": result.get("trust_delta"),
                })
            except Exception as e:
                responses.append({
                    "npc": npc,
                    "text": "",
                    "error": str(e),
                })

        return JSONResponse({"input": text, "responses": responses})

    @app.get("/trust")
    def get_trust():
        """전체 NPC 신뢰도 스냅샷."""
        return JSONResponse({"trust": engine.trust.snapshot()})

    @app.post("/quest_complete/{npc}")
    def quest_complete(npc: str):
        """Quest 완수 시 호출 — 해당 NPC 신뢰도 +10."""
        try:
            result = engine.complete_quest(npc)
            return JSONResponse(result)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/simulate/{npc_a}/{npc_b}")
    def simulate_npc_conversation(npc_a: str, npc_b: str, num_turns: int = 3):
        """두 NPC가 자율적으로 대화. 결과를 양쪽 메모리에 저장.

        Park et al. (Generative Agents) 스타일.
        Query: ?num_turns=3 (각 NPC 발화 횟수, 총 발화 ≤ num_turns × 2).
        """
        try:
            result = engine.simulate_conversation(npc_a, npc_b, num_turns=num_turns)
            return JSONResponse(result)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/simulate_random")
    def simulate_random_pair(num_turns: int = 3):
        """관계 그래프에서 무작위 페어 1쌍 선정 → 대화 시뮬."""
        pair = engine.pick_random_pair()
        if pair is None:
            return JSONResponse({"error": "페어 선정 실패"}, status_code=400)
        a, b = pair
        try:
            result = engine.simulate_conversation(a, b, num_turns=num_turns)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/tick")
    def tick_http(npc_conversation: bool = True, num_turns: int = 2):
        """시간 진행 (HTTP). propagation + NPC-NPC 자율 대화.

        응답은 WebSocket tick_events와 동일하게 평탄화 (Unity JsonUtility 호환).
        """
        try:
            result = engine.tick(
                npc_conversation=npc_conversation,
                npc_conversation_turns=num_turns,
            )
            # event 직렬화
            serialized_events = [
                {
                    "day": ev["day"],
                    "from": ev["from"],
                    "to": ev["to"],
                    "original": ev["original"][:120],
                    "transformed": ev["transformed"][:120],
                    "importance_before": ev["importance_before"],
                    "importance_after": ev["importance_after"],
                }
                for ev in result.get("events", [])
            ]
            payload = {
                "type": "tick_events",
                "day": result["day"],
                "events": serialized_events,
                "memory_counts_dict": engine.memory_counts(),
            }
            conv = result.get("conversation")
            if conv is not None:
                payload["npc_a"] = conv.get("npc_a", "")
                payload["npc_b"] = conv.get("npc_b", "")
                payload["topic"] = (conv.get("topic", "") or "")[:120]
                payload["turns"] = [
                    {
                        "speaker": t["speaker"],
                        "speaker_ko": t.get("speaker_ko", t["speaker"]),
                        "text": t["text"][:200],
                    }
                    for t in conv.get("turns", [])
                ]
                payload["memory_saved"] = bool(conv.get("memory_saved", False))
            return JSONResponse(payload)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

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
                if msg_type == "time_advance":
                    try:
                        result = engine.tick()
                    except Exception as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        continue
                    # event 직렬화 (importance 등 숫자만 그대로, 텍스트는 짧게)
                    serialized = []
                    for ev in result["events"]:
                        serialized.append({
                            "day": ev["day"],
                            "from": ev["from"],
                            "to": ev["to"],
                            "original": ev["original"][:120],
                            "transformed": ev["transformed"][:120],
                            "importance_before": ev["importance_before"],
                            "importance_after": ev["importance_after"],
                        })
                    # NPC-NPC 대화 결과 (있을 시)
                    conv = result.get("conversation")
                    payload = {
                        "type": "tick_events",
                        "day": result["day"],
                        "events": serialized,
                        "memory_counts": engine.memory_counts(),
                    }
                    if conv is not None:
                        payload["npc_a"] = conv.get("npc_a", "")
                        payload["npc_b"] = conv.get("npc_b", "")
                        payload["topic"] = (conv.get("topic", "") or "")[:120]
                        payload["turns"] = [
                            {
                                "speaker": t["speaker"],
                                "speaker_ko": t.get("speaker_ko", t["speaker"]),
                                "text": t["text"][:200],
                            }
                            for t in conv.get("turns", [])
                        ]
                        payload["memory_saved"] = bool(conv.get("memory_saved", False))
                    await ws.send_json(payload)
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
