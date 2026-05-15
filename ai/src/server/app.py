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

    # 외부 인터페이스용 이름 표시: hermann → Hermann
    # 내부(ChromaDB·persona·memory) ID는 그대로 소문자 유지.
    def display(name: str) -> str:
        return name[:1].upper() + name[1:] if name else name

    def normalize(name: str) -> str:
        """외부에서 들어온 이름(대소문자 무관)을 내부 소문자 ID로."""
        if not name:
            return name
        low = name.lower()
        return low if low in engine.characters else name

    def add_display_npc(d: dict) -> dict:
        """respond/quest_complete 결과 dict의 npc 필드를 capitalize."""
        if "npc" in d and isinstance(d["npc"], str):
            d["npc"] = display(d["npc"])
        return d

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "npcs": [display(n) for n in engine.characters]}

    @app.get("/npcs")
    def list_npcs():
        return JSONResponse({
            display(npc): {
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
                    "npc": display(npc),
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
                    "npc": display(npc),
                    "text": "",
                    "error": str(e),
                })

        return JSONResponse({"input": text, "responses": responses})

    @app.get("/trust")
    def get_trust():
        """전체 NPC 신뢰도 스냅샷."""
        snap = engine.trust.snapshot()
        return JSONResponse({
            "trust": {display(k): v for k, v in snap.items()}
        })

    @app.post("/quest_complete/{npc}")
    def quest_complete(npc: str, quest_id: str | None = None):
        """Quest 완수 시 호출 — 해당 NPC 신뢰도 +10. quest_id 옵션."""
        try:
            result = engine.complete_quest(normalize(npc), quest_id=quest_id)
            return JSONResponse(add_display_npc(result))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.get("/quests")
    def list_quests():
        """전체 quest pool + 현재 상태 (디버그/시연용)."""
        from .engine import NPC_QUEST_POOL
        result = {}
        for npc, pool in NPC_QUEST_POOL.items():
            result[display(npc)] = [
                {
                    "id": q["id"],
                    "title": q["title"],
                    "trust_required": q.get("trust_required", 0),
                    "status": engine.quests.status(q["id"]),
                }
                for q in pool
            ]
        return JSONResponse(result)

    def _decorate_conversation(result: dict) -> dict:
        """simulate_conversation 결과의 NPC 이름들을 표시용으로."""
        if "npc_a" in result: result["npc_a"] = display(result["npc_a"])
        if "npc_b" in result: result["npc_b"] = display(result["npc_b"])
        if "turns" in result:
            for t in result["turns"]:
                if "speaker" in t: t["speaker"] = display(t["speaker"])
        return result

    @app.post("/simulate/{npc_a}/{npc_b}")
    def simulate_npc_conversation(npc_a: str, npc_b: str, num_turns: int = 3):
        """두 NPC가 자율적으로 대화. 결과를 양쪽 메모리에 저장.

        Park et al. (Generative Agents) 스타일.
        Query: ?num_turns=3 (각 NPC 발화 횟수, 총 발화 ≤ num_turns × 2).
        """
        try:
            result = engine.simulate_conversation(
                normalize(npc_a), normalize(npc_b), num_turns=num_turns
            )
            return JSONResponse(_decorate_conversation(result))
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
            return JSONResponse(_decorate_conversation(result))
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
                    "from": display(ev["from"]),
                    "to": display(ev["to"]),
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
                "memory_counts_dict": {display(k): v for k, v in engine.memory_counts().items()},
            }
            conv = result.get("conversation")
            if conv is not None:
                payload["npc_a"] = display(conv.get("npc_a", ""))
                payload["npc_b"] = display(conv.get("npc_b", ""))
                payload["topic"] = (conv.get("topic", "") or "")[:120]
                payload["turns"] = [
                    {
                        "speaker": display(t["speaker"]),
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
        # 대소문자 무관 매칭: Hermann/hermann/HERMANN 모두 받아줌
        npc_internal = normalize(npc_name)
        if npc_internal not in engine.characters:
            await ws.close(code=1008, reason=f"unknown npc: {npc_name}")
            return
        npc_name = npc_internal  # 이후 로직은 소문자 ID로 동작

        await ws.accept()
        await ws.send_json({"type": "ready", "npc": display(npc_name)})

        # NPC opener 자동 송신 — 조건 충족 quest 있으면 quest intro, 없으면 greeting.
        opener = engine.get_dialogue_opener(npc_name)
        opener_text = opener.get("text", "")
        opener_quest = opener.get("quest")
        if opener_text:
            await ws.send_json({
                "type": "response",
                "npc": display(npc_name),
                "text": opener_text,
                "latency_ms": 0,
                "memories_used": [],
                "quest": opener_quest,
                "trust": engine.trust.get(npc_name),
                "trust_label": engine.trust.label(npc_name),
                "trust_delta": 0,
            })

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
                            "from": display(ev["from"]),
                            "to": display(ev["to"]),
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
                        payload["npc_a"] = display(conv.get("npc_a", ""))
                        payload["npc_b"] = display(conv.get("npc_b", ""))
                        payload["topic"] = (conv.get("topic", "") or "")[:120]
                        payload["turns"] = [
                            {
                                "speaker": display(t["speaker"]),
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

                # 응답 NPC 이름 capitalize (Hermann 등)
                result = add_display_npc(result)
                await ws.send_json({"type": "response", **result})

        except WebSocketDisconnect:
            pass

    return app


app = create_app()
