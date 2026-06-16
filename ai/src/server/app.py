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
from . import pipeline_log


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

    # --reset 옵션 (환경 변수 NPC_RESET_ON_START=1): 시작 시 자동 메모리 reset + reseed
    import os as _os
    if _os.environ.get("NPC_RESET_ON_START") == "1":
        print("[app] --reset 감지 — ChromaDB 초기화 + 시드 재적재 중...")
        try:
            for npc in engine.characters:
                engine.stores[npc].reset()
            engine.day = 0

            # 시드 메모리 재적재
            import yaml as _yaml
            import uuid as _uuid
            from datetime import datetime, timezone
            from ..memory import MemoryEntry, MemorySource
            seed_path = ROOT / "data" / "seed" / "memories.yaml"
            reseeded = 0
            if seed_path.exists():
                with seed_path.open(encoding="utf-8") as f:
                    seed_data = _yaml.safe_load(f) or {}
                for npc, mems in seed_data.items():
                    if npc not in engine.characters:
                        continue
                    for i, m in enumerate(mems or []):
                        entry = MemoryEntry(
                            id=f"seed_{npc}_{i:03d}_{_uuid.uuid4().hex[:6]}",
                            text=m.get("text", ""),
                            importance=int(m.get("importance", 5)),
                            timestamp=datetime.now(timezone.utc),
                            source=MemorySource.SEED,
                            metadata={"npc": npc},
                        )
                        engine.stores[npc].add(entry)
                        reseeded += 1
            print(f"[app] reset 완료 — 시드 {reseeded}개 재적재. 메모리 카운트: {engine.memory_counts()}")
        except Exception as e:
            print(f"[app] reset 실패: {e}")

    # --prime N 옵션: 시작 후 N tick 자동 실행 (디버그용)
    prime_ticks = int(_os.environ.get("NPC_PRIME_TICKS", "0") or "0")
    if prime_ticks > 0:
        print(f"[app] --prime {prime_ticks} — {prime_ticks} tick 자동 실행 중...")
        for i in range(prime_ticks):
            try:
                result = engine.tick(npc_conversation=False, npc_conversation_turns=1, fast=True)
                refl = result.get("reflection", {})
                refl_count = len(refl.get("reflections", [])) if refl else 0
                print(f"[app] prime tick {i+1}: events={len(result.get('events', []))}, "
                      f"conversation={'O' if result.get('conversation') else 'X'}, "
                      f"reflection={refl_count}")
            except Exception as e:
                print(f"[app] prime tick {i+1} 실패: {e}")
        print(f"[app] prime 완료 — Day {engine.day}. 메모리: {engine.memory_counts()}")

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

    @app.post("/memory/reset")
    def reset_memory(reseed: bool = True):
        """모든 NPC의 ChromaDB 메모리 초기화 + (옵션) 시드 메모리 재적재.

        시나리오 처음부터 다시 시작할 때, 또는 코드 변경 후 기존 메모리가 호환 안 될 때.
        """
        # 1) 모든 NPC store reset
        for npc in engine.characters:
            try:
                engine.stores[npc].reset()
            except Exception as e:
                return JSONResponse({"error": f"{npc} reset 실패: {e}"}, status_code=500)

        # 2) Trust / Quest / 플레이어 식별 상태도 초기화
        engine.trust = type(engine.trust)()
        engine.quests = type(engine.quests)()
        engine.day = 0
        engine.player_name = None          # 메모리 지웠는데 이름만 남으면 모순
        engine._name_known_cache = set()   # 이름 인지 캐시도 함께
        engine._name_regex_cache = None

        # 3) 시드 메모리 재적재
        reseeded = 0
        if reseed:
            try:
                import yaml as _yaml
                seed_path = ROOT / "data" / "seed" / "memories.yaml"
                if seed_path.exists():
                    with seed_path.open(encoding="utf-8") as f:
                        seed_data = _yaml.safe_load(f) or {}
                    from datetime import datetime, timezone
                    import uuid as _uuid
                    from ..memory import MemoryEntry, MemorySource
                    for npc, mems in seed_data.items():
                        if npc not in engine.characters:
                            continue
                        for i, m in enumerate(mems or []):
                            entry = MemoryEntry(
                                id=f"seed_{npc}_{i:03d}_{_uuid.uuid4().hex[:6]}",
                                text=m.get("text", ""),
                                importance=int(m.get("importance", 5)),
                                timestamp=datetime.now(timezone.utc),
                                source=MemorySource.SEED,
                                metadata={"npc": npc},
                            )
                            engine.stores[npc].add(entry)
                            reseeded += 1
            except Exception as e:
                return JSONResponse({
                    "reset": True, "reseeded": reseeded,
                    "error": f"reseed 부분 실패: {e}",
                }, status_code=200)

        return JSONResponse({
            "reset": True,
            "reseeded_memories": reseeded,
            "memory_counts": engine.memory_counts(),
        })

    @app.post("/debug/prime")
    def prime_simulation(ticks: int = 3):
        """디버그용: 게임 시작 직후 호출하면 N tick 자동 실행.

        - propagation으로 시드 메모리가 NPC들 간 전파됨
        - 자율 대화로 NPC-NPC 대화 누적
        - 매 tick rotating reflection 트리거
        결과: Day {ticks} 상태로 점프. 사용자가 게임 들어가면 이미 정보 흐름 있음.
        """
        results = []
        for i in range(ticks):
            try:
                result = engine.tick(npc_conversation=False, npc_conversation_turns=1, fast=True)
                results.append({
                    "day": result["day"],
                    "events": len(result.get("events", [])),
                    "conversation": bool(result.get("conversation")),
                    "reflection": result.get("reflection", {}).get("reflections", []) if result.get("reflection") else [],
                })
            except Exception as e:
                results.append({"day": engine.day, "error": str(e)})

        return JSONResponse({
            "ticks_executed": ticks,
            "final_day": engine.day,
            "tick_results": results,
            "memory_counts": {display(k): v for k, v in engine.memory_counts().items()},
        })

    @app.get("/pipeline/trace")
    def pipeline_trace(n: int = 50):
        """파이프라인 로그 최근 n개 — 발화→저장→전파→언급 흐름 조회.

        발표/디버깅용. Unity '마을 일지' 패널이나 발표 자료로 활용 가능.
        """
        return JSONResponse({"events": pipeline_log.recent(n)})

    @app.post("/pipeline/clear")
    def pipeline_clear():
        """파이프라인 로그 버퍼 초기화 (새 시연 시작 시)."""
        pipeline_log.clear()
        return JSONResponse({"cleared": True})

    @app.post("/reflect/{npc}")
    def reflect_npc(npc: str):
        """특정 NPC에게 reflection 강제 실행 (Park et al. style abstraction)."""
        npc_id = normalize(npc)
        if npc_id not in engine.characters:
            return JSONResponse({"error": "unknown npc"}, status_code=404)
        try:
            result = engine.reflect(npc_id, min_importance_sum=0)  # 강제 — 임계값 무시
            result["npc"] = display(npc_id)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/memory/reflections/{npc}")
    def list_reflections(npc: str):
        """특정 NPC의 reflection 메모리 전체 (시연용)."""
        npc_id = normalize(npc)
        if npc_id not in engine.characters:
            return JSONResponse({"error": "unknown npc"}, status_code=404)
        all_data = engine.stores[npc_id].all()
        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])
        reflections = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if meta.get("source") != "reflection":
                continue
            reflections.append({
                "id": mid,
                "text": docs[i] if i < len(docs) else "",
                "importance": int(meta.get("importance", 5)),
                "day": meta.get("day", 0),
                "timestamp": meta.get("timestamp", ""),
            })
        reflections.sort(key=lambda r: r["timestamp"], reverse=True)
        return JSONResponse({"npc": display(npc_id), "reflections": reflections})

    @app.get("/memory/player/{npc}")
    def player_memories(npc: str):
        """특정 NPC가 가진 플레이어 발화 메모리 전체 (디버그/시연용)."""
        npc_id = normalize(npc)
        if npc_id not in engine.characters:
            return JSONResponse({"error": "unknown npc"}, status_code=404)
        return JSONResponse({
            "npc": display(npc_id),
            "player_memories": engine.stores[npc_id].find_player_all(limit=50),
        })

    @app.post("/debug/complete_all")
    def debug_complete_all():
        """디버그/시연용 — 진행 중(accepted)인 모든 퀘스트를 완료 처리 (Unity H키).

        각 퀘스트마다 정식 complete_quest 흐름 (trust +10 + 경험담 기억 + NPC 반응).
        """
        from .engine import NPC_QUEST_POOL
        results = []
        for npc_id, pool in NPC_QUEST_POOL.items():
            for q in pool:
                if engine.quests.status(q["id"]) == "accepted":
                    try:
                        r = engine.complete_quest(npc_id, quest_id=q["id"])
                        results.append(add_display_npc(r))
                    except Exception as e:
                        results.append({"quest_id": q["id"], "error": str(e)})
        return JSONResponse({"completed": len(results), "results": results})

    @app.post("/debug/toggle_transform")
    def debug_toggle_transform():
        """디버그/시연용 — 내용 왜곡(전파 시 LLM 페르소나 재서술) ON/OFF 토글 (Unity T키).

        OFF(기본): 빠른 모드. 전파 시 내용 보존, 중요도만 왜곡.
        ON: 전파될 때 LLM이 NPC 말투로 내용을 재서술 → 소문 변형이 눈에 보임 (느려짐).
        """
        engine.content_distortion = not engine.content_distortion
        pipeline_log.log(
            "spread",
            f"내용 왜곡 {'ON' if engine.content_distortion else 'OFF'} (T키 토글)",
            content_distortion=engine.content_distortion,
        )
        return JSONResponse({"content_distortion": engine.content_distortion})

    @app.get("/quests/{npc}")
    def quests_for_npc(npc: str):
        """특정 NPC의 퀘스트 리스트 (Unity 퀘스트 패널용).

        각 항목: id/title/description/reward/trust_required/state/eligible.
        state: available(시작 가능) | accepted(진행 중) | completed(완료)
        """
        npc_id = normalize(npc)
        if npc_id not in engine.characters:
            return JSONResponse({"error": "unknown npc"}, status_code=404)
        cur = engine.trust.get(npc_id)
        return JSONResponse({
            "npc": display(npc_id),
            "trust": cur,
            "quests": engine.quests.list_for_npc(npc_id, cur),
        })

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
    def tick_http(npc_conversation: bool | None = None, num_turns: int = 2):
        """시간 진행 (HTTP). propagation + (옵션) NPC-NPC 자율 대화.

        npc_conversation 미지정(None)이면 engine.autonomous_dialogue 플래그(기본 OFF)를 따름.
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

    # 한 세션 = 한 대화. 직전 3쌍(=6 메시지) 유지 — 속도 우선.
    HISTORY_TURNS = 3

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
        # 퀘스트 제안 후 응답 대기 상태 (세션 로컬 — 연결 끊기면 자동 소멸 = 미수락)
        pending_quest: str | None = None

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
                if msg_type == "quest_propose":
                    # 플레이어가 퀘스트 리스트에서 시작 → NPC 퀘스트 대사 (template, 즉답)
                    quest_id = (msg.get("quest_id") or "").strip()
                    result = engine.propose_quest(npc_name, quest_id)
                    if "error" in result:
                        await ws.send_json({"type": "error", "message": result["error"]})
                        continue
                    pending_quest = quest_id
                    quest = dict(result["quest"])
                    quest["giver"] = display(quest.get("giver", npc_name))
                    # history에도 반영 (이후 LLM 대화 맥락 유지)
                    history.append({"role": "assistant", "content": result["text"]})
                    await ws.send_json({
                        "type": "response",
                        "npc": display(npc_name),
                        "text": result["text"],
                        "latency_ms": 0,
                        "memories_used": [],
                        "quest": quest,
                        "quest_stage": "proposed",
                        "trust": engine.trust.get(npc_name),
                        "trust_label": engine.trust.label(npc_name),
                        "trust_delta": 0,
                    })
                    continue
                if msg_type != "chat":
                    await ws.send_json({"type": "error", "message": "unsupported type"})
                    continue

                user_text = msg.get("text", "").strip()
                if not user_text:
                    await ws.send_json({"type": "error", "message": "empty text"})
                    continue

                # 퀘스트 제안 응답 대기 중 → 분류해서 template 즉답 (LLM 생략)
                if pending_quest is not None:
                    qid = pending_quest
                    pending_quest = None
                    reply = engine.handle_quest_reply(npc_name, qid, user_text)
                    # 친밀도는 평소처럼 흐름 (+1 등)
                    trust_delta = engine.trust.on_player_turn(npc_name, user_text)
                    # history 유지 (이후 LLM 대화가 맥락을 앎)
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": reply["text"]})
                    if len(history) > HISTORY_TURNS * 2:
                        history = history[-HISTORY_TURNS * 2:]
                    await ws.send_json({
                        "type": "response",
                        "npc": display(npc_name),
                        "text": reply["text"],
                        "latency_ms": 0,
                        "memories_used": [],
                        "quest": None,
                        "quest_stage": reply["stage"],   # accepted | declined | unclear
                        "quest_id": qid,
                        "trust": engine.trust.get(npc_name),
                        "trust_label": engine.trust.label(npc_name),
                        "trust_delta": trust_delta,
                    })
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
