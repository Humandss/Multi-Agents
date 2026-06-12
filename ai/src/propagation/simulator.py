"""일 단위 정보 전파 시뮬레이션."""

import random
import uuid
from datetime import datetime, timezone

from ..memory import MemoryEntry, MemorySource, MemoryStore
from .graph import RelationGraph
# transformer 인자는 duck-typed (transform(sender, text, source) 메서드만 있으면 됨).
# 실제로는 NpcServer가 transformer 역할 수행. PersonaTransformer 클래스는 legacy.


# 페르소나별 importance 보정 (논문 Table 1)
# - hermann: 사실 그대로 (1.0)
# - mathilda: 사교적, 약간 약화 (0.95)
# - finn: 증폭 (1.15)
# - bernhardt: 실용 가치만 선택적 보존 (0.9)
# - elias: 회의적, 의심해서 약화 (0.7)
IMPORTANCE_FACTOR = {
    "hermann": 1.00,
    "mathilda": 0.95,
    "finn": 1.15,
    "bernhardt": 0.90,
    "elias": 0.70,
}


class PropagationSimulator:
    def __init__(
        self,
        graph: RelationGraph,
        stores: dict[str, MemoryStore],
        transformer,  # duck-typed: transform(sender, text, source) 메서드 필요
        rng_seed: int = 42,
        importance_threshold: int = 7,
        max_memories_per_meeting: int = 2,  # 1→2: 한 만남에 2개 전파 (중요 사건 누락 X)
        use_transform: bool = False,  # 페르소나 변환 LLM 호출 비활성 → 큰 속도 향상
    ):
        self.graph = graph
        self.stores = stores
        self.transformer = transformer
        self.rng = random.Random(rng_seed)
        self.importance_threshold = importance_threshold
        self.max_per_meeting = max_memories_per_meeting
        self.use_transform = use_transform

    def _select_to_share(self, all_data: dict, receiver_npc: str, sender_npc: str):
        """sender가 receiver에게 전달할 메모리 후보 선택.
        - importance >= threshold
        - 이미 receiver에게 전파한 적 없음 (metadata로 추적)
        - 자가-에코 차단: 이 메모리의 chain_origin이 receiver면 skip
        - 보조: 텍스트에 receiver 이름 명시되면 skip

        all_data: sender_store.all() 결과 — tick()이 sender별로 1회만 조회해 전달
        (엣지마다 full-scan 반복 방지. shared_with 마커는 receiver별이라 캐시 안전).
        """
        candidates = []
        for i, meta in enumerate(all_data["metadatas"]):
            imp = int(meta.get("importance", 5))
            if imp < self.importance_threshold:
                continue
            shared = meta.get(f"shared_with_{receiver_npc}", False)
            if shared:
                continue
            # 자가-에코 차단: 이 메모리의 출처가 receiver면 skip
            chain_origin = meta.get("chain_origin")
            if chain_origin == receiver_npc:
                continue
            text = all_data["documents"][i]
            # 보조 차단: 텍스트에 receiver 이름이 명시적으로 들어가면 skip
            if receiver_npc.lower() in text.lower():
                continue
            candidates.append({
                "id": all_data["ids"][i],
                "text": text,
                "importance": imp,
                "metadata": meta,
            })
        # 정렬 우선순위:
        # 1. importance 내림차순
        # 2. 같은 importance면 dialogue 우선 (방금 일어난 일이 시드 사실보다 먼저 퍼져야 자연스러움)
        # 3. 그래도 동률이면 propagation 보다 dialogue/seed 우선 (체인 짧을수록 정확)
        def _src_priority(c):
            src = c["metadata"].get("source", "observation")
            return {"dialogue": 0, "observation": 1, "seed": 2, "propagation": 3}.get(src, 4)
        candidates.sort(key=lambda x: (-x["importance"], _src_priority(x)))
        return candidates[: self.max_per_meeting]

    def _mark_shared(self, sender_store: MemoryStore, mem_id: str, receiver_npc: str):
        """Chroma update로 metadata 갱신 (shared_with_X = true)."""
        # ChromaDB는 update API 제공
        sender_store.collection.update(
            ids=[mem_id],
            metadatas=[{f"shared_with_{receiver_npc}": True}],
        )

    def tick(self, day: int):
        """하루치 시뮬레이션 — 각 엣지에 대해 만남 확률 검사.

        최적화: sender별 store.all()을 tick당 1회만 조회 (read_cache).
        같은 tick에 새 메모리를 받은 NPC는 캐시 무효화 → 기존 동작(같은 tick
        relay 가능)을 그대로 보존하면서 스캔 횟수만 20회 → ~5회로 감소.
        """
        events = []
        read_cache: dict[str, dict] = {}
        for sender, receiver, freq in self.graph.directed_edges():
            if self.rng.random() >= freq:
                continue

            sender_store = self.stores[sender]
            receiver_store = self.stores[receiver]
            all_data = read_cache.get(sender)
            if all_data is None:
                all_data = sender_store.all()
                read_cache[sender] = all_data
            to_share = self._select_to_share(all_data, receiver, sender)

            for mem in to_share:
                source_kind = mem["metadata"].get("source", "observation")
                raw = mem["text"]
                src_meta = mem.get("metadata", {})
                # 플레이어 발화 보존 — receiver가 "플레이어 → sender 발화"임을 명확히 인식.
                is_player_origin = (
                    raw.startswith("플레이어가 말했다: ")
                    or src_meta.get("player_origin") is True
                    or src_meta.get("player") is True
                )
                src_has_personal = bool(src_meta.get("has_personal", False))
                if self.use_transform:
                    transformed = self.transformer.transform(
                        sender, raw, source=source_kind
                    )
                    # transform 후에도 출처 정보 보존
                    if is_player_origin:
                        content = raw[len("플레이어가 말했다: "):][:100]
                        transformed = f"플레이어가 나한테 '{content}'라고 했어"
                else:
                    # 빠른 모드: 페르소나 변환 생략. 출처 명시 보존.
                    if is_player_origin:
                        content = raw[len("플레이어가 말했다: "):][:100]
                        transformed = f"플레이어가 나한테 '{content}'라고 했어"
                    elif "한테 들었다: " in raw:
                        transformed = raw.split("한테 들었다: ", 1)[1][:120]
                    else:
                        transformed = raw[:120]
                new_imp = max(
                    1, min(10, round(mem["importance"] * IMPORTANCE_FACTOR.get(sender, 1.0)))
                )
                # chain_origin: 이 fact가 처음 발생한 NPC. 체인 유지.
                # - 기존 chain_origin이 있으면 그대로 (전파 사슬 추적)
                # - 없으면 sender가 시작점
                chain_origin = mem["metadata"].get("chain_origin", sender)

                # 플레이어 원본 정보는 metadata에 마커 — receiver도 find_player_personal()로 회상 가능
                new_meta = {
                    "from": sender,
                    "day": day,
                    "original_id": mem["id"],
                    "original_importance": mem["importance"],
                    "chain_origin": chain_origin,
                }
                if is_player_origin:
                    new_meta["player_origin"] = True
                if src_has_personal:
                    new_meta["has_personal"] = True

                new_entry = MemoryEntry(
                    id=f"prop_d{day}_{uuid.uuid4().hex[:8]}",
                    text=f"{sender}한테 들었다: {transformed}",
                    importance=new_imp,
                    timestamp=datetime.now(timezone.utc),
                    source=MemorySource.PROPAGATION,
                    metadata=new_meta,
                )
                receiver_store.add(new_entry)
                # receiver가 새 메모리를 받음 → 이후 receiver가 sender 역할일 때
                # 최신 상태를 보도록 캐시 무효화 (같은 tick relay 동작 보존)
                read_cache.pop(receiver, None)
                self._mark_shared(sender_store, mem["id"], receiver)

                events.append({
                    "day": day,
                    "from": sender,
                    "to": receiver,
                    "original": mem["text"],
                    "transformed": transformed,
                    "importance_before": mem["importance"],
                    "importance_after": new_imp,
                    "player_origin": is_player_origin,
                    "chain_origin": chain_origin,
                })
        return events

    def run(self, days: int):
        all_events = []
        for d in range(1, days + 1):
            events = self.tick(d)
            all_events.extend(events)
        return all_events
