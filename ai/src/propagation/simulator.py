"""일 단위 정보 전파 시뮬레이션."""

import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..memory import MemoryEntry, MemorySource, MemoryStore
from .graph import RelationGraph
from .transformer import PersonaTransformer


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
        transformer: PersonaTransformer,
        rng_seed: int = 42,
        importance_threshold: int = 6,
        max_memories_per_meeting: int = 3,
    ):
        self.graph = graph
        self.stores = stores
        self.transformer = transformer
        self.rng = random.Random(rng_seed)
        self.importance_threshold = importance_threshold
        self.max_per_meeting = max_memories_per_meeting

    def _select_to_share(self, sender_store: MemoryStore, receiver_npc: str, sender_npc: str):
        """sender가 receiver에게 전달할 메모리 후보 선택.
        - importance >= threshold
        - 이미 receiver에게 전파한 적 없음 (metadata로 추적)
        - 자가-에코 차단: 이 메모리의 chain_origin이 receiver면 skip
        - 보조: 텍스트에 receiver 이름 명시되면 skip
        """
        all_data = sender_store.all()
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
        """하루치 시뮬레이션 — 각 엣지에 대해 만남 확률 검사."""
        events = []
        for sender, receiver, freq in self.graph.directed_edges():
            if self.rng.random() >= freq:
                continue

            sender_store = self.stores[sender]
            receiver_store = self.stores[receiver]
            to_share = self._select_to_share(sender_store, receiver, sender)

            for mem in to_share:
                source_kind = mem["metadata"].get("source", "observation")
                transformed = self.transformer.transform(
                    sender, mem["text"], source=source_kind
                )
                new_imp = max(
                    1, min(10, round(mem["importance"] * IMPORTANCE_FACTOR.get(sender, 1.0)))
                )
                # chain_origin: 이 fact가 처음 발생한 NPC. 체인 유지.
                # - 기존 chain_origin이 있으면 그대로 (전파 사슬 추적)
                # - 없으면 sender가 시작점
                chain_origin = mem["metadata"].get("chain_origin", sender)

                new_entry = MemoryEntry(
                    id=f"prop_d{day}_{uuid.uuid4().hex[:8]}",
                    text=f"{sender}한테 들었다: {transformed}",
                    importance=new_imp,
                    timestamp=datetime.now(timezone.utc),
                    source=MemorySource.PROPAGATION,
                    metadata={
                        "from": sender,
                        "day": day,
                        "original_id": mem["id"],
                        "original_importance": mem["importance"],
                        "chain_origin": chain_origin,
                    },
                )
                receiver_store.add(new_entry)
                self._mark_shared(sender_store, mem["id"], receiver)

                events.append({
                    "day": day,
                    "from": sender,
                    "to": receiver,
                    "original": mem["text"],
                    "transformed": transformed,
                    "importance_before": mem["importance"],
                    "importance_after": new_imp,
                })
        return events

    def run(self, days: int):
        all_events = []
        for d in range(1, days + 1):
            events = self.tick(d)
            all_events.extend(events)
        return all_events
