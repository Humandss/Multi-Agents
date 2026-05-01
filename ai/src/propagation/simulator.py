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
        max_memories_per_meeting: int = 2,
    ):
        self.graph = graph
        self.stores = stores
        self.transformer = transformer
        self.rng = random.Random(rng_seed)
        self.importance_threshold = importance_threshold
        self.max_per_meeting = max_memories_per_meeting

    def _select_to_share(self, sender_store: MemoryStore, receiver_npc: str):
        """sender가 receiver에게 전달할 메모리 후보 선택.
        - importance >= threshold
        - 이미 receiver에게 전파한 적 없음 (metadata로 추적)
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
            candidates.append({
                "id": all_data["ids"][i],
                "text": all_data["documents"][i],
                "importance": imp,
                "metadata": meta,
            })
        # 중요도 높은 순 + 약간 랜덤
        candidates.sort(key=lambda x: x["importance"], reverse=True)
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
            to_share = self._select_to_share(sender_store, receiver)

            for mem in to_share:
                transformed = self.transformer.transform(sender, mem["text"])
                new_imp = max(
                    1, min(10, round(mem["importance"] * IMPORTANCE_FACTOR.get(sender, 1.0)))
                )

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
