"""전파 로직 실제 작동 데모 — 진짜 PropagationSimulator를 돌려 이벤트를 출력.

simulator.py의 _select_to_share(임계값 7) + IMPORTANCE_FACTOR(페르소나 보정)
 + chain_origin + 같은 틱 relay 를 그대로 사용. 모델/GPU 불필요 (in-memory store).

사용: uv run python scripts/demo_propagation.py
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryEntry, MemorySource  # noqa: E402
from src.propagation.graph import RelationGraph  # noqa: E402
from src.propagation.simulator import (  # noqa: E402
    IMPORTANCE_FACTOR,
    PropagationSimulator,
)

NPCS = ["mathilda", "finn", "elias", "bernhardt", "hermann"]


class MemStore:
    """시뮬레이터가 쓰는 .all()/.add()/.collection.update()만 구현한 in-memory store."""

    def __init__(self):
        self._docs = {}

        class _Coll:
            def __init__(s, p):
                s._p = p

            def update(s, ids, metadatas):
                for i, m in zip(ids, metadatas):
                    if i in s._p._docs:
                        s._p._docs[i][1].update(m)

        self.collection = _Coll(self)

    def add(self, e):
        meta = {
            "importance": e.importance,
            "timestamp": e.timestamp.isoformat(),
            "source": e.source.value,
            **e.metadata,
        }
        self._docs[e.id] = [e.text, meta]

    def all(self):
        ids = list(self._docs)
        return {
            "ids": ids,
            "documents": [self._docs[i][0] for i in ids],
            "metadatas": [self._docs[i][1] for i in ids],
        }


class _NoTransform:
    def transform(self, sender, text, source="observation"):
        return text


def complete_graph(freq):
    edges = []
    for i in range(len(NPCS)):
        for j in range(i + 1, len(NPCS)):
            edges.append((NPCS[i], NPCS[j], freq))
    return RelationGraph(edges)


def run(origin, freq=0.6, ticks=5, seed=42):
    print("=" * 66)
    print(f"  시작 NPC: {origin}  (factor ×{IMPORTANCE_FACTOR[origin]:.2f})    freq={freq}")
    print("=" * 66)

    graph = complete_graph(freq)
    stores = {n: MemStore() for n in NPCS}
    stores[origin].add(MemoryEntry(
        id=f"seed_{uuid.uuid4().hex[:8]}",
        text="광장에서 큰 곰을 봤다",
        importance=8,
        timestamp=datetime.now(timezone.utc),
        source=MemorySource.OBSERVATION,
        metadata={"day": 0},
    ))

    sim = PropagationSimulator(
        graph, stores, _NoTransform(),
        rng_seed=seed, importance_threshold=7, use_transform=False,
    )

    knows = {origin}
    print(f"  day0: {origin} 이(가) '곰'을 imp 8로 보유 (직접 목격)\n")

    for day in range(1, ticks + 1):
        events = sim.tick(day)
        if not events:
            print(f"  day{day}: (새 전파 없음)")
            continue
        for ev in events:
            f, t = ev["from"], ev["to"]
            ib, ia = ev["importance_before"], ev["importance_after"]
            fac = IMPORTANCE_FACTOR[f]
            mark = "  <- imp<7 : 재전파 불가 (체인 끊김)" if ia < 7 else ""
            print(f"  day{day}: {f:10s} -> {t:10s} | imp {ib} x{fac:.2f} = {ia}{mark}")
            knows.add(t)
        print(f"          [ 아는 사람 {len(knows)}/5 ]")
    print(f"\n  >>> {ticks}틱 결과: {len(knows)}/5 명 도달\n")


if __name__ == "__main__":
    run("elias", freq=0.6, ticks=5)   # 댐: imp 깎여서 체인 끊김
    run("finn", freq=0.6, ticks=5)    # 확성기: imp 키워서 연쇄
