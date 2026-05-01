"""의미 유사도 + 중요도 + 최신성 가중 검색.

Generative Agents (Park et al., 2023) 메모리 스트림 방식 단순화.
"""

from datetime import datetime, timezone

from .store import MemoryStore


def _parse_ts(s):
    try:
        ts = datetime.fromisoformat(s)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


class MemoryRetriever:
    def __init__(self, store: MemoryStore, w_sim=0.75, w_imp=0.10, w_rec=0.15):
        self.store = store
        self.w_sim = w_sim
        self.w_imp = w_imp
        self.w_rec = w_rec

    def search(self, query: str, k: int = 5, pool: int = 20):
        """pool개 후보 중 가중 점수 상위 k개 반환."""
        results = self.store.query(query, k=pool)
        ids = results["ids"][0]
        if not ids:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        now = datetime.now(timezone.utc)
        scored = []
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            sim = max(0.0, 1.0 - dist)
            imp = float(meta.get("importance", 5)) / 10.0

            ts = _parse_ts(meta.get("timestamp", ""))
            if ts is None:
                rec = 0.5
            else:
                days = (now - ts).days
                rec = max(0.0, 1.0 - days / 30.0)

            score = self.w_sim * sim + self.w_imp * imp + self.w_rec * rec
            scored.append({
                "id": id_,
                "text": doc,
                "metadata": meta,
                "similarity": sim,
                "importance": meta.get("importance", 5),
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
