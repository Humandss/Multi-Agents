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
    def __init__(
        self,
        store: MemoryStore,
        w_sim=0.70,
        w_imp=0.10,
        w_rec=0.20,
        min_similarity: float = 0.50,
        propagation_bonus: float = 0.05,
    ):
        """가중점수: w_sim*sim + w_imp*imp + w_rec*recency + propagation_bonus(if propagation).

        - w_sim 0.75 → 0.70 (의미 유사도 비중 약간 ↓, 다양성 ↑)
        - w_rec 0.15 → 0.20 (최근 propagation 정보 우선)
        - min_similarity 0.50: LoRA 폐기 + LLM-only 환경에서는 회상이 핵심 가치라
          0.55에서 0.50으로 낮춤. 측정 결과 elias 0.545, finn 0.494 같은 정당한
          매칭이 0.55 임계값 바로 아래로 떨어져 회상 실패. short query (<8자)는
          retriever.search에서 별도 스킵 처리하므로 generic noise는 그쪽에서 차단.
        - propagation_bonus — 체인 거친 정보가 더 잘 떠오르도록
        """
        self.store = store
        self.w_sim = w_sim
        self.w_imp = w_imp
        self.w_rec = w_rec
        self.min_similarity = min_similarity
        self.propagation_bonus = propagation_bonus

    def search(
        self,
        query: str,
        k: int = 5,
        pool: int = 20,
        exclude_sources: set[str] | None = None,
    ):
        """pool개 후보 중 의미 유사도 임계값 이상 + 가중 점수 상위 k개 반환.

        min_similarity 이하 매칭은 무관한 메모리로 간주하고 버린다.
        예: "안녕하세요"는 어떤 메모리와도 충분히 유사하지 않아 빈 리스트 반환 →
        LoRA는 평소 인사 패턴으로 응답.

        짧은 query (8자 미만) — 인사·감탄사·단답류는 회상 자체를 스킵.
        이런 query는 generic Korean baseline noise로 임계값 통과해도 의미 매칭 약하고,
        회상 컨텍스트가 단답·무뚝뚝 페르소나(hermann)를 깨뜨리는 부작용 더 큼.

        exclude_sources: 특정 소스 (예: 'dialogue') 메모리는 결과에서 제외.
        본인이 한 말을 그대로 회상하는 어색한 루프 방지용.
        """
        if len(query.strip()) < 8:
            return []
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
            if exclude_sources and meta.get("source") in exclude_sources:
                continue
            sim = max(0.0, 1.0 - dist)
            if sim < self.min_similarity:
                continue
            imp = float(meta.get("importance", 5)) / 10.0

            ts = _parse_ts(meta.get("timestamp", ""))
            if ts is None:
                rec = 0.5
            else:
                # 7일 이내 메모리에 강한 가중 (게임 내 시뮬 시간 기준)
                days = (now - ts).days
                if days <= 1:
                    rec = 1.0
                elif days <= 7:
                    rec = 0.85
                else:
                    rec = max(0.0, 1.0 - days / 30.0)

            score = self.w_sim * sim + self.w_imp * imp + self.w_rec * rec
            # propagation 소스 보너스 — 다른 NPC한테 들은 정보가 더 잘 떠오름
            if meta.get("source") == "propagation":
                score += self.propagation_bonus
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
