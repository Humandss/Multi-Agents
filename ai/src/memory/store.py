"""NPC별 ChromaDB 메모리 컬렉션."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .schema import MemoryEntry

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class MemoryStore:
    def __init__(self, npc_name, base_dir, embedding_model=DEFAULT_EMBEDDING_MODEL):
        self.npc_name = npc_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.base_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=npc_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def _meta(self, entry: MemoryEntry):
        return {
            "importance": entry.importance,
            "timestamp": entry.timestamp.isoformat(),
            "source": entry.source.value,
            **entry.metadata,
        }

    def add(self, entry: MemoryEntry):
        self.collection.add(
            ids=[entry.id],
            documents=[entry.text],
            metadatas=[self._meta(entry)],
        )

    def add_many(self, entries: list[MemoryEntry]):
        if not entries:
            return
        self.collection.add(
            ids=[e.id for e in entries],
            documents=[e.text for e in entries],
            metadatas=[self._meta(e) for e in entries],
        )

    def query(self, text: str, k: int = 5):
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.query(
            query_texts=[text],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

    def all(self):
        return self.collection.get(include=["documents", "metadatas"])

    def find_player_personal(self, limit: int = 5) -> list[dict]:
        """플레이어 자기소개 메모리만 검색 (has_personal=True + player=True).

        ChromaDB $and 필터가 버전별로 다를 수 있어 player=True만 필터 + Python 후필터.
        """
        try:
            data = self.collection.get(
                where={"player": True},
                include=["documents", "metadatas"],
            )
        except Exception:
            # where 절 실패 시 전체 조회 후 Python 필터
            data = self.collection.get(include=["documents", "metadatas"])

        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        results = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            # 후필터: player + has_personal
            if not meta.get("player"):
                continue
            if not meta.get("has_personal"):
                continue
            results.append({
                "id": mid,
                "text": docs[i] if i < len(docs) else "",
                "importance": int(meta.get("importance", 5)),
                "metadata": meta,
                "similarity": 1.0,
                "score": 1.0,
            })
        # 정렬: importance 내림차순 → timestamp 내림차순 (최신 자기소개 우선)
        results.sort(key=lambda x: (-x["importance"], -x.get("metadata", {}).get("timestamp", "").__hash__() if False else 0))
        # timestamp 비교는 ISO 문자열 직접 비교 가능
        results.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True)
        results.sort(key=lambda x: x["importance"], reverse=True)  # 최종 정렬 (안정)
        return results[:limit]

    def find_player_all(self, limit: int = 20) -> list[dict]:
        """플레이어 발화 전체 (디버그용)."""
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        results = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if not meta.get("player"):
                continue
            results.append({
                "id": mid,
                "text": docs[i] if i < len(docs) else "",
                "importance": int(meta.get("importance", 5)),
                "has_personal": bool(meta.get("has_personal", False)),
                "timestamp": meta.get("timestamp", ""),
            })
        results.sort(key=lambda x: -x["importance"])
        return results[:limit]

    def count(self):
        return self.collection.count()

    def reset(self):
        try:
            self.client.delete_collection(self.npc_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.npc_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def prune(self, max_keep: int = 80, preserve_sources: tuple = ("seed",)) -> int:
        """메모리 수가 max_keep 초과 시 importance 낮고 오래된 것부터 삭제.

        - preserve_sources: 항상 보존할 source (기본 'seed')
        - 플레이어 발화 (player=True) 항상 보존 — 프로젝트 핵심 가치.
        - 플레이어 정보 propagation (text에 "플레이어가" 포함) 보존 — 다른 NPC가 plyaer 알도록.
        - 반환: 삭제된 메모리 수
        """
        if self.collection.count() <= max_keep:
            return 0

        all_data = self.all()
        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]

        # (id, importance, timestamp, source, is_player) 튜플로 정렬용 데이터
        entries = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            src = meta.get("source", "observation")
            imp = int(meta.get("importance", 5))
            ts = meta.get("timestamp", "")
            is_player = bool(meta.get("player", False))
            entries.append({
                "id": mid,
                "importance": imp,
                "timestamp": ts,
                "source": src,
                "is_player": is_player,
            })

        # 보존:
        # 1) preserve_sources (seed)
        # 2) 플레이어 직접 발화 (metadata.player=True)
        # 3) 플레이어 정보 propagation (text에 "플레이어가" 포함) — 다른 NPC가 plyaer 알게 함
        def _is_player_propagation(e):
            return "플레이어가" in (docs[ids.index(e["id"])] if e["id"] in ids else "")

        preserved = [e for e in entries
                     if e["source"] in preserve_sources
                     or e["is_player"]
                     or _is_player_propagation(e)]
        prunable = [e for e in entries
                    if e["source"] not in preserve_sources
                    and not e["is_player"]
                    and not _is_player_propagation(e)]

        # prunable 정렬: importance 낮고 오래된 것 우선 삭제
        prunable.sort(key=lambda x: (x["importance"], x["timestamp"]))

        budget = max(0, max_keep - len(preserved))
        if len(prunable) > budget:
            to_delete = [e["id"] for e in prunable[:len(prunable) - budget]]
            if to_delete:
                self.collection.delete(ids=to_delete)
                return len(to_delete)
        return 0
