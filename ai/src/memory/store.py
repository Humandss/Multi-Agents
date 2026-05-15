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

    def prune(self, max_keep: int = 60, preserve_sources: tuple = ("seed",)) -> int:
        """메모리 수가 max_keep 초과 시 importance 낮고 오래된 것부터 삭제.

        - preserve_sources: 항상 보존할 source (기본 'seed')
        - 반환: 삭제된 메모리 수
        """
        if self.collection.count() <= max_keep:
            return 0

        all_data = self.all()
        ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]

        # (id, importance, timestamp, source) 튜플로 정렬용 데이터
        entries = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            src = meta.get("source", "observation")
            imp = int(meta.get("importance", 5))
            ts = meta.get("timestamp", "")
            entries.append({
                "id": mid,
                "importance": imp,
                "timestamp": ts,
                "source": src,
            })

        # 보존: preserve_sources에 해당하는 것은 항상 보존
        preserved = [e for e in entries if e["source"] in preserve_sources]
        prunable = [e for e in entries if e["source"] not in preserve_sources]

        # prunable 정렬: importance 낮고 오래된 것 우선 삭제
        prunable.sort(key=lambda x: (x["importance"], x["timestamp"]))

        # max_keep 안에 들어가도록 prunable 일부만 keep
        budget = max(0, max_keep - len(preserved))
        # 마지막 budget개 (importance 높은) 보존, 나머지 삭제
        if len(prunable) > budget:
            to_delete = [e["id"] for e in prunable[:len(prunable) - budget]]
            if to_delete:
                self.collection.delete(ids=to_delete)
                return len(to_delete)
        return 0
