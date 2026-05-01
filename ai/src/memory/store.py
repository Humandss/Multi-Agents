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
