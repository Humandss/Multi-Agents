"""NPC 관계 그래프."""

from pathlib import Path

import yaml


class RelationGraph:
    def __init__(self, edges):
        # edges: list of (a, b, freq)
        self._edges = []
        self._neighbors = {}
        for a, b, f in edges:
            self._edges.append((a, b, float(f)))
            self._neighbors.setdefault(a, []).append((b, float(f)))
            self._neighbors.setdefault(b, []).append((a, float(f)))

    @classmethod
    def load(cls, path: Path):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data["edges"])

    def edges(self):
        """모든 (a, b, freq) 반환 (한 방향)."""
        return list(self._edges)

    def directed_edges(self):
        """양방향으로 풀어서 반환 — 시뮬레이션이 양쪽 다 처리하도록."""
        result = []
        for a, b, f in self._edges:
            result.append((a, b, f))
            result.append((b, a, f))
        return result

    def neighbors(self, npc):
        return list(self._neighbors.get(npc, []))

    def all_npcs(self):
        npcs = set()
        for a, b, _ in self._edges:
            npcs.add(a)
            npcs.add(b)
        return sorted(npcs)
