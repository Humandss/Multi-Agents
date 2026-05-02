"""정보 전파율 측정.

특정 NPC에게 정보 주입 → N일 시뮬 → 다른 NPC들에게 도달한 비율 측정.

지표:
  - reach_ratio: 도달 NPC 수 / 전체 NPC 수
  - reach_by_day: 일별 누적 도달 비율
  - first_reached_day: 각 NPC가 정보를 처음 받은 날
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropagationStats:
    initial_npc: str
    initial_fact: str
    total_npcs: int
    reached: set[str] = field(default_factory=set)
    first_reached_day: dict[str, int] = field(default_factory=dict)
    reach_by_day: dict[int, float] = field(default_factory=dict)

    @property
    def reach_ratio(self) -> float:
        # 자기 자신 제외 (initial_npc는 출발점이라 도달 X 처리)
        others = self.total_npcs - 1
        if others <= 0:
            return 0.0
        # reached에서 initial_npc는 제외하지 않음 (자체 보유는 도달 아님)
        external = len(self.reached - {self.initial_npc})
        return external / others


def compute_propagation_stats(
    initial_npc: str,
    initial_fact: str,
    events: list[dict],
    all_npcs: list[str],
) -> PropagationStats:
    """events 로그에서 propagation 통계 계산.

    events: PropagationSimulator.tick() 출력 (모든 day 합쳐서)
            각 ev: {day, from, to, original, transformed, ...}
    """
    stats = PropagationStats(
        initial_npc=initial_npc,
        initial_fact=initial_fact,
        total_npcs=len(all_npcs),
    )
    stats.reached.add(initial_npc)
    stats.first_reached_day[initial_npc] = 0

    # day 순으로 정렬해서 처리
    sorted_events = sorted(events, key=lambda e: e["day"])
    max_day = max((e["day"] for e in sorted_events), default=0)

    for ev in sorted_events:
        receiver = ev["to"]
        if receiver in stats.reached:
            continue
        # 이 transformed가 initial_fact 관련된 정보인지 판단 — 단순화: 모든 event를 카운트
        # (정확한 판단은 distortion BertDistortion으로 가능)
        stats.reached.add(receiver)
        stats.first_reached_day[receiver] = ev["day"]

    # 일별 누적
    for d in range(0, max_day + 1):
        reached_by_d = sum(1 for npc, day in stats.first_reached_day.items() if day <= d)
        external = reached_by_d - (1 if initial_npc in stats.first_reached_day else 0)
        denom = len(all_npcs) - 1
        stats.reach_by_day[d] = external / denom if denom > 0 else 0.0

    return stats


def filter_events_by_relevance(
    events: list[dict],
    initial_fact: str,
    embedder,
    threshold: float = 0.45,
) -> list[dict]:
    """initial_fact와 의미 유사도 threshold 이상인 events만 추출.

    BertDistortion 인스턴스를 embedder로 받음.
    initial_fact와 무관한 일반 propagation은 제외 — 진짜 그 fact가 퍼졌는지만 추적.
    """
    relevant = []
    for ev in events:
        sim = embedder.measure(initial_fact, ev["transformed"])
        if sim >= threshold:
            ev2 = {**ev, "_relevance": sim}
            relevant.append(ev2)
    return relevant
