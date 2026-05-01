"""정보 전파 시뮬레이션 실행 + 트레이스 출력.

사용:
    uv run python scripts/run_simulation.py                          # 기본: 7일
    uv run python scripts/run_simulation.py --days 5
    uv run python scripts/run_simulation.py --inject-to mathilda --inject "북쪽 산에서 빛나는 돌을 봤다."
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryEntry, MemorySource, MemoryStore  # noqa: E402
from src.propagation import PersonaTransformer, PropagationSimulator, RelationGraph  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
RELATIONS_PATH = ROOT / "configs" / "relations.yaml"
ADAPTERS_DIR = ROOT / "output" / "adapters"
CHROMA_DIR = ROOT / "data" / "chroma"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-to", choices=CHARACTERS, default=None)
    parser.add_argument("--inject", default=None, help="시작 시점에 주입할 사건 텍스트")
    parser.add_argument("--inject-importance", type=int, default=8)
    parser.add_argument("--save-events", default=None, help="이벤트 로그 저장 경로 (.json)")
    args = parser.parse_args()

    print("그래프 로딩...")
    graph = RelationGraph.load(RELATIONS_PATH)
    print(f"  엣지 {len(graph.edges())}, NPC {len(graph.all_npcs())}")

    print("메모리 스토어 초기화...")
    stores = {}
    for npc in CHARACTERS:
        stores[npc] = MemoryStore(npc_name=npc, base_dir=CHROMA_DIR / npc)
        print(f"  {npc}: {stores[npc].count()}개 보유")

    if args.inject_to and args.inject:
        print(f"\n주입: [{args.inject_to}] {args.inject}")
        stores[args.inject_to].add(MemoryEntry(
            id=f"inject_{uuid.uuid4().hex[:8]}",
            text=args.inject,
            importance=args.inject_importance,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.OBSERVATION,
        ))

    print("\n페르소나 변형기 로딩 (5종 어댑터)...")
    transformer = PersonaTransformer(
        adapter_paths={npc: ADAPTERS_DIR / npc for npc in CHARACTERS}
    )

    sim = PropagationSimulator(graph, stores, transformer, rng_seed=args.seed)

    print(f"\n=== {args.days}일 시뮬레이션 시작 ===")
    all_events = []
    for d in range(1, args.days + 1):
        events = sim.tick(d)
        if events:
            print(f"\n[Day {d}] {len(events)}개 전달")
            for ev in events:
                arrow = f"{ev['from']:>10} → {ev['to']:<10}"
                print(f"  {arrow} (imp {ev['importance_before']}→{ev['importance_after']})")
                print(f"    원본: {ev['original'][:80]}")
                print(f"    변형: {ev['transformed'][:80]}")
        else:
            print(f"\n[Day {d}] 만남 없음")
        all_events.extend(events)

    print(f"\n=== 시뮬 종료 — 총 {len(all_events)}개 전달 이벤트 ===")
    print("\n각 NPC 메모리 현황:")
    for npc in CHARACTERS:
        cnt = stores[npc].count()
        print(f"  {npc}: {cnt}개")

    if args.save_events:
        path = Path(args.save_events)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(all_events, f, ensure_ascii=False, indent=2)
        print(f"\n이벤트 로그 저장: {path}")


if __name__ == "__main__":
    main()
