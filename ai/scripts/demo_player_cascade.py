"""플레이어 발화 전파 cascade 데모 그림.

플레이어가 한 NPC에게 한 말이 며칠에 걸쳐 마을로 퍼지는 경로를 그림 한 장으로.
전파 로직만 사용 → LLM/GPU 불필요 (use_transform=False).

    uv run python scripts/demo_player_cascade.py
    uv run python scripts/demo_player_cascade.py --utterance "광장에서 곰을 봤어요" --inject-to mathilda --days 5
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from src.memory import MemoryEntry, MemorySource, MemoryStore  # noqa: E402
from src.propagation import PropagationSimulator, RelationGraph  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
CHROMA_DIR = ROOT / "data" / "chroma_demo"
RELATIONS_PATH = ROOT / "configs" / "relations.yaml"
SEED_PATH = ROOT / "data" / "seed" / "memories.yaml"
OUT_PATH = ROOT / "output" / "figures" / "player_cascade.png"


def _set_korean_font():
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _reseed(stores):
    seed = yaml.safe_load(SEED_PATH.open(encoding="utf-8")) or {}
    for npc in CHARACTERS:
        stores[npc].reset()
        for i, m in enumerate(seed.get(npc, []) or []):
            stores[npc].add(MemoryEntry(
                id=f"seed_{npc}_{i:03d}_{uuid.uuid4().hex[:6]}",
                text=m.get("text", ""),
                importance=int(m.get("importance", 5)),
                timestamp=datetime.now(timezone.utc),
                source=MemorySource.SEED,
                metadata={"npc": npc},
            ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--utterance", default="광장에서 곰을 봤어요")
    p.add_argument("--inject-to", default="mathilda", choices=CHARACTERS)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spread", type=float, default=0.9,
                   help="전파 확률. 0.9=게임 기본(하루 만에 전원 도달). 낮추면 단계적 cascade 분석용 (예: --spread 0.35 --days 7).")
    args = p.parse_args()

    stores = {n: MemoryStore(npc_name=n, base_dir=CHROMA_DIR / n) for n in CHARACTERS}
    _reseed(stores)

    # 플레이어 발화 주입 (engine._save_player_turn과 동일 형식)
    stores[args.inject_to].add(MemoryEntry(
        id=f"dlg_{uuid.uuid4().hex[:8]}",
        text=f"플레이어가 말했다: {args.utterance}",
        importance=9,
        timestamp=datetime.now(timezone.utc),
        source=MemorySource.DIALOGUE,
        metadata={"player": True, "player_origin": True},
    ))

    # relations.yaml은 게임 서버용(freq 0.9)이라 한 tick에 다 퍼진다.
    # 데모에서만 freq를 낮춰(--spread) 여러 날에 걸쳐 단계적으로 번지게 한다.
    raw_edges = yaml.safe_load(RELATIONS_PATH.open(encoding="utf-8"))["edges"]
    graph = RelationGraph([(e[0], e[1], args.spread) for e in raw_edges])
    sim = PropagationSimulator(
        graph=graph,
        stores=stores, transformer=None, rng_seed=args.seed, use_transform=False,
    )

    # 도달 경로/시점 추적 (player_origin 이벤트만)
    first_day = {args.inject_to: 0}
    edges = []  # (from, to, day)
    for d in range(1, args.days + 1):
        for ev in sim.tick(d):
            if not ev.get("player_origin"):
                continue
            edges.append((ev["from"], ev["to"], d))
            if ev["to"] not in first_day:
                first_day[ev["to"]] = d
        print(f"Day {d}: 도달 {len(first_day) - 1}/{len(CHARACTERS) - 1}명")

    _plot(args, first_day, edges)


def _plot(args, first_day, edges):
    _set_korean_font()
    order = [args.inject_to] + [n for n in CHARACTERS if n != args.inject_to]
    ypos = {npc: i for i, npc in enumerate(order)}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 전파 경로 화살표 (첫 도달 경로만 — 깔끔하게)
    drawn = set()
    for frm, to, day in edges:
        if to in drawn or to not in first_day or first_day[to] != day:
            continue
        if frm not in first_day:
            continue
        drawn.add(to)
        ax.annotate("", xy=(day, ypos[to]), xytext=(first_day[frm], ypos[frm]),
                    arrowprops=dict(arrowstyle="->", color="#999", lw=1.2), zorder=1)
    # NPC 도달 점
    for npc, day in first_day.items():
        ax.scatter(day, ypos[npc], s=220, zorder=3,
                   color="#e74c3c" if npc == args.inject_to else "#3498db")
        ax.annotate(npc, (day, ypos[npc]), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=11)

    ax.set_xlabel("Day")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.5, args.days + 0.5)
    ax.set_xticks(range(0, args.days + 1))
    ax.set_title(f'플레이어 발화 전파: "{args.utterance}"  ({args.inject_to} → 마을)')
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
