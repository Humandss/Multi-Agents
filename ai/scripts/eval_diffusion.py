"""정보 확산율(information diffusion) 평가 — Park et al. 2023 방식.

한 NPC에 사실을 주입하고 N틱 전파시켜, 마을 NPC 중 몇 명이 그 정보를 알게 되는지
틱별로 측정한다. (Park et al.이 "이틀 후 에이전트 몇 %가 소식을 알았나"를 잰 것과 동일.)

출력:
  - reach curve   : 틱별 누적 도달 NPC 수 (0 → 전원)
  - arrival tick  : NPC별 최초 도달 시점
  - 전파 경로     : 누가 → 누구에게, 출처(origin)는 누구, 누구한테 받았는지
  - freq sweep    : 접촉 빈도별 확산 곡선 겹쳐 그리기 (--freqs)
  - ablation      : 전파 비활성 시 출처 1명 고정 — 전파 시스템의 기여 대조

실제 PropagationSimulator + 관계 그래프 + 중요도 임계값/페르소나 보정을 그대로 사용한다.
도달(reach)은 임베딩과 무관하므로 in-memory store로 측정 → 모델 로드 없이 즉시 실행되고,
데모용 ChromaDB(data/chroma)는 건드리지 않는다.

사용:
    uv run python scripts/eval_diffusion.py --freq 0.3              # 단일 곡선 + 전파 경로
    uv run python scripts/eval_diffusion.py --freqs 0.2,0.3,0.5     # 빈도별 곡선 겹쳐 그리기
    uv run python scripts/eval_diffusion.py --inject-to mathilda --fact "광장에 곰이 나타났다" --keyword 곰
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Windows cp949 콘솔 한국어 외 문자 print 실패 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryEntry, MemorySource  # noqa: E402
from src.propagation.graph import RelationGraph  # noqa: E402
from src.propagation.simulator import PropagationSimulator  # noqa: E402

RELATIONS_PATH = ROOT / "configs" / "relations.yaml"
RESULTS_DIR = ROOT / "output" / "eval"


class MemStore:
    """PropagationSimulator가 쓰는 인터페이스만 구현한 in-memory store.

    시뮬레이터는 store의 .all() / .add() / .collection.update() 만 사용하며,
    도달(reach) 측정은 임베딩이 필요 없다. 따라서 ChromaDB/임베딩 모델 없이
    딕셔너리만으로 동작 — 실행이 즉시 끝나고 데모 DB를 오염시키지 않는다.
    """

    def __init__(self):
        self._docs: dict[str, list] = {}  # id -> [text, meta]

        class _Coll:
            def __init__(self, parent):
                self._p = parent

            def update(self, ids, metadatas):
                for i, m in zip(ids, metadatas):  # ChromaDB update처럼 주어진 키만 병합
                    if i in self._p._docs:
                        self._p._docs[i][1].update(m)

        self.collection = _Coll(self)

    def add(self, entry: MemoryEntry):
        meta = {
            "importance": entry.importance,
            "timestamp": entry.timestamp.isoformat(),
            "source": entry.source.value,
            **entry.metadata,
        }
        self._docs[entry.id] = [entry.text, meta]

    def all(self):
        ids = list(self._docs)
        return {
            "ids": ids,
            "documents": [self._docs[i][0] for i in ids],
            "metadatas": [self._docs[i][1] for i in ids],
        }


class _NoTransform:
    """fast 모드에서는 호출되지 않지만, 시뮬레이터 생성자가 요구하는 더미."""

    def transform(self, sender, text, source="observation"):
        return text


def _pct(a: int, b: int) -> int:
    return round(100 * a / b) if b else 0


def run_diffusion(graph, npcs, *, fact, keyword, importance, inject_to, ticks, seed):
    """한 번의 확산 시뮬레이션.

    반환: (curve, arrival, received_from, trace)
      curve         : [(tick, 누적 도달 수)]
      arrival       : {npc: 최초 도달 tick}
      received_from : {npc: (tick, 직접 전달자, 출처origin)}
      trace         : [(tick, from, to, origin, content)]  — 전파 1건씩
    """
    stores = {n: MemStore() for n in npcs}
    stores[inject_to].add(MemoryEntry(
        id=f"seed_{uuid.uuid4().hex[:8]}",
        text=fact,
        importance=importance,
        timestamp=datetime.now(timezone.utc),
        source=MemorySource.OBSERVATION,
        metadata={"day": 0},
    ))
    transformer = _NoTransform()

    reached = {inject_to}
    arrival = {inject_to: 0}
    received_from: dict[str, tuple] = {}
    trace: list[tuple] = []
    curve = [(0, 1)]

    for d in range(1, ticks + 1):
        # 날짜마다 다른 만남(per-tick seed) — 실제 전파 로직(그래프·임계값·페르소나 보정) 그대로.
        sim = PropagationSimulator(graph, stores, transformer,
                                   use_transform=False, rng_seed=seed + d)
        events = sim.tick(d)
        for e in events:
            content = e.get("transformed", "")
            if keyword not in content and keyword not in e.get("original", ""):
                continue  # 우리가 주입한 사실과 무관한 전파는 제외 (보통은 없음)
            origin = e.get("chain_origin", e["from"])
            trace.append((d, e["from"], e["to"], origin, content))
            if e["to"] not in reached:
                reached.add(e["to"])
                arrival[e["to"]] = d
                received_from[e["to"]] = (d, e["from"], origin)
        curve.append((d, len(reached)))
        if len(reached) == len(npcs):
            break

    return curve, arrival, received_from, trace


def _series(curve, max_tick, total):
    """curve [(tick,count)]를 0..max_tick 길이로 채움 (완료 후엔 마지막 값 유지)."""
    d = dict(curve)
    out, last = [], d.get(0, 1)
    for t in range(max_tick + 1):
        if t in d:
            last = d[t]
        out.append(last)
    return out


def single_run(graph, npcs, args, freq_label):
    total = len(npcs)
    curve, arrival, received_from, trace = run_diffusion(
        graph, npcs, fact=args.fact, keyword=args.keyword,
        importance=args.importance, inject_to=args.inject_to,
        ticks=args.ticks, seed=args.seed)

    print(f"[eval] 정보 확산 측정")
    print(f"  사실 : \"{args.fact}\"  (판정 키워드 '{args.keyword}')")
    print(f"  출처 : {args.inject_to}   마을 : {total}명   "
          f"중요도 : {args.importance}   접촉빈도 : {freq_label}\n")
    print(f"  틱 0: {args.inject_to} 만 앎  (1/{total})")
    for tick, cnt in curve[1:]:
        print(f"  틱 {tick}: {cnt}/{total} 도달 ({_pct(cnt, total)}%)")
    final = curve[-1][1]
    if final == total:
        print(f"\n  → 전체 도달! {curve[-1][0]}틱 만에 {total}명 전원")

    # 전파 경로 — 새로 도달한 순간만 (이미 아는 NPC끼리 재전송은 CSV에만 기록).
    print("\n[전파 경로] — 정보가 새로 도달한 순간 (출처 origin 추적)")
    first = sorted(((d, frm, to, origin) for to, (d, frm, origin) in received_from.items()))
    print(f"  틱 0: {args.inject_to} (최초 목격자 = 출처)")
    for (d, frm, to, origin) in first:
        print(f"  틱 {d}: {frm} → {to}   (출처: {origin})")

    print("\n[각 NPC가 누구한테 처음 들었나]")
    for n in npcs:
        if n == args.inject_to:
            print(f"  {n}: (출처 — 최초 목격자)")
        elif n in received_from:
            d, frm, origin = received_from[n]
            print(f"  {n}: {frm} 한테 들음 (틱 {d}, 출처 {origin})")
        else:
            print(f"  {n}: 미도달")

    print(f"\n[요약] 최종 도달 {final}/{total} ({_pct(final, total)}%)  "
          f"· 전파 OFF였다면 1/{total} 고정")

    # CSV: reach curve (ON/OFF)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "diffusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tick", "reached_with_propagation", "reached_without_propagation"])
        for tick, cnt in curve:
            w.writerow([tick, cnt, 1])
    # CSV: 전파 경로 (보고서 부록용)
    with open(RESULTS_DIR / "diffusion_trace.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tick", "from", "to", "origin", "content"])
        for row in trace:
            w.writerow(row)
    print(f"\n[저장] {RESULTS_DIR / 'diffusion.csv'}  (확산 곡선 ON/OFF)")
    print(f"[저장] {RESULTS_DIR / 'diffusion_trace.csv'}  (전파 경로: from/to/origin)")

    _plot_single(curve, total)
    _plot_network(npcs, graph.edges(), received_from, arrival, args.inject_to)


def sweep_run(base_graph, npcs, args, freqs):
    total = len(npcs)
    results = {}  # freq -> curve
    print(f"[eval] 접촉 빈도별 확산 (sweep) — 빈도 {freqs}\n")
    for fq in freqs:
        g = RelationGraph([(a, b, fq) for a, b, _ in base_graph.edges()])
        curve, _arr, _rf, _tr = run_diffusion(
            g, npcs, fact=args.fact, keyword=args.keyword,
            importance=args.importance, inject_to=args.inject_to,
            ticks=args.ticks, seed=args.seed)
        results[fq] = curve
        last_tick, final = curve[-1]
        done = f"{last_tick}틱에 전원" if final == total else f"{args.ticks}틱 후 {final}/{total}"
        print(f"  빈도 {fq}: {done}")

    max_tick = max(c[-1][0] for c in results.values())
    max_tick = max(max_tick, args.ticks if any(c[-1][1] < total for c in results.values()) else max_tick)

    # CSV: tick + 빈도별 열
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "diffusion_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tick"] + [f"freq_{fq}" for fq in freqs])
        for t in range(max_tick + 1):
            w.writerow([t] + [_series(results[fq], max_tick, total)[t] for fq in freqs])
    print(f"\n[저장] {RESULTS_DIR / 'diffusion_sweep.csv'}")

    _plot_sweep(results, freqs, max_tick, total)


def _plot_single(curve, total):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r[0] for r in curve]
        ys = [r[1] for r in curve]
        plt.figure(figsize=(6, 4))
        plt.plot(ts, ys, "o-", label="propagation ON")
        plt.plot(ts, [1] * len(ts), "--", color="gray", label="propagation OFF (baseline)")
        plt.xlabel("tick (N-key presses)")
        plt.ylabel("# NPCs who know")
        plt.title("Information diffusion")
        plt.ylim(0, total + 0.3)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(RESULTS_DIR / "diffusion.png", dpi=130, bbox_inches="tight")
        print(f"[저장] {RESULTS_DIR / 'diffusion.png'}")
    except Exception as e:
        print(f"[matplotlib 미설치 — CSV로 그래프 그리면 됨]  ({e})")


def _plot_network(npcs, edges, received_from, arrival, inject_to):
    """관계 네트워크 위에 실제 전파 경로를 겹쳐 그림.

    회색 선  = 관계(누가 누구와 연결) · 파란 화살표 = 실제 전파(누가→누구) ·
    노드 색  = 도달 틱(언제 알게 됐나) · origin = 최초 목격자.
    """
    try:
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = sorted(npcs)
        pos = {}
        for i, n in enumerate(order):
            ang = math.pi / 2 + 2 * math.pi * i / len(order)  # 위에서 시작, 시계 반대
            pos[n] = (math.cos(ang), math.sin(ang))

        maxt = max([t for t in arrival.values()] + [1])
        fig, ax = plt.subplots(figsize=(6.2, 6))

        # 관계 엣지 (옅은 회색) — 연결돼 있음을 보여줌
        for a, b, _f in edges:
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="gray", alpha=0.25, lw=1, zorder=1)

        # 실제 전파 경로 (첫 도달) — 굵은 파란 화살표
        for to, (d, frm, _origin) in received_from.items():
            if frm in pos and to in pos:
                ax.annotate("", xy=pos[to], xytext=pos[frm],
                            arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=2.2,
                                            shrinkA=20, shrinkB=20), zorder=2)

        # 노드 — 도달 틱으로 색칠
        xs = [pos[n][0] for n in order]
        ys = [pos[n][1] for n in order]
        cs = [arrival.get(n, maxt + 1) for n in order]
        sc = ax.scatter(xs, ys, s=1500, c=cs, cmap="YlOrRd",
                        vmin=0, vmax=maxt, edgecolors="black", linewidths=1.2, zorder=3)
        for n in order:
            tick_lbl = f"\n(t{arrival[n]})" if n in arrival else "\n(-)"
            star = "*" if n == inject_to else ""
            ax.annotate(star + n + tick_lbl, pos[n], ha="center", va="center",
                        fontsize=9, zorder=4)

        cb = plt.colorbar(sc, ax=ax, shrink=0.65)
        cb.set_label("arrival tick")
        ax.set_title(f"Propagation over relationship network  (origin*: {inject_to})")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        plt.savefig(RESULTS_DIR / "diffusion_network.png", dpi=130, bbox_inches="tight")
        print(f"[저장] {RESULTS_DIR / 'diffusion_network.png'}  (관계망 위 전파 경로)")
    except Exception as e:
        print(f"[network 그림 생략 — matplotlib 필요]  ({e})")


def _plot_sweep(results, freqs, max_tick, total):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        for fq in freqs:
            ys = _series(results[fq], max_tick, total)
            plt.plot(range(max_tick + 1), ys, "o-", label=f"freq={fq}")
        plt.plot(range(max_tick + 1), [1] * (max_tick + 1), "--",
                 color="gray", label="propagation OFF")
        plt.xlabel("tick (N-key presses)")
        plt.ylabel("# NPCs who know")
        plt.title("Information diffusion by contact frequency")
        plt.ylim(0, total + 0.3)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(RESULTS_DIR / "diffusion_sweep.png", dpi=130, bbox_inches="tight")
        print(f"[저장] {RESULTS_DIR / 'diffusion_sweep.png'}")
    except Exception as e:
        print(f"[matplotlib 미설치 — CSV로 그래프 그리면 됨]  ({e})")


PERSONA_FACTOR = {"hermann": 1.00, "mathilda": 0.95, "finn": 1.15,
                  "bernhardt": 0.90, "elias": 0.70}


def origin_sweep_run(base_graph, npcs, args, origins):
    """출처 NPC를 바꿔가며 확산 비교 — 성격별 중요도 보정이 전파에 주는 영향.

    동일 시드 → meeting 패턴은 같고 성격(보정 계수)만 다르므로, 곡선 차이는
    온전히 페르소나 효과(릴레이 가능/차단)에서 비롯된다.
    """
    total = len(npcs)
    f = args.freq if args.freq is not None else 0.3
    graph = RelationGraph([(a, b, f) for a, b, _ in base_graph.edges()])
    results = {}
    print(f"[eval] 출처 성격별 확산 — 접촉빈도 {f}, 동일 시드(만남 동일·성격만 차이)\n")
    for origin in origins:
        if origin not in npcs:
            print(f"  (건너뜀: '{origin}' 없음)")
            continue
        curve, _a, _rf, _tr = run_diffusion(
            graph, npcs, fact=args.fact, keyword=args.keyword,
            importance=args.importance, inject_to=origin, ticks=args.ticks, seed=args.seed)
        results[origin] = curve
        last_tick, final = curve[-1]
        fac = PERSONA_FACTOR.get(origin, 1.0)
        done = f"{last_tick}틱에 전원" if final == total else f"{args.ticks}틱 후 {final}/{total}"
        print(f"  {origin} (×{fac:.2f}): {done}")

    if not results:
        return
    max_tick = max(c[-1][0] for c in results.values())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "diffusion_by_origin.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["tick"] + list(results.keys()))
        for t in range(max_tick + 1):
            w.writerow([t] + [_series(results[o], max_tick, total)[t] for o in results])
    print(f"\n[저장] {RESULTS_DIR / 'diffusion_by_origin.csv'}")
    _plot_origin_sweep(results, max_tick, total)


def _plot_origin_sweep(results, max_tick, total):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6.4, 4.2))
        for origin in results:
            ys = _series(results[origin], max_tick, total)
            fac = PERSONA_FACTOR.get(origin, 1.0)
            plt.plot(range(max_tick + 1), ys, "o-", label=f"{origin} (x{fac:.2f})")
        plt.xlabel("tick (N-key presses)")
        plt.ylabel("# NPCs who know")
        plt.title("Information diffusion by source persona")
        plt.ylim(0, total + 0.3)
        plt.grid(alpha=0.3)
        plt.legend(title="origin (importance factor)")
        plt.savefig(RESULTS_DIR / "diffusion_by_origin.png", dpi=130, bbox_inches="tight")
        print(f"[저장] {RESULTS_DIR / 'diffusion_by_origin.png'}")
    except Exception as e:
        print(f"[matplotlib 미설치 — CSV로 그래프 그리면 됨]  ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-to", default="hermann", help="사실을 처음 주입할 NPC")
    ap.add_argument("--ticks", type=int, default=10, help="전파 틱 수 (N키 횟수)")
    ap.add_argument("--fact", default="광장에 곰이 나타났다", help="주입할 사실")
    ap.add_argument("--keyword", default="곰", help="도달 판정 키워드 (fact 안의 고유 단어)")
    ap.add_argument("--importance", type=int, default=9,
                    help="주입 사실의 중요도 (>=7이어야 전파됨)")
    ap.add_argument("--freq", type=float, default=None,
                    help="접촉 빈도 오버라이드 (0~1). 데모 그래프는 0.9라 1틱에 포화 — "
                         "점진적 곡선은 0.2~0.4 권장")
    ap.add_argument("--freqs", type=str, default=None,
                    help="빈도 sweep (쉼표 구분, 예: 0.2,0.3,0.5) — 곡선 겹쳐 그리기")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드 (재현용)")
    ap.add_argument("--topology", choices=["complete", "chain", "hub"], default="complete",
                    help="관계망 위상. complete=실제 그래프(전원 직접 연결), "
                         "chain=일렬(다단계 릴레이 전파가 뚜렷), hub=마틸다(허브) 중심")
    ap.add_argument("--origins", type=str, default=None,
                    help="출처 NPC sweep (쉼표 구분 또는 'all'). 성격(중요도 보정)별 확산 차이 비교")
    args = ap.parse_args()

    if not RELATIONS_PATH.exists():
        print(f"[eval] 관계 그래프 없음: {RELATIONS_PATH}")
        return
    base_graph = RelationGraph.load(RELATIONS_PATH)
    npcs = base_graph.all_npcs()
    if args.inject_to not in npcs:
        print(f"[eval] --inject-to '{args.inject_to}' 가 그래프에 없음. 가능: {npcs}")
        return

    # 위상 오버라이드 — 릴레이 전파를 명확히 보이려면 sparse 위상 사용 (그림용 illustrative).
    if args.topology != "complete":
        f = args.freq if args.freq is not None else 0.9
        if args.topology == "chain":
            order = [n for n in ["hermann", "bernhardt", "mathilda", "finn", "elias"] if n in npcs]
            edges = [(order[i], order[i + 1], f) for i in range(len(order) - 1)]
        else:  # hub — 마틸다(정보 허브) 중심
            hub = "mathilda" if "mathilda" in npcs else npcs[0]
            edges = [(hub, n, f) for n in npcs if n != hub]
        base_graph = RelationGraph(edges)
        npcs = base_graph.all_npcs()
        print(f"[eval] 위상: {args.topology} (엣지 {len(edges)}개, 빈도 {f})")

    if args.origins:
        if args.origins.strip() == "all":
            origins = sorted(npcs, key=lambda n: -PERSONA_FACTOR.get(n, 1.0))
        else:
            origins = [o.strip() for o in args.origins.split(",") if o.strip()]
        origin_sweep_run(base_graph, npcs, args, origins)
    elif args.freqs:
        freqs = [float(x) for x in args.freqs.split(",") if x.strip()]
        sweep_run(base_graph, npcs, args, freqs)
    else:
        if args.freq is not None:
            graph = RelationGraph([(a, b, args.freq) for a, b, _ in base_graph.edges()])
            label = f"{args.freq} (오버라이드)"
        else:
            graph = base_graph
            label = "그래프 기본(0.9)"
        single_run(graph, npcs, args, label)


if __name__ == "__main__":
    main()
