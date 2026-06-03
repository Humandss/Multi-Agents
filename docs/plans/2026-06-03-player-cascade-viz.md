# 플레이어 발화 전파 시각화 — 구현 계획

> 스펙: [docs/specs/2026-06-03-player-cascade-viz.md](../specs/2026-06-03-player-cascade-viz.md)
> 체크박스(`- [ ]`)로 task별 추적.

**목표:** 플레이어가 NPC에게 한 말이 마을로 퍼지는 걸 (1) matplotlib cascade 그림, (2) Unity 5노드 펄스로 보이게 한다.

**접근:** 백엔드 전파 이벤트에 출처 필드(`player_origin`/`chain_origin`)를 노출 → 그 데이터로 그림(GPU 불필요) → Unity 펄스. **그림 먼저(안전빵), 펄스 나중(보너스).**

**기술 스택:** Python(uv) · 기존 `PropagationSimulator` · matplotlib(신규 의존) · Unity C#.

**검증 방식:** 이 repo는 pytest가 없다 → 각 task는 **"실행해서 출력 확인"**으로 검증(TDD 단위테스트 대신). 사용자 합의 사항.

---

## 파일 구조

| 파일 | 작업 | 책임 |
|---|---|---|
| `ai/src/propagation/simulator.py` | 수정 | event dict에 `player_origin`/`chain_origin` 추가 |
| `ai/src/server/app.py` | 수정 | `time_advance` 직렬화에 두 필드 노출 |
| `ai/pyproject.toml` | 수정 | matplotlib 의존 추가 |
| `ai/scripts/demo_player_cascade.py` | 신규 | 발화 주입→전파→cascade 그림 (모델 불필요) |
| `Assets/Scripts/NpcChat/PropagationGraphView.cs` | 신규(후속) | 게임 내 5노드 펄스 |

---

### Task 1: 전파 이벤트에 출처 필드 노출

**파일:**
- 수정: `ai/src/propagation/simulator.py` (`tick` 내 `events.append`, ~167행)
- 수정: `ai/src/server/app.py` (`time_advance` 핸들러 `serialized.append`, ~485행)

- [ ] **Step 1: simulator event에 두 필드 추가**

`ai/src/propagation/simulator.py`의 `tick()` 안 `events.append({...})`를 아래로 교체. `is_player_origin`(~113행)·`chain_origin`(~141행)은 같은 루프에서 이미 계산되어 있다:

```python
                events.append({
                    "day": day,
                    "from": sender,
                    "to": receiver,
                    "original": mem["text"],
                    "transformed": transformed,
                    "importance_before": mem["importance"],
                    "importance_after": new_imp,
                    "player_origin": is_player_origin,   # 추가
                    "chain_origin": chain_origin,        # 추가
                })
```

- [ ] **Step 2: app.py 직렬화에 두 필드 추가**

`ai/src/server/app.py`의 `time_advance` 핸들러 `serialized.append({...})`를 아래로 교체(필드 추가만 — 하위호환):

```python
                        serialized.append({
                            "day": ev["day"],
                            "from": display(ev["from"]),
                            "to": display(ev["to"]),
                            "original": ev["original"][:120],
                            "transformed": ev["transformed"][:120],
                            "importance_before": ev["importance_before"],
                            "importance_after": ev["importance_after"],
                            "player_origin": ev.get("player_origin", False),  # 추가
                            "chain_origin": display(ev["chain_origin"]) if ev.get("chain_origin") else "",  # 추가
                        })
```

- [ ] **Step 3: 검증 — 전파에서 player_origin이 잡히는지 (모델/GPU 불필요)**

```bash
cd ai
uv run python -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from datetime import datetime, timezone
from src.memory import MemoryEntry, MemorySource, MemoryStore
from src.propagation import PropagationSimulator, RelationGraph
CH=['elias','hermann','mathilda','finn','bernhardt']
st={n:MemoryStore(npc_name=n, base_dir=Path('data/chroma_test')/n) for n in CH}
for n in CH: st[n].reset()
st['mathilda'].add(MemoryEntry(id='t1', text='플레이어가 말했다: 광장에서 곰을 봤어요', importance=9, timestamp=datetime.now(timezone.utc), source=MemorySource.DIALOGUE, metadata={'player':True,'player_origin':True}))
sim=PropagationSimulator(graph=RelationGraph.load('configs/relations.yaml'), stores=st, transformer=None, use_transform=False)
ev=sim.tick(1)
print('player_origin 이벤트:', [(e['from'],e['to']) for e in ev if e.get('player_origin')])
"
```
기대: `player_origin 이벤트: [('mathilda', 'hermann'), ('mathilda', 'finn'), ...]` (비어있지 않음).

- [ ] **Step 4: 커밋**

```bash
git add ai/src/propagation/simulator.py ai/src/server/app.py
git commit -m "feat: 전파 이벤트에 player_origin/chain_origin 노출"
```

---

### Task 2: cascade 그림 스크립트 (안전빵)

**파일:**
- 수정: `ai/pyproject.toml` (matplotlib)
- 신규: `ai/scripts/demo_player_cascade.py`
- 출력: `ai/output/figures/player_cascade.png`

- [ ] **Step 1: matplotlib 추가**

```bash
cd ai
uv add matplotlib
```

- [ ] **Step 2: 스크립트 작성** — `ai/scripts/demo_player_cascade.py` 신규:

```python
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
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
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

    sim = PropagationSimulator(
        graph=RelationGraph.load(RELATIONS_PATH),
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
```

- [ ] **Step 3: 실행 + 그림 확인**

```bash
cd ai
uv run python scripts/demo_player_cascade.py
```
기대 출력: `Day 1: 도달 N/4명` … `저장: .../player_cascade.png`.
`ai/output/figures/player_cascade.png`를 열어 mathilda(빨강)→타 NPC(파랑) 도달 시점 + 화살표 경로 확인.

- [ ] **Step 4: 커밋**

```bash
git add ai/pyproject.toml ai/uv.lock ai/scripts/demo_player_cascade.py
git commit -m "feat: 플레이어 발화 전파 cascade 그림 스크립트"
```

> 그림 자체를 git에 보존하려면(발표 자료): `.gitignore`의 `ai/output/` 아래에 `!ai/output/figures/` 예외를 넣고 png도 `git add -f`. **선택** — 안 해도 로컬에 생성됨.

---

### Task 3 (후속, 보너스): Unity 5노드 펄스

> Task 1–2(그림)가 "안전빵"으로 완성된 뒤 착수. **착수 시점에 아래 Unity 파일을 먼저 읽고 이 task를 코드 단위로 확장한다**(현재는 개요 — Unity 직렬화 구조 미조사라 정확한 코드 미작성):
> - `Assets/Scripts/NpcChat/NpcMessage.cs` — `ServerMessage`에 `events`/`player_origin` 필드 유무
> - `Assets/Scripts/NpcChat/GameTimeController.cs` — `OnTickCompleted` 시그니처
> - `Assets/Scripts/NpcChat/NpcEventToast.cs` — 구독 패턴(재활용 템플릿)

**개요 단계:**
1. `NpcMessage.cs`의 `ServerMessage`에 `events`(from/to/player_origin) 필드 추가 — 서버 `tick_events` 페이로드와 매칭
2. 신규 `PropagationGraphView.cs`: 5 NPC 노드를 작은 패널에 **고정 좌표**로 배치 + `relations.yaml` 엣지 정적 표시
3. `GameTimeController.OnTickCompleted` 구독 → `player_origin==true`인 `from→to` 엣지를 펄스(색/굵기 0.5~1s 애니메이션)
4. 검증: Unity 에디터 Play → NPC에 "광장에서 곰을 봤어요" → 시간 넘김 → 해당 엣지 펄스 확인

---

## Self-Review

- **스펙 커버:** ① 백엔드 노출 = Task 1 ② 그림 = Task 2 ③ 펄스 = Task 3(개요). 세 산출 모두 task 존재. ✓
- **Task 3가 개요인 이유:** Unity 직렬화 구조(`NpcMessage`)를 아직 안 읽어 정확한 C# 코드를 못 박았다. 스펙상 "나중/보너스"이고, 그림 완성 후 착수 시 확장한다고 명시 — placeholder가 아니라 의도된 단계적 처리.
- **타입 일관성:** `player_origin`(bool)/`chain_origin`(str) 필드명이 simulator→app.py→demo 스크립트에서 동일. `is_player_origin`은 simulator 내부 변수, event 키는 `player_origin`로 통일. ✓
