# 플레이어 발화 전파 시각화 — 설계

> 작성 2026-06-03 · 맥락: 학교 수업 기말 과제(게임학부), 마감 6월 중순 추정.

## 목표
플레이어가 NPC에게 한 말이 마을 전체로 퍼지는 과정을 두 가지로 "눈에 보이게" 만든다.
- **(b) 발표용 정적 cascade 그림** — matplotlib, 재현 가능
- **(a) 게임 내 실시간 5노드 펄스** — Unity

## 왜
- 정보 전파는 이 프로젝트의 핵심 차별점인데, 지금은 채팅 히스토리에 작은 글씨로 흩어져 채점자에게 안 꽂힌다. 데이터는 다 흐르는데 **연출이 없다.**
- "내가 한 말이 다음 날 다른 NPC 입에서 나온다"는 게임 판타지라, 시드 fact 전파보다 시연 임팩트가 크다.
- 신규 시스템이 아니라 **기존 데이터에 연출을 더하는** 작업.

## 현재 상태 (코드로 검증됨)
플레이어 발화 전파는 백엔드에 **이미 구현되어 있다**:
- `_save_player_turn` ([engine.py:1833](../../ai/src/server/engine.py)): 평서문은 importance 7~9로 저장(질문만 4) → 전파 임계값(7) 통과 보장. fact 발화("곰 봤어요")는 9.
- 저장 형태: 텍스트 `"플레이어가 말했다: ..."` + metadata `player:True`.
- `simulator.tick` ([simulator.py](../../ai/src/propagation/simulator.py)): `is_player_origin` 특별취급 → `"플레이어가 나한테 '...'라고 했어"`로 이웃에 전파. `new_meta`에 `player_origin`/`chain_origin` 보존.

→ 빠진 것은 데이터가 아니라 **노출 + 시각화**뿐.

## 범위
### 할 것
1. **백엔드 노출**: 전파 이벤트 직렬화에 `player_origin`, `chain_origin` 노출 (메타데이터엔 이미 있음).
2. **발표용 cascade 그림 (b)**: 재현 스크립트 → matplotlib. **먼저 — 안전빵.**
3. **게임 내 5노드 펄스 (a)**: `time_advance` 이벤트의 `player_origin` 엣지를 펄스. **나중 — 보너스.**

### 안 할 것 (YAGNI)
- "누가 무엇을 아는가" 상태 패널 (C 라인으로 통일)
- 범용 노드그래프 / 자동 레이아웃 (NPC 5명 고정 배치로 충분)
- `fact_label` / true-false misinformation (이번 범위 아님)
- LLM 페르소나 변환(`use_transform`) 활성화

## 구성 요소

### 1. 백엔드 — 전파 이벤트 노출 (작음, 공통 토대)
- `simulator.tick`의 events 항목에 `player_origin`, `chain_origin` 포함 (현재 `new_meta`엔 있으나 events dict엔 미포함 → 추가).
- `app.py`의 `time_advance` / `/tick` 직렬화(serialized)에 두 필드 추가. **필드 추가만 — 하위호환**, 기존 클라이언트 영향 없음.

### 2. 발표용 cascade 그림 (b) — 먼저
- 새 스크립트: `ai/scripts/demo_player_cascade.py`
- 입력: 플레이어 발화 1개(기본 `"광장에서 곰을 봤어요"`), 주입 NPC(기본 `mathilda`), `days`(기본 5), 고정 seed.
- 처리: ChromaDB reset+seed → 발화를 주입 NPC의 DIALOGUE로 add(importance 9, `player:True`) → `days`만큼 `simulator.tick` → `player_origin` 이벤트 수집.
- 출력: 그림 1장 → `ai/output/figures/player_cascade.png`
  - 형태: x축 Day, y축 NPC, 정보 도달 시점을 점/선으로 연결한 cascade. legacy `propagation_eval`의 `first_reached_day` / `reach_by_day` 구조 재활용.
- 재현 가능(고정 seed) → 발표·보고서에 그대로 사용.

### 3. 게임 내 5노드 펄스 (a) — 나중
- Unity 새 컴포넌트: `Assets/Scripts/NpcChat/PropagationGraphView.cs`
- 5 NPC 노드를 작은 패널에 **고정 좌표**로 배치 + `relations.yaml` 엣지 정적 표시.
- `GameTimeController.OnTickCompleted` 구독 → events 중 `player_origin==true`인 `from→to` 엣지를 펄스(색/굵기 0.5~1s 애니메이션).
- 기존 `NpcEventToast` / `GameTimeController` 패턴 재활용. 토글 키로 표시.

## 데이터 흐름
```
플레이어 채팅 → _save_player_turn(imp 9, player:True)
  → [time_advance] → simulator.tick → events(player_origin)
      → (WS)     → Unity 5노드 펄스
      → (스크립트) → matplotlib cascade 그림
```

## 성공 기준
- **스크립트**: `"곰 봤어요"` 주입 후 그림에 주입NPC→이웃→… 도달 경로가 한눈에 보인다.
- **게임**: 시간을 넘기면 플레이어發 정보가 퍼진 엣지가 펄스로 보인다.
- **외부인이 설명 없이** "플레이어 말이 마을에 퍼졌다"를 이해한다.

## 구현 순서
1. 백엔드 필드 노출 (공통 토대, 작음)
2. 재현 스크립트 + 그림 (안전빵 — 여기까지면 발표 그림 확보)
3. Unity 펄스 (시간 남으면)

## 리스크 / 메모
- 스크립트 그림을 먼저 → 게임 펄스를 못 끝내도 발표 그림은 확보된다.
- 펄스는 NPC 5명 고정이라 그래프 레이아웃 비용이 없다. 주 작업은 가독성(엣지 겹침) 다듬기.
- importance 9 전파 보장은 확인됨. **단 질문문은 전파 안 됨** — 시연 때 평서문("~봤어요")으로 말할 것.
