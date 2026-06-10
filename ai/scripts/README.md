# scripts/ — 실행 스크립트 상세

각 스크립트의 동작·입출력·내부 흐름 정리.
**차별화 핵심**(전파/자율대화)을 직접 다루는 스크립트는 ★ 표시 + 깊이 있게 설명.

분류:
- 🟢 운영 — 시스템 구동
- 🔵 검증 — 서버 켠 상태에서 HTTP로 기능 확인
- 🟣 발표 자료 — 그래프/시각화 (GPU 불필요)

| 스크립트 | 분류 | 서버 필요 | GPU | 차별화 핵심 |
|------|------|------|------|------|
| `run_server.py` | 🟢 | 자기가 서버 | ✅ | |
| `seed_memory.py` | 🟢 | ❌ | (임베딩) | |
| `verify_setup.py` | 🟢 | ❌ | ✅ | |
| `test_compare.py` | 🔵 | ✅ | (서버) | |
| `test_npc_conversation.py` | 🔵 | ✅ | (서버) | ★ 자율 대화 |
| `test_tick_http.py` | 🔵 | ✅ | (서버) | ★ 전파+자율대화 |
| `demo_player_cascade.py` | 🟣 | ❌ | ❌ | ★★ 전파 시각화 |

---

## 🟢 운영 스크립트

### `run_server.py` — 서버 런처

```bash
python -m uv run python scripts/run_server.py            # 기본 (reset + prime 3)
python -m uv run python scripts/run_server.py --no-reset # 메모리 유지
python -m uv run python scripts/run_server.py --prime 0  # tick 자동실행 끔
```

**역할:** FastAPI 서버를 띄우는 진입점. 직접 추론하지 않고 환경변수만 설정해 uvicorn에 위임.

**내부 흐름:**
```
1. argparse 파싱 (--host --port --no-reset --prime)
2. 환경변수 설정:
   - reset 안 끄면 → NPC_RESET_ON_START=1
   - --prime N    → NPC_PRIME_TICKS=N
3. uvicorn.run("src.server.app:app")  ← app.py가 실제 서버
```

**app.py가 환경변수 읽고 하는 일:**
- `NPC_RESET_ON_START=1` → ChromaDB 초기화 + 시드 49개 재적재
- `NPC_PRIME_TICKS=3` → 시작 시 3 tick 자동 실행 (Day 3 상태로)

| 옵션 | 효과 |
|------|------|
| (기본) | reset + 시드재적재 + 3 tick prime |
| `--no-reset` | 기존 메모리 유지 |
| `--prime N` | N tick 자동 (`0`이면 끔) |
| `--port` | 포트 지정 |

**주의:** 모델 로딩 ~1분 + prime ~30초. `start_server.bat`이 이걸 호출.

---

### `seed_memory.py` — 시드 메모리 적재

```bash
python -m uv run python scripts/seed_memory.py --reset
python -m uv run python scripts/seed_memory.py --char elias --reset
```

**역할:** `data/seed/memories.yaml` → ChromaDB 단방향 적재.

**내부 흐름:**
```
1. memories.yaml 로드 (NPC별 9-11개)
2. NPC 순회:
   ├─ MemoryStore(npc) — ChromaDB 연결 + BGE-M3 임베딩
   ├─ [--reset] 컬렉션 초기화
   ├─ 이미 메모리 있으면 skip (중복 방지)
   └─ MemoryEntry(source=SEED) → add_many() (벡터 변환 후 저장)
3. "elias: 11개 추가" 출력
```

**언제:** 보통 직접 안 씀 (run_server `--reset`이 동일 작업). 시드만 갈아끼울 때.

---

### `verify_setup.py` — 환경 점검

```bash
python -m uv run python scripts/verify_setup.py
```

**역할:** GPU/CUDA/양자화 단계별 검사. 모델 로딩 없이 5초.

**내부 흐름 (순차, 실패 시 즉시 중단):**
```
1. Python/PyTorch 버전
2. torch.cuda.is_available()  ──FAIL──► 종료
3. GPU 이름 + VRAM
4. bf16 지원
5. matmul 실연산 (1024² bfloat16) — CUDA 작동
6. bitsandbytes Linear4bit — 양자화 작동
7. transformers/peft import
8. "전부 OK" 또는 실패 단계
```

**언제:** 새 PC 세팅 후, 모델 안 뜰 때 진단.

---

## 🔵 검증 스크립트 (서버 켠 상태)

### `test_compare.py` — 5종 NPC 페르소나 비교

```bash
python -m uv run python scripts/test_compare.py --text "광산 얘기 들었어요?"
python -m uv run python scripts/test_compare.py --interactive
```

**역할:** 같은 질문을 5 NPC에 → 페르소나별 응답 비교.

**내부 흐름:**
```
1. POST /compare {text} ──HTTP(urllib)──► 서버
2. 서버: for npc in 5종 → engine.respond(npc, text)
3. 5개 응답 ──► 스크립트
4. print_responses(): NPC 색상별 출력
   elias (7867ms): 흠... 그 사건이 마음에 걸리오.
       ♥ 친밀도 +1 → 31/100 (지인)
       ↳ [seed imp=8] 광산 사고 22명...
       ★ QUEST: 광산 조사
```

**출력 정보:** 응답 + 친밀도 변화 + 회상 메모리(source/importance) + quest.

**언제:** Unity 없이 페르소나 차이 빠른 확인. 발표 시연 "같은 사실, 5종 다른 표현".

---

### ★ `test_npc_conversation.py` — NPC 자율 대화 (차별화 핵심)

```bash
python -m uv run python scripts/test_npc_conversation.py --a mathilda --b finn
python -m uv run python scripts/test_npc_conversation.py --random
python -m uv run python scripts/test_npc_conversation.py --num_turns 4
```

**역할:** 두 NPC가 자기들끼리 대화하게 트리거 (Park et al. Agent-Agent Conversation).

**내부 흐름:**
```
1. URL 구성:
   --random → POST /simulate_random?num_turns=N
   --a --b  → POST /simulate/{a}/{b}?num_turns=N
2. 서버: engine.simulate_conversation(a, b, N):
   ├─ ① 화제 선정 — a의 ChromaDB에서 "마을 사건 소식" 검색
   │      (시드/전파/플레이어발화/통찰 중 무작위)
   ├─ ② a가 화제 던짐 (EXAONE 생성)
   ├─ ③ 번갈아 대화 (num_turns×2회, 각자 EXAONE)
   │      서로의 history 공유하며 응답
   └─ ④ 대화 전체를 양쪽 ChromaDB에 저장 (source=CONVERSATION)
3. turns[] ──► 스크립트
4. print_conversation(): NPC 색상별 대화
   마틸다: 어머, 핀! 광산 곰 얘기 들었어요?
   핀: 오 그대여, 영웅의 시간이...
   → 양쪽 메모리 저장됨
```

**★ 왜 차별화 핵심인가:**
- **화제가 DB에서** 나옴 → 플레이어 발화/전파된 정보가 자율 대화 주제가 됨
- **대화 결과가 양쪽 메모리에** 저장 → 다음에 그 NPC와 대화하면 회상됨
- 즉 **자율 대화 = 또 다른 정보 전파 경로** (플레이어가 한 말 → A 기억 → 대화 → B 기억)
- 일반 LLM NPC는 플레이어가 말 걸어야만 반응하지만, 이건 **NPC끼리 능동적으로** 정보 교환

**언제:** 자율 대화가 페르소나 유지하는지 검증. 발표 핵심 시연.

**주의:** `num_turns×2×5초` 예상. `--topic`은 현재 미지원(자동 선정).

---

### ★ `test_tick_http.py` — 시간 진행 통합 (전파 + 자율대화)

```bash
python -m uv run python scripts/test_tick_http.py            # 전파 + 대화 + 통찰
python -m uv run python scripts/test_tick_http.py --no_conversation  # 전파만
```

**역할:** N키와 동일한 `/tick`을 HTTP로. **차별화 3종(전파·자율대화·reflection)이 한 번에** 일어나는 걸 확인.

**내부 흐름:**
```
1. POST /tick?num_turns=N ──HTTP──► 서버
2. 서버 engine.tick():
   ├─ ① PropagationSimulator.tick()  — 전파 (관계 그래프 따라 메모리 확산)
   ├─ ② simulate_conversation()       — 무작위 1쌍 자율 대화
   ├─ ③ reflect()                     — rotating 1 NPC 통찰 추출
   └─ ④ prune()                       — 메모리 정리
3. {events, turns, memory_counts} ──► 스크립트
4. print_tick(): 3섹션
   [전파] 25개 이벤트 (from→to, importance 변화, 원문→변형)
   [자율대화] mathilda ↔ finn (화제 + turns)
   [메모리 카운트] elias: 45 ...
```

**★ 전파 출력 상세:**
```
[전파] hermann → mathilda  imp 6→6
    원: 검 재료 부족...                    ← sender 원본 기억
    변: 재료 부족이라니, 산 너머 광산...   ← receiver가 받은 형태 ("X한테 들었다")
```
→ importance 변화(페르소나 보정), 원문→변형(출처 보존) 확인 가능.

**언제:** Unity 없이 **시간 진행 전체 결과** 확인. `--no_conversation`이면 전파만 (빠름).

---

## 🟣 발표 자료 (GPU 불필요)

### ★★ `demo_player_cascade.py` — 전파 cascade 그래프 (차별화 시각화)

```bash
python -m uv run python scripts/demo_player_cascade.py \
  --utterance "내 이름은 반욱현이야" --inject-to elias --spread 0.35 --days 7
```

**역할:** 플레이어 발화 한 마디가 며칠에 걸쳐 마을로 퍼지는 **전파 경로를 그림 한 장**으로. 발표 정량 자료.

**내부 흐름:**
```
1. chroma_demo DB 생성 (실제 게임 DB와 분리 — 안전)
2. _reseed(): 5 NPC 시드 재적재
3. inject-to NPC에 플레이어 발화 주입 (player_origin=True)
4. relations.yaml 로드 → freq를 --spread로 덮어씀 (단계적 cascade용)
5. PropagationSimulator(use_transform=False)  ← LLM/GPU 불필요!
6. for day in 1..days:
   ├─ sim.tick(day)
   ├─ player_origin 이벤트만 추적
   ├─ first_day[npc] = 첫 도달 시점
   └─ edges = (from, to, day) 경로
   "Day 1: 도달 2/4명" ...
7. matplotlib:
   ├─ X축 = Day, Y축 = NPC
   ├─ 빨강 = 시작점, 파랑 = 전파 도달
   └─ 화살표 = 전파 경로 (누가 누구에게)
8. output/figures/player_cascade.png 저장
```

**★★ 왜 가장 강력한 발표 자료인가:**
- **실제 게임과 같은 `PropagationSimulator`** 사용 — 데모용 가짜 로직 아님
- 차이는 freq뿐: 게임 0.9(하루에 전원) vs 데모 0.35(단계적으로 보이게)
- → "같은 알고리즘을, 관찰하기 쉽게 느린 파라미터로 돌린 것" (정직)
- **GPU/LLM 불필요** → 전파 로직만 떼어내 빠르게 (수초)
- **재현 가능** (`--seed 42` 고정)
- 정량 시각화 = "측정했다"는 연구자 인상 (다른 학생들의 "만들었다"와 차별)

**파라미터 의미:**
| 옵션 | 의미 |
|------|------|
| `--utterance` | 플레이어 발화 내용 |
| `--inject-to` | 누구에게 처음 말하나 |
| `--spread` | 전파 확률 (0.35=단계적, 0.9=즉시 전원) |
| `--days` | 며칠 시뮬 |
| `--seed` | 난수 고정 (재현용) |

**발표 활용:**
```
[그래프 1] "광장에서 곰을 봤어요" — mathilda 시작
[그래프 2] "내 이름은 반욱현이야" — elias 시작
→ "어떤 정보든, 누구에게 말하든, 마을로 퍼진다"
```

---

## 차별화 핵심 요약 (발표용)

| 스크립트 | 보여주는 차별점 |
|------|------|
| `demo_player_cascade.py` | **전파** — 발화가 N일에 걸쳐 마을 확산 (정량 그래프) |
| `test_npc_conversation.py` | **자율 대화** — NPC끼리 능동적 정보 교환 |
| `test_tick_http.py` | **통합** — 전파+자율대화+reflection이 한 tick에 |

이 셋이 "정적 RAG 챗봇"이 아닌 **"동적 멀티 에이전트 메모리"** (Park et al.)임을 보여주는 핵심 도구.

---

## 실행 플로우 (실전 순서)

```
[1회성]
  verify_setup.py        환경 OK 확인
       ↓
[시연/개발]
  run_server.py          서버 띄움 (90%)
       ↓
  [터미널 검증]              [Unity 플레이]
  test_compare           ←─┐  F키 대화
  test_tick_http         ←─┤  N키 시간진행
  test_npc_conversation  ←─┘
       ↓
[발표 자료 (서버 끄고도)]
  demo_player_cascade    cascade 그래프
```
