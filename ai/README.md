# ai — 한국어 게임 NPC 대화 시스템

EXAONE 3.5 7.8B (로컬 4bit) + ChromaDB RAG + NPC 간 정보 전파 + 자율 대화 + Reflection.
Park et al. *Generative Agents* (2023) framework를 Unity 게임 NPC에 적용.

> **LoRA → prompting 전환**: LoRA fine-tuning이 측정 결과 baseline prompting에 모든 dialogue 차원에서 짐 (n=450 + n=315턴). LoRA 비활성 + system prompt 기반으로 전환. LoRA 학습/평가 코드는 `scripts/legacy/`, `legacy/`에 보존 (ablation 재현용).

---

## 핵심 기능

| 기능 | 설명 |
|------|------|
| **5종 NPC 페르소나** | elias(마법사), hermann(대장장이), mathilda(술집주인), finn(음유시인), bernhardt(잡화상). system prompt + 어조 후처리. |
| **ChromaDB RAG** | NPC별 메모리 컬렉션 + BGE-M3 한국어 임베딩 + 가중 검색(유사도·중요도·최신성). |
| **정보 전파** | 관계 그래프 기반. 플레이어 발화가 NPC 사이로 확산. |
| **NPC-NPC 자율 대화** | 두 NPC가 자기들끼리 대화 → 양쪽 메모리에 저장 (Park et al.). |
| **Reflection** | 누적 메모리 → LLM이 추상 통찰 추출 (Park et al.). |
| **친밀도(Trust)** | 0-100, 4등급. 대화/quest로 변동. 등급별 greeting. |
| **Quest** | NPC별 quest pool. trust 조건 충족 시 자동 제안. |
| **파이프라인 로그** | 발화→저장→전파→언급 + 친밀도/통찰/자율대화 색상 로그. |
| **망각곡선 + 회상 강화** | Ebbinghaus 지수 감쇠(SAGE 계열): 일화 기억만 감쇠, 회상될수록 반감기 성장(`recall_count`). seed·reflection은 면제. prune도 retention 낮은 것부터. |
| **인식론적 기억 타입** | Hindsight 계열: 회상 프롬프트에 (직접 들음/전해 들은 소문/나의 생각/아는 사실) 라벨 → NPC가 확신도를 다르게 표현 ("~라더라" vs 확신). |

---

## 폴더 구조

```
src/                      ★ 메인 뼈대 (실제 시스템)
  memory/                 RAG (저장 + 검색)
    schema.py             메모리 데이터 구조 (MemoryEntry, MemorySource)
    store.py              ChromaDB 저장소 (NPC당 1 컬렉션)
    retriever.py          가중 검색 (Park et al. 점수 공식)
    chat.py               회상 → 프롬프트 결합
  propagation/            정보 전파
    graph.py              NPC 관계 그래프
    simulator.py          전파 시뮬레이션 (tick)
  server/                 추론 + API
    engine.py             NpcServer — 핵심 두뇌 (추론·메모리·전파·trust·quest)
    app.py                FastAPI 엔드포인트
    pipeline_log.py       파이프라인 로그

scripts/                  실행/검증 스크립트 (아래 표 참조)
  legacy/                 LoRA 학습·평가 코드 (폐기, 보존)

configs/relations.yaml    NPC 관계 그래프 (전파 빈도)
data/seed/memories.yaml   시드 메모리 (NPC당 9-11개)
data/eval/test_prompts.yaml  페르소나 정의 (system prompt 재료)
data/chroma/              ChromaDB 영속화 (gitignore)
output/figures/           cascade 그래프 출력 (gitignore)
legacy/                   LoRA 학습 데이터/어댑터/평가 모듈 (폐기, 보존)
```

---

## 세팅

```bash
cd ai
python -m uv sync
python -m uv run huggingface-cli login         # HF 토큰 (EXAONE 다운로드)
python -m uv run python scripts/verify_setup.py  # GPU/CUDA/bnb 점검
```

EXAONE: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
추론 VRAM: EXAONE 7.8B 4bit ≈ 5-6GB (8GB GPU 작동).

---

## 서버 실행 (메인 워크플로우)

```bash
python -m uv run python scripts/run_server.py    # 기본: 자동 reset + 3 tick prime
# 또는 start_server.bat 더블클릭
```

옵션:
| 옵션 | 효과 |
|------|------|
| (기본) | ChromaDB 초기화 + 시드 재적재 + 3 tick 자동 실행 |
| `--no-reset` | 기존 메모리 유지 (이어서 시연) |
| `--prime N` | 시작 시 N tick 자동 (`--prime 0`이면 끔) |
| `--port 8000` | 포트 지정 |

서버 뜨면 Unity에서 WebSocket 연결, 또는 아래 검증 스크립트 사용.

---

## 스크립트

> **각 스크립트 상세 (입출력·내부 흐름·차별화 핵심)**: [`scripts/README.md`](scripts/README.md)

### 운영
| 스크립트 | 동작 | GPU |
|------|------|-----|
| `run_server.py` | FastAPI 서버 런처 (`src.server.app` 가동). reset/prime 환경변수 전달. | ✅ |
| `seed_memory.py` | `memories.yaml` → ChromaDB 적재. `--char`, `--reset`. | (임베딩) |
| `verify_setup.py` | CUDA/bf16/bitsandbytes/transformers 단계별 점검. | ✅ |

### 검증 (서버 켠 상태, HTTP 호출)
| 스크립트 | 동작 | 차별화 |
|------|------|------|
| `test_compare.py` | 같은 질문을 5 NPC에 → 페르소나 비교 (trust/quest/회상). `--interactive`. | |
| `test_npc_conversation.py` | 두 NPC 자율 대화. `--a --b`, `--random`, `--num_turns`. | ★ 자율 대화 |
| `test_tick_http.py` | 시간 진행(`/tick`) — 전파+자율대화+reflection 통합. `--no_conversation`. | ★ 전파+대화 |

### 발표 자료 (GPU 불필요)
| 스크립트 | 동작 | 차별화 |
|------|------|------|
| `demo_player_cascade.py` | 플레이어 발화 전파 cascade 그래프. 실제 게임과 같은 `PropagationSimulator`(freq만 조정). | ★★ 전파 시각화 |

```bash
# 예: "내 이름은 반욱현" 발화가 7일에 걸쳐 마을로 퍼지는 그래프
python -m uv run python scripts/demo_player_cascade.py \
  --utterance "내 이름은 반욱현이야" --inject-to elias --spread 0.35 --days 7
```

---

## API 엔드포인트

| 메서드 | 경로 | 용도 |
|------|------|------|
| WS | `/ws/{npc}` | Unity 실시간 대화 |
| GET | `/healthz` | 서버 상태 + NPC 목록 |
| GET | `/npcs` | NPC별 메모리 수 |
| POST | `/compare` | 5 NPC 비교 (`{text}`) |
| POST | `/tick` | 시간 진행 (전파+자율대화+reflection) |
| POST | `/simulate/{a}/{b}` | 두 NPC 자율 대화 |
| POST | `/reflect/{npc}` | reflection 강제 실행 |
| POST | `/quest_complete/{npc}` | quest 완료 → trust +10 |
| GET | `/trust` | 전체 NPC 친밀도 |
| GET | `/quests` | quest pool + 상태 |
| GET | `/pipeline/trace?n=50` | 파이프라인 로그 (발화→저장→전파→언급) |
| POST | `/debug/prime?ticks=3` | N tick 자동 실행 |
| GET | `/memory/player/{npc}` | NPC가 가진 플레이어 메모리 |
| GET | `/memory/reflections/{npc}` | NPC 통찰 메모리 |
| POST | `/memory/reset` | 메모리 초기화 + 시드 재적재 |
| GET | `/docs` | Swagger UI (브라우저 테스트) |

### WebSocket 프로토콜

```jsonc
// 클라이언트 → 서버
{"type": "chat", "text": "안녕하세요"}
{"type": "time_advance"}                   // 시간 진행 (N키)

// 서버 → 클라이언트
{"type": "ready", "npc": "Elias"}          // 연결 직후
{"type": "response", "npc": "Elias", "text": "흠, 무슨 일이오?",
 "memories_used": [{"text": "...", "importance": 8, "source": "seed"}],
 "trust": 32, "trust_label": "지인", "trust_delta": 2, "latency_ms": 1234}
{"type": "tick_events", "day": 1, "events": [...], "turns": [...]}  // 시간 진행 결과
{"type": "error", "message": "..."}
```

---

## 핵심 데이터 흐름

```
[대화]  Unity ──WS──► app.py /ws/{npc} ──► engine.respond()
          ① 발화 로그
          ② retriever.search() → store.query() (ChromaDB)  →  chat.build_user_prompt()
          ③ EXAONE.generate() → _clean_response() → _restore_player_name()
          ④ _save_player_turn() → store.add()  +  trust 업데이트  +  언급 로그

[시간진행]  N키 ──► engine.tick()
          ③ PropagationSimulator.tick()  (전파)
          ○ simulate_conversation()      (자율 대화)
          * reflect()                    (추상 통찰)
            store.prune()                (메모리 정리)
```

---

## Reflection 동작 (Park et al.)

매 tick 1 NPC씩 (rotating):
1. 최근 25개 메모리 (importance 합 80+ 시)
2. LLM에게 "핵심 통찰 3가지" 추출 요청
3. REFLECTION source, importance 9로 저장 → 이후 우선 회상

결과: NPC가 단편 사실 나열이 아닌 "마을 분위기"를 통합해 응답.

---

## 전파 동작

`relations.yaml`의 NPC 관계 그래프 기반 (freq = 만남 확률):
1. 매 tick 각 엣지에서 `random() < freq`면 만남
2. sender의 importance 7+ 메모리 → receiver에게 전달 (최대 2개)
3. 플레이어 정보는 `player_origin` 메타 보존 → 다른 NPC도 회상 가능
4. `chain_origin`으로 전파 사슬 추적, `shared_with`로 중복 방지

게임 freq는 0.9 (하루 만에 전원 도달). 데모(`demo_player_cascade.py`)는 freq를 낮춰 단계적 cascade 시각화.

---

## LoRA Ablation 재현 (보통 안 함)

LoRA 폐기 결정의 측정 근거. 코드는 `scripts/legacy/`, 어댑터는 `legacy/output/adapters/`에 보존.

```bash
# 평가 (legacy)
python -m uv run python scripts/legacy/eval_persona.py --baseline lora
python -m uv run python scripts/legacy/eval_multiturn_drift.py
```

자세한 결과: `memory/eval_iteration_findings.md` (사용자 메모리).
