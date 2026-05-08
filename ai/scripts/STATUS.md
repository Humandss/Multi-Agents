# Scripts 사용 상태

LoRA 폐기 결정 (2026-05-03) 후 active vs deprecated 분류.

## ✅ Active — 메인 워크플로우

| 스크립트 | 용도 |
|---|---|
| `run_server.py` | FastAPI + WebSocket 추론 서버 (메인 entry point) |
| `seed_memory.py` | ChromaDB에 시드 메모리 적재 (1회) |
| `verify_setup.py` | CUDA/bnb/transformers 동작 확인 |
| `run_simulation.py` | 정보 전파 시뮬레이션 (CLI) |

## ✅ Active — 평가 인프라

| 스크립트 | 용도 |
|---|---|
| `eval_persona.py` | 페르소나 일관성 평가 (3 baseline × 5 NPC) |
| `eval_propagation.py` | 정보 전파율 + 왜곡률 (LoRA vs prompting baseline 비교) |
| `eval_multiturn_drift.py` | 멀티턴 페르소나 drift 측정 |

## ✅ Active — 테스트

| 스크립트 | 용도 |
|---|---|
| `test_server.py` | WebSocket 서버 연결 테스트 (5 NPC + interactive 모드) |
| `test_tick.py` | propagation tick 테스트 |
| `test_memory_chat.py` | 메모리 + chat 통합 테스트 |

## ⚠️ Deprecated — LoRA 학습 관련 (ablation 측정 재현 시만)

| 스크립트 | 용도 | 상태 |
|---|---|---|
| `train_lora.py` | LoRA 학습 (캐릭터당 ~12분 × 5 = ~1시간) | 보존: ablation 재현 가능 |
| `preprocess.py` | data/raw → data/processed 전처리 | LoRA 학습 시만 필요 |
| `build_multiturn_data.py` | 멀티턴 학습 데이터 생성 | LoRA 학습 시만 |
| `generate_casual_data.py` | 캐주얼 데이터 자동 생성 | LoRA 학습 시만 |
| `generate_rag_data.py` | RAG-aware 데이터 자동 생성 | LoRA 학습 시만 |
| `test_inference.py` | LoRA 어댑터 standalone 추론 테스트 | LoRA 활성 시만 작동 |
| `test_casual.py` | LoRA 캐주얼 응답 테스트 | LoRA 활성 시만 |
| `test_multiturn.py` | LoRA 멀티턴 응답 테스트 | LoRA 활성 시만 |

## 폐기 결정 근거 (요약)

측정 결과 (n=450 페르소나 + n=315턴 multi-turn):
- 단발 페르소나: prompting 4.12 > LoRA 3.62
- Multi-turn drift: 모든 턴에서 prompting > LoRA > full
- 정보 전파: LoRA 약간 우위 (속도 +50%, 일관성)

→ Dialogue 품질 우선 + 시스템 단순화로 **LoRA 비활성**.
→ LoRA 코드/어댑터/데이터는 **ablation 측정 재현성 위해 보존**.

자세한 측정 + 결정 근거: 사용자 메모리 `eval_iteration_findings.md`
