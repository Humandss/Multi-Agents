# ai

한국어 게임 NPC 대화 시스템: LLM (EXAONE 3.5 7.8B) + ChromaDB 메모리 + NPC 간 정보 전파 + Unity WebSocket.

> **Note**: LoRA fine-tuning은 측정 결과 baseline prompting에 모든 dialogue 차원에서 짐 (n=450 + n=315턴). LoRA 비활성 + system prompt 기반으로 전환. LoRA 코드/어댑터는 ablation 측정 재현성 위해 보존.

## 폴더 구조

```
configs/                 모델/관계 그래프 설정
data/seed/memories.yaml  시드 메모리 (NPC당 5-6개)
data/eval/               평가 데이터 (test_prompts, multiturn_scenarios)
data/raw/                LoRA 학습 데이터 원본 (deprecated)
data/processed/          전처리 결과 (gitignore, deprecated)
data/chroma/             ChromaDB 영속화 (gitignore)
src/memory/              메모리 시스템 (store, retriever, schema, chat)
src/propagation/         정보 전파 시뮬레이션
src/server/              FastAPI WebSocket 서버 (NpcServer)
src/eval/                평가 (persona_score, distortion, propagation_rate)
scripts/                 실행 스크립트 (active + deprecated 분류 → scripts/STATUS.md 참조)
output/adapters/{char}/  학습된 LoRA (gitignore, ablation 측정용 보존)
output/eval/             평가 결과 (persona_scores, propagation_eval, multiturn_drift)
```

## 세팅

```bash
cd ai
python -m uv sync
python -m uv run huggingface-cli login    # HF 토큰 (EXAONE 다운로드용)
python -m uv run python scripts/verify_setup.py
python -m uv run python scripts/seed_memory.py
```

EXAONE 모델: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct

## 서버 실행 (메인 워크플로우)

```bash
python -m uv run python scripts/run_server.py --port 8000
# 또는 start_server.bat
```

다른 터미널에서:
```bash
python -m uv run python scripts/test_server.py                  # 5종 NPC 한 번씩
python -m uv run python scripts/test_server.py --interactive --char elias
```

## 정보 전파 시뮬레이션

```bash
python -m uv run python scripts/run_simulation.py --days 5
```

## 평가

```bash
# 페르소나 점수 (n=450)
python -m uv run python scripts/eval_persona.py --char all --baseline all --n_per_category 5

# 멀티턴 페르소나 drift (n=315 turns)
python -m uv run python scripts/eval_multiturn_drift.py

# 정보 전파율 + 왜곡률 (LoRA vs prompting baseline)
python -m uv run python scripts/eval_propagation.py --baseline lora --days 7
python -m uv run python scripts/eval_propagation.py --baseline prompting --days 7
```

평가 결과는 `output/eval/` 에 저장.

### Prototocol (Unity ↔ Python)

엔드포인트:
- `GET  /healthz` 서버 상태
- `GET  /npcs` NPC별 메모리 수
- `WS   /ws/{npc_name}` 대화

WebSocket 메시지:
```jsonc
// 클라이언트 → 서버
{"type": "chat", "text": "안녕하세요"}

// 서버 → 클라이언트 (연결 직후)
{"type": "ready", "npc": "elias"}

// 서버 → 클라이언트 (응답)
{
  "type": "response",
  "npc": "elias",
  "text": "흠. 무슨 일이오?",
  "memories_used": [
    {"text": "...", "importance": 8, "source": "seed"}
  ],
  "latency_ms": 1234
}

// 서버 → 클라이언트 (에러)
{"type": "error", "message": "..."}
```

## LoRA Ablation (재현 시)

LoRA 어댑터는 보존되어 있어 측정 재현 가능:

```bash
# LoRA 학습 (재실행 필요시 — 보통 안 함)
python -m uv run python scripts/preprocess.py
python -m uv run python scripts/train_lora.py --char all --epochs 5

# LoRA 활성 모드로 서버 실행 (NpcServer use_lora=True)
# scripts/run_server.py 수정 또는 직접 NpcServer(use_lora=True) 호출

# Eval에서 lora baseline 비교
python -m uv run python scripts/eval_persona.py --baseline lora
```

자세한 측정 결과 + 결정 근거: `memory/eval_iteration_findings.md` (사용자 메모리)

## VRAM 부족 시

`configs/training.yaml`에서 (LoRA 재학습 시만):
- `lora.r: 16`
- `train.per_device_batch_size: 1`
- `train.max_seq_length: 512`

추론 (서버) VRAM:
- 8GB GPU에서 EXAONE 7.8B 4-bit 양자화로 작동 (~5-6GB 점유)
