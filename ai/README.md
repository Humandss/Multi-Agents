# ai

게임 NPC 대화: 페르소나 LoRA + 메모리 + 정보 전파 + WebSocket 추론 서버.

## 폴더

```
configs/                 학습/관계 그래프 설정
data/raw/                원본 JSONL (캐릭터당 1개) + RAG 증강
data/processed/          chat 포맷 변환 결과 (gitignore)
data/seed/memories.yaml  시드 메모리
data/chroma/             ChromaDB 영속화 (gitignore)
src/memory/              메모리 시스템 (store, retriever, chat)
src/propagation/         정보 전파 시뮬레이션
src/server/              FastAPI WebSocket 서버
scripts/                 실행 스크립트
output/adapters/{char}/  학습된 LoRA (gitignore)
```

## 세팅

```bash
cd ai
python -m uv sync
uv run hf auth login        # HF 토큰
uv run python scripts/verify_setup.py
```

EXAONE 모델: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct

## 학습

```bash
uv run python scripts/preprocess.py
uv run python scripts/train_lora.py --char elias --epochs 3
uv run python scripts/test_inference.py --char elias
```

5종 일괄:
```bash
uv run python scripts/train_lora.py --char all
```

RAG-aware 학습 데이터 증강 (베이스 EXAONE으로 자동 생성):
```bash
uv run python scripts/generate_rag_data.py --n 30
uv run python scripts/preprocess.py        # _rag.jsonl도 자동 합침
uv run python scripts/train_lora.py --char all
```

## 메모리 시스템

```bash
uv run python scripts/seed_memory.py        # 시드 적재
uv run python scripts/seed_memory.py --reset
uv run python scripts/test_memory_chat.py --char elias
uv run python scripts/test_memory_chat.py --char elias --inject "..." --query "..."
```

## 정보 전파 시뮬레이션

```bash
uv run python scripts/run_simulation.py --days 5
uv run python scripts/run_simulation.py --inject-to mathilda --inject "..." --save-events output/sim.json
```

## WebSocket 서버 (Unity 통합용)

```bash
uv run python scripts/run_server.py --port 8000
```

다른 터미널에서:
```bash
uv run python scripts/test_server.py                  # 5종 NPC 한 번씩
uv run python scripts/test_server.py --interactive --char elias
```

### 프로토콜

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

## VRAM 부족 시

`configs/training.yaml`에서:
- `lora.r: 16`
- `train.per_device_batch_size: 1`
- `train.max_seq_length: 512`
