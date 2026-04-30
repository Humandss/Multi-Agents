# ai

게임 NPC 대화용 LoRA 학습 + 추론.

## 폴더

```
configs/training.yaml   학습 설정
data/raw/               원본 JSONL (캐릭터당 1개)
data/processed/         chat 포맷 변환 결과
scripts/                실행 스크립트
output/adapters/{char}/ 학습된 LoRA
```

## 세팅

```bash
cd ai
python -m uv sync
uv run hf auth login        # HF 토큰 입력
uv run python scripts/verify_setup.py
```

EXAONE 모델 페이지: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct

## 학습

```bash
uv run python scripts/preprocess.py
uv run python scripts/train_lora.py --char elias --epochs 3
uv run python scripts/test_inference.py --char elias
```

5종 일괄 학습:
```bash
uv run python scripts/train_lora.py --char all
```

대화형 모드:
```bash
uv run python scripts/test_inference.py --char elias --interactive
```

## VRAM 부족 시

`configs/training.yaml`에서:
- `lora.r: 16`
- `train.per_device_batch_size: 1`
- `train.max_seq_length: 512`
