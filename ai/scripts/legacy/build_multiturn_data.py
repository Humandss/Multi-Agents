"""multiturn_seeds.yaml을 학습용 JSONL로 변환.

LLM 증강 없이 그대로 사용 (멀티턴 시퀀스는 직접 작성한 게 더 안정적).
출력은 ai/data/raw/{character}_multiturn.jsonl.

사용:
    uv run python scripts/build_multiturn_data.py
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed" / "multiturn_seeds.yaml"
RAW_DIR = ROOT / "data" / "raw"


def main():
    with SEED_PATH.open(encoding="utf-8") as f:
        all_seeds = yaml.safe_load(f)

    for character, sequences in all_seeds.items():
        out_path = RAW_DIR / f"{character}_multiturn.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for seq in sequences:
                row = {
                    "character": character,
                    "category": seq.get("category", "multiturn"),
                    "messages": seq["turns"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{character}: {len(sequences)}개 시퀀스 → {out_path}")


if __name__ == "__main__":
    main()
