"""raw JSONL을 chat 메시지 포맷으로 변환."""

import argparse
import json
from pathlib import Path

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def convert_one(character):
    src = RAW_DIR / f"{character}.jsonl"
    dst = PROCESSED_DIR / f"{character}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"원본 없음: {src}")

    counts = {}
    n = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_text = row.get("input", "").strip()
            asst_text = row.get("output", "").strip()
            cat = row.get("category", "unknown")
            if not user_text or not asst_text:
                print(f"  skip {src.name}:{i} (빈값)")
                continue
            sample = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": asst_text},
                ],
                "category": cat,
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[cat] = counts.get(cat, 0) + 1
            n += 1
    return n, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", choices=CHARACTERS, default=None)
    args = parser.parse_args()

    targets = [args.char] if args.char else CHARACTERS
    total = 0
    for char in targets:
        n, counts = convert_one(char)
        total += n
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{char}: {n}개 ({cat_str})")
    print(f"\n총 {total}개 → {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
