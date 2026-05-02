"""raw JSONL을 chat 메시지 포맷으로 변환."""

import argparse
import json
from pathlib import Path

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_src"] = path.name
            row["_line"] = i
            rows.append(row)
    return rows


def _to_processed(row):
    """raw row → processed (chat messages 포맷). 단일턴/멀티턴 모두 지원."""
    cat = row.get("category", "unknown")
    if "messages" in row:
        # 이미 messages 포맷 (멀티턴)
        return {"messages": row["messages"], "category": cat}
    # 단일턴 input/output 포맷
    user_text = row.get("input", "").strip()
    asst_text = row.get("output", "").strip()
    if not user_text or not asst_text:
        return None
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ],
        "category": cat,
    }


def convert_one(character):
    src = RAW_DIR / f"{character}.jsonl"
    dst = PROCESSED_DIR / f"{character}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"원본 없음: {src}")

    # 모든 데이터 소스: 원본 + RAG 증강 + 캐주얼 증강 + 멀티턴
    suffixes = ["", "_rag", "_casual", "_multiturn"]
    rows = []
    for suffix in suffixes:
        path = RAW_DIR / f"{character}{suffix}.jsonl"
        if path.exists():
            rows.extend(_load_rows(path))

    counts = {}
    n = 0
    with dst.open("w", encoding="utf-8") as fout:
        for row in rows:
            sample = _to_processed(row)
            if sample is None:
                print(f"  skip {row['_src']}:{row['_line']} (빈값)")
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            cat = sample["category"]
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
