"""시드 메모리를 ChromaDB에 적재."""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryEntry, MemorySource, MemoryStore  # noqa: E402

CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]
SEED_PATH = ROOT / "data" / "seed" / "memories.yaml"
CHROMA_DIR = ROOT / "data" / "chroma"


def seed_one(character, entries_yaml, reset=False):
    store = MemoryStore(npc_name=character, base_dir=CHROMA_DIR / character)
    if reset:
        store.reset()
    if store.count() > 0 and not reset:
        print(f"  {character}: 이미 {store.count()}개 있음 (--reset으로 초기화 가능)")
        return

    now = datetime.now(timezone.utc)
    entries = []
    for i, item in enumerate(entries_yaml):
        entries.append(MemoryEntry(
            id=f"seed_{character}_{i:03d}_{uuid.uuid4().hex[:6]}",
            text=item["text"],
            importance=item.get("importance", 5),
            timestamp=now,
            source=MemorySource.SEED,
        ))
    store.add_many(entries)
    print(f"  {character}: {len(entries)}개 시드 추가 (총 {store.count()}개)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", choices=CHARACTERS, default=None)
    parser.add_argument("--reset", action="store_true", help="기존 메모리 초기화 후 적재")
    args = parser.parse_args()

    with SEED_PATH.open(encoding="utf-8") as f:
        seeds = yaml.safe_load(f)

    targets = [args.char] if args.char else CHARACTERS

    print(f"시드 적재 시작 (reset={args.reset})")
    for char in targets:
        if char not in seeds:
            print(f"  {char}: 시드 없음, skip")
            continue
        seed_one(char, seeds[char], reset=args.reset)
    print("끝.")


if __name__ == "__main__":
    main()
