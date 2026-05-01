"""LoRA + 메모리 통합 E2E 테스트.

데모 시나리오:
  1. 시드 메모리만 있는 elias에게 질문 → 시드 기반 응답
  2. 새로운 사건을 elias 메모리에 추가
  3. 같은 질문 → 새 사건이 회상되는지 확인
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryEntry, MemorySource  # noqa: E402
from src.memory.chat import NpcChat  # noqa: E402

CHROMA_DIR = ROOT / "data" / "chroma"
ADAPTERS_DIR = ROOT / "output" / "adapters"


def hr(label=""):
    print("\n" + "=" * 60)
    if label:
        print(label)
        print("=" * 60)


def show(npc, user_text, response, memories):
    print(f"\n[유저] {user_text}")
    if memories:
        print("[검색된 기억]")
        for m in memories:
            print(f"  · ({m['importance']:>2}) {m['text']}")
    else:
        print("[검색된 기억] (없음)")
    print(f"[{npc}] {response}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char", default="elias")
    parser.add_argument("--query", default="용 사냥꾼 얘기 들었어요?")
    parser.add_argument(
        "--inject",
        default=None,
        help="새 메모리 추가 후 다시 질문 (회상 검증용)",
    )
    args = parser.parse_args()

    adapter_path = ADAPTERS_DIR / args.char
    if not adapter_path.exists():
        print(f"어댑터 없음: {adapter_path}")
        return 1

    print(f"NPC: {args.char}, 어댑터 + 메모리 로딩...")
    chat = NpcChat(
        npc_name=args.char,
        adapter_dir=adapter_path,
        chroma_dir=CHROMA_DIR,
    )
    print(f"메모리 {chat.store.count()}개 보유")

    hr("[1] 시드 메모리만 있는 상태")
    resp, mems = chat.respond(args.query, return_memories=True)
    show(args.char, args.query, resp, mems)

    if args.inject:
        hr("[2] 새 사건 주입")
        new_entry = MemoryEntry(
            id=f"event_{uuid.uuid4().hex[:8]}",
            text=args.inject,
            importance=8,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.PROPAGATION,
        )
        chat.store.add(new_entry)
        print(f"주입: {args.inject}")
        print(f"현재 메모리 {chat.store.count()}개")

        hr("[3] 같은 질문 다시")
        resp2, mems2 = chat.respond(args.query, return_memories=True)
        show(args.char, args.query, resp2, mems2)


if __name__ == "__main__":
    main()
