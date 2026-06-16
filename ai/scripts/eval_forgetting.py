"""망각 곡선 시뮬레이션 — 기억이 시간에 따라 잊히는 과정을 로그(표)로 출력.

실제 retriever.retention()과 동일한 함수를 사용하므로 데모와 일치한다.
모델/GPU 불필요, 즉시 실행 → 콘솔 표를 그대로 보고서 그림으로 캡처 가능.

retention = exp(-Δd / (3.0 × (1 + recall_count)))
  · 일화 기억(dialogue/observation/propagation/conversation)만 감쇠
  · seed(정체성)·reflection(통찰)은 망각 면제

사용:
    uv run python scripts/eval_forgetting.py
"""

import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory.retriever import EPISODIC_SOURCES, retention  # noqa: E402

DAYS = [0, 1, 3, 7, 14]

# (라벨, source, recall_count) — 데모용 대표 기억 4종
SAMPLES = [
    ("사소한 잡담", "dialogue", 0),
    ("곰 목격(2회 회상)", "observation", 2),
    ("나는 마법사다(정체성)", "seed", 0),
    ("용 사건에 대한 통찰", "reflection", 0),
]


def _w(s: str) -> int:
    """동아시아 문자 폭(2칸) 고려한 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _w(s))


def main():
    print("[망각 곡선 시뮬레이션]  retention = exp(-Δd / (3.0 × (1 + recall_count)))")
    print("  · 일화 기억만 감쇠 · seed(정체성)·reflection(통찰)은 면제\n")

    lcol = 36
    head = _pad("기억 [출처]", lcol) + "".join(f"{'day' + str(d):>7}" for d in DAYS)
    print("  " + head)
    print("  " + "-" * _w(head))

    for label, src, rc in SAMPLES:
        exempt = src not in EPISODIC_SOURCES
        cells = "".join(f"{(1.0 if exempt else retention(d, rc)):>7.2f}" for d in DAYS)
        tag = "   ← 면제" if exempt else ""
        print("  " + _pad(f"{label} [{src}]", lcol) + cells + tag)

    print("\n  → 회상하지 않은 일화 기억(잡담)은 빠르게 잊히고,")
    print("     자주 회상한 기억(곰, 2회)은 오래 남으며,")
    print("     정체성(seed)·통찰(reflection)은 잊히지 않는다.")


if __name__ == "__main__":
    main()
