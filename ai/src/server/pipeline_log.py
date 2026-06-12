"""파이프라인 로그 — 발화→저장→전파→언급 4단계 흐름 추적.

본 프로젝트의 핵심 파이프라인을 콘솔(색상) + 메모리 버퍼(endpoint 조회)에 기록.
발표/디버깅용 — 실제 서버 로직에 심어 데모와 실제가 일치하도록.

단계:
  ① UTTER  플레이어 발화 → NPC
  ② STORE  NPC ChromaDB 저장 (source/importance/메타)
  ③ SPREAD 전파 (from→to, player_origin)
  ④ RECALL 다른 NPC 회상 + 응답
"""

from collections import deque

# ANSI 색상 (Windows 터미널 대부분 지원, VS Code/Windows Terminal OK)
_C = {
    "utter": "\033[96m",    # cyan
    "store": "\033[93m",    # yellow
    "spread": "\033[95m",   # magenta
    "recall": "\033[92m",   # green
    "trust": "\033[91m",    # red
    "reflect": "\033[94m",  # blue
    "chat": "\033[35m",     # 진한 magenta
    "quest": "\033[33m",    # 진한 yellow
    "reset": "\033[0m",
    "dim": "\033[90m",
}

_STAGE_LABEL = {
    "utter": "1.발화",
    "store": "2.저장",
    "spread": "3.전파",
    "recall": "4.언급",
    "trust": "+ 친밀도",
    "reflect": "* 통찰",
    "chat": "o 자율대화",
    "quest": "! 퀘스트",
}

# on/off 토글 — 발표/디버깅 시 True, 운영 시 False 가능
ENABLED = True

# 최근 이벤트 버퍼 (endpoint 조회용)
_BUFFER: deque = deque(maxlen=200)
_seq = 0


def _next_seq() -> int:
    global _seq
    _seq += 1
    return _seq


def log(stage: str, message: str, **fields):
    """파이프라인 단계 로그.

    stage: utter | store | spread | recall
    message: 사람이 읽을 한 줄 요약
    fields: 추가 구조화 데이터 (endpoint JSON에 포함)
    """
    if not ENABLED:
        return
    label = _STAGE_LABEL.get(stage, stage)
    color = _C.get(stage, "")
    # 콘솔 출력 (색상). cp949 콘솔에서 특수문자 인코딩 에러 방지.
    line = f"{color}[PIPELINE {label}]{_C['reset']} {message}"
    try:
        print(line)
    except UnicodeEncodeError:
        # 인코딩 불가 문자 제거 후 출력
        print(line.encode("utf-8", "replace").decode("utf-8", "replace"))
    # 버퍼 저장
    _BUFFER.append({
        "seq": _next_seq(),
        "stage": stage,
        "label": label,
        "message": message,
        **fields,
    })


def recent(n: int = 50) -> list:
    """최근 n개 파이프라인 이벤트 (endpoint 조회용)."""
    items = list(_BUFFER)
    return items[-n:]


def clear():
    """버퍼 초기화."""
    _BUFFER.clear()
