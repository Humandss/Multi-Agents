"""회상 메모리 → user prompt 통합.

Production: src/server/engine.py의 NpcServer가 이 함수만 사용.
LoRA 시절의 NpcChat 클래스는 legacy로 분리됨 (ai/legacy/).
"""

# 인식론적 기억 타입 라벨 (Hindsight 계열 — 사실/경험/믿음 분리).
# source(출처)를 인식론 타입으로 변환해 LLM이 확신도를 조절하게 함:
#   직접 들음(확신) / 전해 들은 소문(전언 투) / 나의 생각(추정 투) / 아는 사실(확신)
_EPISTEMIC_LABEL = {
    "dialogue": "직접 들음",
    "observation": "직접 겪음",
    "propagation": "전해 들은 소문",
    "conversation": "나눈 대화",
    "reflection": "나의 생각",
    "seed": "아는 사실",
}


def build_user_prompt(retrieved, user_text):
    """검색된 메모리를 자연어 prefix로 user 메시지에 녹임.

    구조:
        [참고 기억: (인식론라벨) 내용; ...]
        질문: {user_text}

    각 기억 앞에 인식론 라벨을 붙여 NPC가 확신도를 다르게 표현하도록
    (소문은 '~라더라', 생각은 '~인 듯', 직접 들은 건 확신).
    NPC-NPC 대화나 propagation 메모리는 80자로 단축 (장문 끌려감 방지).
    """
    if not retrieved:
        return user_text

    parts = []
    has_personal = False  # personal 정보 강조용
    for m in retrieved[:3]:
        text = m["text"].strip()
        meta = m.get("metadata", {})
        is_player_personal = bool(meta.get("player")) and bool(meta.get("has_personal"))
        label = _EPISTEMIC_LABEL.get(meta.get("source", ""), "기억")

        if text.startswith("플레이어가 말했다:"):
            content = text[len("플레이어가 말했다:"):].strip()
            if is_player_personal:
                # 플레이어 자기소개는 별도 prefix로 강조
                parts.append(f"★ 플레이어가 본인 정보를 알려줬음: '{content}'")
                has_personal = True
            else:
                parts.append(f"(직접 들음) 플레이어가 말함: {content}")
        elif "와 대화:" in text or "한테 들었다:" in text:
            parts.append(f"({label}) {text[:80]}")
        else:
            parts.append(f"({label}) {text}")

    facts = "; ".join(parts)
    # 회상에 plyaer 자기소개 있으면 추가 지시 명시
    extra = ""
    if has_personal:
        extra = (
            "\n→ 위 ★ 정보는 플레이어가 직접 알려준 본인 정보임. "
            "관련 질문에 그 내용을 그대로 인용해 답할 것."
        )
    return f"[참고 기억: {facts}]{extra}\n질문: {user_text}"
