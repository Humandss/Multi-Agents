"""회상 메모리 → user prompt 통합.

Production: src/server/engine.py의 NpcServer가 이 함수만 사용.
LoRA 시절의 NpcChat 클래스는 legacy로 분리됨 (ai/legacy/).
"""


def build_user_prompt(retrieved, user_text):
    """검색된 메모리를 자연어 prefix로 user 메시지에 녹임.

    구조:
        [참고 기억: ...]
        질문: {user_text}

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

        if text.startswith("플레이어가 말했다:"):
            content = text[len("플레이어가 말했다:"):].strip()
            if is_player_personal:
                # 플레이어 자기소개는 별도 prefix로 강조
                parts.append(f"★ 플레이어가 본인 정보를 알려줬음: '{content}'")
                has_personal = True
            else:
                parts.append(f"전에 플레이어가 말함: {content}")
        elif "와 대화:" in text or "한테 들었다:" in text:
            parts.append(text[:80])
        else:
            parts.append(text)

    facts = "; ".join(parts)
    # 회상에 plyaer 자기소개 있으면 추가 지시 명시
    extra = ""
    if has_personal:
        extra = (
            "\n→ 위 ★ 정보는 플레이어가 직접 알려준 본인 정보임. "
            "관련 질문에 그 내용을 그대로 인용해 답할 것."
        )
    return f"[참고 기억: {facts}]{extra}\n질문: {user_text}"
