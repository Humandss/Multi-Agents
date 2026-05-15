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
    for m in retrieved[:3]:
        text = m["text"].strip()
        if text.startswith("플레이어가 말했다:"):
            content = text[len("플레이어가 말했다:"):].strip()
            parts.append(f"전에 듣기로 {content}")
        elif "와 대화:" in text or "한테 들었다:" in text:
            # NPC-NPC 대화나 propagation 메모리는 너무 길게 끌고 들어오지 않도록 단축
            parts.append(text[:80])
        else:
            parts.append(text)

    facts = "; ".join(parts)
    # 명확히 구조화: 회상 따로 + 질문 따로. LLM이 회상에 끌려가지 않고 질문에 집중하도록.
    return f"[참고 기억: {facts}]\n질문: {user_text}"
