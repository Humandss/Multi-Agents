"""5종 NPC를 한 프로세스에서 추론하는 통합 엔진.

베이스 EXAONE 1개 + 5종 LoRA 어댑터를 PEFT의 load_adapter / set_adapter로
스위칭하면서 사용한다. 메모리 store/retriever도 NPC별로 보유.
또한 NPC 간 정보 전파(시간 기반)도 동일 프로세스에서 수행한다.
"""

import json
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 응답에서 강제 제거할 emoji/특수문자 — system prompt instruction이 무시되는 경우 후처리.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"      # 일반 이모지/픽토그램
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"      # ♪✨ 등 dingbats + misc symbols
    "\U0001F1E0-\U0001F1FF"      # 국기
    "\U0001F600-\U0001F64F"      # 이모티콘
    "\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)


_BRACKET_NOISE = re.compile(r"\[[^\[\]]{1,15}\]")  # [이옵니다], [^-^], [리 등 짧은 대괄호 표기

# 영문 NPC 이름 → 한글 표기 (system prompt에서 영문 이름 사용해서 응답에 leak되는 문제 해결)
_NAME_NORMALIZE = [
    (re.compile(r"\b[Hh]err?mann\b"), "헤르만"),
    (re.compile(r"\b[Mm]athilda\b"), "마틸다"),
    (re.compile(r"\b[Bb]ernhardt\b"), "베른하르트"),
    (re.compile(r"\b[Ee]lias\b"), "엘리아스"),
    (re.compile(r"\b[Ff]inn\b"), "핀"),
    # 한글 변형도 통일
    (re.compile(r"베르나르드|베르나르트"), "베른하르트"),
    (re.compile(r"마트닐라|마트일다|수학틸라|수학틸다"), "마틸다"),
    (re.compile(r"헤르몬|헤른|헤르몽"), "헤르만"),
    (re.compile(r"엘리어스|엘시아스|엘리아斯"), "엘리아스"),
    # 일본어/외국어 조사 leak ("금단の책" 같은)
    (re.compile(r"の"), "의"),
    (re.compile(r"[぀-ゟ゠-ヿ]+"), ""),  # 히라가나/카타카나 제거
]

# 이중 조사/어미 정리 (모델이 system prompt 어미 명령 따라가다가 본래 어미와 충돌)
_PARTICLE_FIX = [
    # 이중 조사: 첫 번째만 남김 (예: 헤르만이가 → 헤르만이)
    (re.compile(r"(이|는|은|을|를|과|와)(가|은|를|을|와|과)\b"),
     lambda m: m.group(1)),
    # 일반 ~니다/니까 + 추가 어미 (바랍니다오, 사료됩니다오, 권장드립니다지만 등)
    (re.compile(r"(니다|니까)(요|오|지만|게요|이오|으나)"),
     lambda m: m.group(1)),
    # 이중 어미: ~ㅂ니다 + 추가 어미 (구체적 패턴 — 위 일반 패턴 보조)
    (re.compile(r"(습니다|입니다|옵니다|됩니다|십니다)(요|오|지만|게요|이오|으나)"),
     lambda m: m.group(1)),
    # 과한 사극체 변환 (elias): ~소이옵니다 → ~소, ~옵소서 → ~오, ~사옵니다 → ~오
    (re.compile(r"소이옵니다"), "소"),
    (re.compile(r"옵소서"), "오"),
    (re.compile(r"하옵나이까"), "하오"),
    (re.compile(r"(하|되|있|없|받|드리)옵니다"), lambda m: m.group(1) + "오"),
    (re.compile(r"이옵나이다"), "이오"),
    # finn 과도한 어미: ~사옵니다 → ~노라, ~이옵니다 (드물게 finn) → ~이라
    (re.compile(r"사옵니다"), "노라"),
    # elias 어미 시도 실패 패턴 (계시리오까요, 무엇이오까 등)
    (re.compile(r"(리오|이오|시오)까요?"), lambda m: m.group(1)),
    # finn 이중 어미 (게요이지요, 게요사옵니다 등)
    (re.compile(r"게요(이지요|이옵니다|사옵니다|리라)"), lambda m: m.group(1)),
    # ~까요 + 추가 어미
    (re.compile(r"까요(요|오)"), "까요"),
    # 어미 변형: ~리이다 → ~리다 (finn 어미 변형)
    (re.compile(r"리리이다"), "리이다"),
    (re.compile(r"리이다이다"), "리이다"),
    # 단순 중복: 요요/오오 → 요/오
    (re.compile(r"요요(?=[\s.,!?]|$)"), "요"),
    (re.compile(r"오오(?=[\s.,!?]|$)"), "오"),
    # 조사 + 으나 (어색한 결합)
    (re.compile(r"(다|요)으나"), lambda m: m.group(1) + "나"),
    # 중복 공백 정리
    (re.compile(r"  +"), " "),
    # 문장부호 앞 공백 제거
    (re.compile(r"\s+([.,!?])"), lambda m: m.group(1)),
    # 마침표 연속 (... 제외하고 길게)
    (re.compile(r"\.{4,}"), "..."),
    # 응답 시작에 어색한 접속사 (그리고, 그래서)
    (re.compile(r"^(그리고|그래서|그러면|그러니까)[\s,]+"), ""),
]


_SENT_ENDS = (".", "?", "!", "。", "?", "!", "~")


def _cut_to_last_sentence(text: str) -> str:
    """마지막 완전한 문장 종결까지만 유지. max_new_tokens 한계로 잘린 끝 부분 제거.

    예: "어서 오시지요. 무엇을 찾으십..." → "어서 오시지요."
    예: "흠, 안녕하시오. 환영하나이다 오시" → "흠, 안녕하시오. 환영하오."
    """
    text = text.rstrip()
    if not text:
        return text
    # 이미 종결 부호로 끝나면 OK
    if text[-1] in _SENT_ENDS:
        return text
    # 마지막 종결 부호 위치 찾기
    last_end = max(text.rfind(c) for c in _SENT_ENDS)
    if last_end >= 0:
        # 종결 후 자투리 있으면 잘림 — 종결까지만 유지
        # 단 종결 직후 한 글자 정도면 무시 (자연스러운 응답 형태일 수도)
        tail = text[last_end + 1:].strip()
        if len(tail) >= 2:  # 2자 이상 미완성 꼬리만 자름
            return text[:last_end + 1]
    return text


def _has_jongseong(ch: str) -> bool:
    """한글 받침 유무 — 받침 있으면 True (예: 헤르만의 ㄴ → True, 마틸다 → False)."""
    if not ch or len(ch) != 1:
        return False
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return False
    return ((code - 0xAC00) % 28) != 0


_NPC_NAMES_KO = ["엘리아스", "헤르만", "마틸다", "핀", "베른하르트"]

def _fix_korean_particles(text: str) -> str:
    """NPC 이름 뒤 한글 조사 자동 보정 (받침 따라 와/과, 이/가, 은/는, 을/를)."""
    for name in _NPC_NAMES_KO:
        last_ch = name[-1]
        has_jong = _has_jongseong(last_ch)
        # 와/과
        if has_jong:
            text = text.replace(f"{name}와", f"{name}과")
        else:
            text = text.replace(f"{name}과", f"{name}와")
        # 이/가 (다만 "마틸다이" 같은 명백한 어휘 충돌 회피 위해 단어 경계 신경)
        # 은/는, 을/를도 받침 따라
        if has_jong:
            text = re.sub(rf"{name}가\b", f"{name}이", text)
            text = re.sub(rf"{name}는\b", f"{name}은", text)
            text = re.sub(rf"{name}를\b", f"{name}을", text)
        else:
            text = re.sub(rf"{name}이\b", f"{name}가", text)
            text = re.sub(rf"{name}은\b", f"{name}는", text)
            text = re.sub(rf"{name}을\b", f"{name}를", text)
    return text


# NPC별 어미 후처리 — 각 NPC가 일관된 어미를 쓰도록.
# 모델이 system prompt 어미 명령 무시할 때 강제 변환 (LLM 응답 잘 따라가지 않는 패턴).
_NPC_POSTFIX = {
    "elias": [
        # 응답 시작 어색 감탄사 정리 ("아호", "아하" 등 → "흠")
        (re.compile(r"^(아호|아하|어허|오호|에헴)([\s,]+)"), r"흠\2"),
        # "~겠습니까?" / "~시겠습니까?" → "~겠소?" / "~시겠소?"
        (re.compile(r"([가-힣])시겠습니까\?"), lambda m: m.group(1) + "시겠소?"),
        (re.compile(r"([가-힣])시겠습니까\b"), lambda m: m.group(1) + "시겠소"),
        (re.compile(r"겠습니까\?"), "겠소?"),
        (re.compile(r"겠습니까\b"), "겠소"),
        # "~나니" / "~으나니" → "~오" (Elias 어미 leak)
        (re.compile(r"([가-힣])으나니\b"), lambda m: m.group(1) + "오"),
        (re.compile(r"([가-힣])나니\b"), lambda m: m.group(1) + "오"),
        (re.compile(r"([가-힣])나니\.\.\.?"), lambda m: m.group(1) + "오..."),
        # 어색한 사극체 변형 정리 (LLM이 다양한 사극체 시도하다 실패하는 케이스)
        (re.compile(r"하나이다\b"), "하오"),
        (re.compile(r"이나이다\b"), "이오"),
        (re.compile(r"([가-힣])나이다\b"), lambda m: m.group(1) + "오"),
        (re.compile(r"십시오\?"), "시오?"),
        (re.compile(r"십시오\b"), "시오"),
        (re.compile(r"겠소이다\b"), "겠소"),
        (re.compile(r"겠소다\b"), "겠소"),
        # "~소다" / "~소이다" → "~소"
        (re.compile(r"([가-힣])소이다\b"), lambda m: m.group(1) + "소"),
        (re.compile(r"([가-힣])소다\b"), lambda m: m.group(1) + "소"),
        # "~다오" / "~이다오" → "~오" / "~이오"
        (re.compile(r"있다오\b"), "있소"),
        (re.compile(r"없다오\b"), "없소"),
        (re.compile(r"이다오\b"), "이오"),
        (re.compile(r"([가-힣])다오\b"), lambda m: m.group(1) + "오"),
        # "~으니이다" / "~니이다" → "~소" / "~오"
        (re.compile(r"있으니이다\b"), "있소"),
        (re.compile(r"없으니이다\b"), "없소"),
        (re.compile(r"([가-힣])으니이다\b"), lambda m: m.group(1) + "소"),
        (re.compile(r"([가-힣])니이다\b"), lambda m: m.group(1) + "오"),
        # "~오리다" / "~으리다" → "~오"
        (re.compile(r"하오리다\b"), "하오"),
        (re.compile(r"([가-힣])오리다\b"), lambda m: m.group(1) + "오"),
        (re.compile(r"([가-힣])으리다\b"), lambda m: m.group(1) + "오"),
        # "~이라오" → "~이오"
        (re.compile(r"이라오\b"), "이오"),
        # "환영하나이다 오시" 같은 잘림 — "오시" 단독 등장 시 제거
        (re.compile(r"\s+오시(?=[\s.,!?]|$)"), ""),
        # "~십니까?" → "~시오?" (사극체 학자 어조 강제)
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "시오?"),
        (re.compile(r"([가-힣])십니까\b"), lambda m: m.group(1) + "시오"),
        # "~나요?" / "~가요?" → "~오?"
        (re.compile(r"([가-힣])(시|으시)?나요\?"), lambda m: m.group(1) + "시오?" if m.group(2) else m.group(1) + "오?"),
        (re.compile(r"([가-힣])(시|으시)?가요\?"), lambda m: m.group(1) + "시오?" if m.group(2) else m.group(1) + "오?"),
        # "~다마저도" 같은 어색 어미
        (re.compile(r"다마저도\b"), "오"),
        (re.compile(r"마저도\b"), ""),
        # "~세요" 명령형 → "~시오"
        (re.compile(r"([가-힣])세요\b"), lambda m: m.group(1) + "시오"),
        # "~내세요" / "~해주세요" → "~내시오" / "~해주시오"
        (re.compile(r"([가-힣])주세요\b"), lambda m: m.group(1) + "주시오"),
        # "~합니다" → "~하오"
        (re.compile(r"합니다\b"), "하오"),
        (re.compile(r"됩니다\b"), "되오"),
        (re.compile(r"입니다\b"), "이오"),
        (re.compile(r"있습니다\b"), "있소"),
        (re.compile(r"없습니다\b"), "없소"),
        (re.compile(r"겠습니다\b"), "겠소"),
        (re.compile(r"드립니다\b"), "드리오"),
        (re.compile(r"드리겠습니다\b"), "드리겠소"),
        # 어색한 "~신지오" → "~신지"
        (re.compile(r"신지오"), "신지"),
        # "~시죠" / "~시지요" → "~시오"
        (re.compile(r"([가-힣])시죠\b"), lambda m: m.group(1) + "시오"),
        (re.compile(r"([가-힣])시지요\b"), lambda m: m.group(1) + "시오"),
        # "~시리오" → "~시오"
        (re.compile(r"([가-힣])시리오\b"), lambda m: m.group(1) + "시오"),
        # "~시군요" / "~시는군요" → "~시오" (인지·감탄 어미 → 학자 어조)
        (re.compile(r"안녕하시군요"), "안녕하시오"),
        (re.compile(r"([가-힣])시는군요\b"), lambda m: m.group(1) + "시는구려"),
        (re.compile(r"([가-힣])시군요\b"), lambda m: m.group(1) + "시구려"),
        # 일반 "~군요" → "~구려" (예: "그렇군요" → "그렇구려")
        (re.compile(r"([가-힣])군요\b"), lambda m: m.group(1) + "구려"),
        (re.compile(r"([가-힣])네요\b"), lambda m: m.group(1) + "구려"),
        # "~죠?" → "~오?"
        (re.compile(r"죠\?"), "오?"),
        (re.compile(r"죠\."), "오."),
        # "궁금합니다" 등 자주 leak 패턴
        (re.compile(r"궁금합니다\b"), "궁금하오"),
        (re.compile(r"감사합니다\b"), "고맙소"),
        (re.compile(r"바랍니다\b"), "바라오"),
        (re.compile(r"생각합니다\b"), "생각하오"),
        (re.compile(r"부탁합니다\b"), "부탁하오"),
    ],
    "finn": [
        # finn 시적 어조 강제: "~합니다" → "~하노라" 등
        (re.compile(r"합니다\b"), "하노라"),
        (re.compile(r"입니다\b"), "이라"),
        (re.compile(r"있습니다\b"), "있노라"),
        (re.compile(r"없습니다\b"), "없노라"),
        (re.compile(r"됩니다\b"), "되노라"),
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "시는가?"),
        # "~죠" → "~노라"
        (re.compile(r"하죠\b"), "하노라"),
        (re.compile(r"있죠\b"), "있노라"),
        # quest description 명령형 → 시적 권유
        (re.compile(r"([가-힣])해주세요\b"), lambda m: m.group(1) + "해주오"),
        (re.compile(r"([가-힣])주세요\b"), lambda m: m.group(1) + "주오"),
        (re.compile(r"([가-힣])세요\b"), lambda m: m.group(1) + "시오"),
        (re.compile(r"([가-힣])요\b"), lambda m: m.group(1) + "오"),
    ],
    "mathilda": [
        # mathilda는 "~네요/~죠/~어요" 자연. "~십니까" 어색 → "~세요"
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "세요?"),
        (re.compile(r"([가-힣])십니까\b"), lambda m: m.group(1) + "세요"),
        # "~합니다" 너무 형식적 → "~해요"
        (re.compile(r"합니다\b"), "해요"),
        (re.compile(r"입니다\b"), "이에요"),
        (re.compile(r"있습니다\b"), "있어요"),
        (re.compile(r"없습니다\b"), "없어요"),
        # 사극체 leak → 친근 어조
        (re.compile(r"([가-힣])하오\b"), lambda m: m.group(1) + "해요"),
        (re.compile(r"([가-힣])시오\b"), lambda m: m.group(1) + "세요"),
        (re.compile(r"([가-힣])이오\b"), lambda m: m.group(1) + "이에요"),
        (re.compile(r"([가-힣])구려\b"), lambda m: m.group(1) + "네요"),
    ],
    "hermann": [
        # hermann은 반말. 존댓말 leak 시 반말로 강제.
        # 1) 흔한 정중 표현 → 반말
        (re.compile(r"아니요\b"), "아니"),
        (re.compile(r"아니에요\b"), "아니"),
        (re.compile(r"그래요\b"), "그래"),
        (re.compile(r"맞아요\b"), "맞아"),
        (re.compile(r"글쎄요\b"), "글쎄"),
        (re.compile(r"고마워요\b"), "고맙다"),
        (re.compile(r"감사해요\b"), "고맙다"),
        (re.compile(r"미안해요\b"), "미안"),
        (re.compile(r"괜찮아요\b"), "괜찮아"),
        # 2) 합니다/입니다 류
        (re.compile(r"하세요\b"), "해"),
        (re.compile(r"드릴게요\b"), "줄게"),
        (re.compile(r"드립니다\b"), "준다"),
        (re.compile(r"있습니다\b"), "있어"),
        (re.compile(r"없습니다\b"), "없어"),
        (re.compile(r"합니다\b"), "해"),
        (re.compile(r"입니다\b"), "이야"),
        (re.compile(r"됩니다\b"), "돼"),
        # 3) 추천/권장 류 - hermann은 무뚝뚝하게 거절
        (re.compile(r"권장할게요?\b"), "가봐"),
        (re.compile(r"권장합니다\b"), "가봐"),
        (re.compile(r"권장해\b"), "가봐"),
        (re.compile(r"추천드립니다\b"), "추천한다"),
        (re.compile(r"추천드려요?\b"), "추천한다"),
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "냐?"),
        (re.compile(r"(이|있|없)어요\b"), lambda m: m.group(1) + "어"),
        # quest description 명령형 → 반말 (순서 중요: 긴 패턴 먼저)
        (re.compile(r"([가-힣])해주세요\b"), lambda m: m.group(1) + "해줘"),
        (re.compile(r"([가-힣])아주세요\b"), lambda m: m.group(1) + "아줘"),
        (re.compile(r"([가-힣])어주세요\b"), lambda m: m.group(1) + "어줘"),
        (re.compile(r"주세요\b"), "줘"),
        (re.compile(r"([가-힣])보세요\b"), lambda m: m.group(1) + "봐"),
        (re.compile(r"([가-힣])하세요\b"), lambda m: m.group(1) + "해"),
        # 일반 "~세요" → "~라" (밝혀내세요 → 밝혀내라)
        (re.compile(r"([가-힣])세요\b"), lambda m: m.group(1) + "라"),
        (re.compile(r"([가-힣])하시오\b"), lambda m: m.group(1) + "해"),
        (re.compile(r"([가-힣])시오\b"), lambda m: m.group(1) + "라"),
    ],
    "bernhardt": [
        # bernhardt는 "~지요/~습니다" 자연. 일부 패턴만.
        # "~죠" → "~지요" (더 정중하게)
        (re.compile(r"하죠\b"), "하지요"),
        (re.compile(r"있죠\b"), "있지요"),
        # "~십시오죠" / "~시오죠" 같은 어색한 어미 중복 정리
        (re.compile(r"십시오죠\b"), "십시오"),
        (re.compile(r"십시오\.죠"), "십시오."),
        (re.compile(r"시지요죠\b"), "시지요"),
        (re.compile(r"([가-힣])오죠\b"), lambda m: m.group(1) + "지요"),
        # 일반 죠 → 지요 (단어 끝 자연스러운 부분만)
        (re.compile(r"([가-힣])\s죠\b"), lambda m: m.group(1) + " 지요"),
        # quest description 명령형 → 정중한 거래 어조
        (re.compile(r"([가-힣])해주세요\b"), lambda m: m.group(1) + "해주시지요"),
        (re.compile(r"([가-힣])아주세요\b"), lambda m: m.group(1) + "아주시지요"),
        (re.compile(r"([가-힣])어주세요\b"), lambda m: m.group(1) + "어주시지요"),
        (re.compile(r"주세요\b"), "주시지요"),
        (re.compile(r"([가-힣])세요\b"), lambda m: m.group(1) + "시지요"),
        # 사극체 leak → 현대 정중
        (re.compile(r"([가-힣])하오\b"), lambda m: m.group(1) + "합니다"),
        (re.compile(r"([가-힣])이오\b"), lambda m: m.group(1) + "입니다"),
        # 과한 사극체 어미 정리
        (re.compile(r"아니옵니다\b"), "아닙니다"),
        (re.compile(r"이옵니다\b"), "입니다"),
        (re.compile(r"하옵니다\b"), "합니다"),
        (re.compile(r"되옵니다\b"), "됩니다"),
        (re.compile(r"드리옵니다\b"), "드립니다"),
        (re.compile(r"있사옵니다\b"), "있습니다"),
        (re.compile(r"없사옵니다\b"), "없습니다"),
        (re.compile(r"사옵니다\b"), "습니다"),
        # "~마는" → "~만" (어색한 격식체 어미)
        (re.compile(r"습니다마는\b"), "습니다만"),
        (re.compile(r"입니다마는\b"), "입니다만"),
        (re.compile(r"합니다마는\b"), "합니다만"),
        (re.compile(r"니다마는\b"), "니다만"),
        # "~으리라" / "~리라" → 현대 추정
        (re.compile(r"있으리라\b"), "있을 것"),
        (re.compile(r"없으리라\b"), "없을 것"),
        (re.compile(r"되리라\b"), "될 것"),
        (re.compile(r"하리라\b"), "할 것"),
        # "~사오니" → "~하니" (사극체)
        (re.compile(r"사오니\b"), "하니"),
        (re.compile(r"오니까\b"), "으니까"),
    ],
}


def _clean_response(text: str, npc: str = None) -> str:
    """LLM 응답에서 emoji/특수문자/대괄호 표기 제거 + NPC 이름 한글 정규화 + 미완성 끝 cut.

    system prompt의 형식 안내 + 영문 NPC 이름이 응답에 leak되는 부작용 정리.
    npc가 주어지면 NPC별 어미 후처리도 적용.
    """
    text = _EMOJI_PATTERN.sub("", text)
    text = _BRACKET_NOISE.sub("", text)
    # 닫히지 않은 [ 시작 — 응답 끝에서 cut됨
    text = re.sub(r"\[[^\[\]]{0,15}$", "", text)
    # NPC 이름 한글 정규화
    for pattern, replacement in _NAME_NORMALIZE:
        text = pattern.sub(replacement, text)
    # 이중 조사/어미 정리
    for pattern, replacement in _PARTICLE_FIX:
        text = pattern.sub(replacement, text)
    # NPC별 어미 후처리
    if npc and npc in _NPC_POSTFIX:
        for pattern, replacement in _NPC_POSTFIX[npc]:
            text = pattern.sub(replacement, text)
    # NPC 이름 뒤 조사 보정
    text = _fix_korean_particles(text)
    # 미완성 끝 문장 자르기
    text = _cut_to_last_sentence(text)
    return text.strip()

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..memory import MemoryEntry, MemoryRetriever, MemorySource, MemoryStore
from ..memory.chat import build_user_prompt
from ..propagation.graph import RelationGraph
from ..propagation.simulator import PropagationSimulator

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
BASE_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"

DEFAULT_CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]

# NPC별 강한 instruction — EXAONE의 정중한 baseline 깨고 페르소나 강제
# 형식 단순화: 모델이 어미 표시 대괄호를 응답에 포함시키는 부작용 회피.
# 직업 명시: NPC가 다른 NPC 영역 침범하는 것 방지.
# Quest hook: 회상 정보가 흥미로운 사건이면 페르소나에 맞게 흘림 (플레이어가 단서 놓치지 않도록).
NPC_STRICT_RULES = {
    "elias": (
        "직업: 마법사·학자. 마법·학문·검증 질문만 답하시오.\n"
        "차분한 학자 어조, 약간 회의적. **어미는 오직 ~오 / ~이오 / ~구려 3가지만**. "
        "다른 사극체(~나이다 ~소이다 ~옵소서 ~십시오 ~군요 ~네요) 절대 사용 금지. "
        "현대체(~니다 ~습니다) 절대 사용 금지.\n"
        "한 문장만. 흠 으로 시작 자주.\n"
        "예: 안녕하세요 -> 흠, 무슨 일이오?\n"
        "예: 마법 어디서 배웠어요 -> 옛 도시에서 50년 익혔구려.\n"
        "예: 광산은? -> 흠... 그 일이 마음에 걸리오.\n"
        "예: 환영해주세요 -> 어서 오시오, 무엇이 궁금하오?"
    ),
    "hermann": (
        "직업: 대장장이. 검·쇠·도구 거래만. 약초·음식은 다른 NPC로 보내시오.\n"
        "반말 단답. 존댓말 절대 금지. 한 문장. "
        "어. 음. ... 같은 짧은 시작. 어미는 다 해 지.\n"
        "회상 사건이면 모험가로 부르며 호출 "
        "(예: '너 모험가지? 그 검 봤냐?').\n"
        "예: 안녕하세요 -> 어. 무슨 일.\n"
        "예: 검 추천해줘 -> 음, 강철 단검부터 써봐."
    ),
    "mathilda": (
        "직업: 술집 주인. 음식·음료·소문만. 검·약초는 다른 NPC로 보내시오.\n"
        "따뜻하고 수다스러움. 어머나 어머 아유 자주. "
        "어미는 ~어요 ~네요 ~죠. 한 문장.\n"
        "예: 안녕하세요 -> 어머나, 어서 와요!\n"
        "예: 좋은 술 있어요? -> 아유, 오늘은 특히 맥주가 맛있어요.\n"
        "예: 무슨 소문 있어요 -> 어머, 광산 얘기 들었어요?"
    ),
    "finn": (
        "직업: 음유시인. 노래·이야기·전설만. 거래는 다른 NPC로 보내시오.\n"
        "시적이지만 짧게. 오 그대 자주. 어미는 ~이라 ~리라 ~노라 위주. "
        "절대 금지: ~사옵니다 ~이옵니다 너무 과한 사극체.\n"
        "한 문장.\n"
        "예: 안녕하세요 -> 오 그대, 별빛 같은 걸음이라.\n"
        "예: 노래 한 곡 -> 그대를 위해 옛 영웅의 노래를 부르리라.\n"
        "예: 광산은? -> 그곳엔 22인의 혼이 잠들었노라."
    ),
    "bernhardt": (
        "직업: 잡화상. 약초·잡화만. 검·무기는 헤르만으로 보내시오.\n"
        "정중하고 거래 실용적. 어서 흠 같은 시작. "
        "어미는 ~지요 ~이오 ~습니다 짧게. 한 문장.\n"
        "예: 안녕하세요 -> 어서 오시지요, 뭐 찾으십니까?\n"
        "예: 약초 있어요? -> 흠, 회복약이라면 셋 정도 있지요.\n"
        "예: 비싸네요 -> 좋은 물건이라 그 값이지요."
    ),
}

# NPC별 generation 파라미터 차별화
# - hermann/elias: 짧고 무뚝뚝/회의적 → 낮은 temp + 짧은 max_tokens
# - mathilda/finn: 수다스럽고 시적 → 약간 높은 temp + 긴 max_tokens
# - bernhardt: 정중한 거래상 → 중간
# 속도 우선 — 한두 문장이면 충분. 페르소나도 더 자연스러움.
GEN_PARAMS = {
    "hermann":   {"temperature": 0.35, "max_new_tokens": 40, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "elias":     {"temperature": 0.35, "max_new_tokens": 50, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "mathilda":  {"temperature": 0.50, "max_new_tokens": 60, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
    "finn":      {"temperature": 0.45, "max_new_tokens": 60, "repetition_penalty": 1.18, "no_repeat_ngram_size": 3},
    "bernhardt": {"temperature": 0.40, "max_new_tokens": 55, "repetition_penalty": 1.18, "no_repeat_ngram_size": 4},
}

# NPC별 Quest Pool — 미리 정의된 quest. 조건(trust 등) 충족 시 NPC가 먼저 제안.
# id는 unique. trust_required: 이 trust 이상이어야 quest 제안.
# intro: NPC가 quest를 줄 때 첫 대사 (greeting 대체).
NPC_QUEST_POOL = {
    "elias": [
        {
            "id": "elias_mine_secret",
            "title": "광산 봉인의 진실",
            "description": "100년 전 봉인된 광산 입구를 조사해 그 비극의 흔적을 확인해주시오.",
            "reward": "고대 마법 지식",
            "trust_required": 40,
            "intro": "모험가 양반, 마침 잘 오셨소. 자네에게 부탁할 일이 하나 있소.",
        },
        {
            "id": "elias_old_text",
            "title": "잃어버린 마법서 조각",
            "description": "마을 어딘가에 흩어진 옛 마법서 조각을 모아 오시오.",
            "reward": "마법 시약",
            "trust_required": 60,
            "intro": "흠, 자네에게만 털어놓는 이야기인데... 도움이 필요하오.",
        },
    ],
    "hermann": [
        {
            "id": "hermann_meteor_ore",
            "title": "운철 광석 채집",
            "description": "산 너머 동굴에서 운철 광석을 캐 와라. 검 만드는 데 필요하다.",
            "reward": "강철 단검",
            "trust_required": 35,
            "intro": "어이 모험가, 잠깐. 너 이거 좀 해줘.",
        },
        {
            "id": "hermann_old_hammer",
            "title": "잃어버린 망치",
            "description": "할아버지 망치를 광산 근처에서 잃어버렸다. 찾아 와줘.",
            "reward": "특제 강철 무기",
            "trust_required": 65,
            "intro": "야, 너한테만 말하는 건데. 부탁 하나 들어줄래?",
        },
    ],
    "mathilda": [
        {
            "id": "mathilda_rumor_check",
            "title": "이상한 그림자 소문",
            "description": "밤마다 광장에 나타난다는 검은 그림자의 정체를 알아봐주세요.",
            "reward": "특별 양조 술",
            "trust_required": 35,
            "intro": "어머나 모험가 분, 마침 잘 오셨네요. 부탁 하나 들어주실래요?",
        },
        {
            "id": "mathilda_rare_herb",
            "title": "희귀 약초 구매",
            "description": "산 너머 가격이 올랐다는 희귀 약초를 구해 와주세요.",
            "reward": "최고급 요리",
            "trust_required": 55,
            "intro": "아유, 우리 단골! 마침 부탁할 게 있어요.",
        },
    ],
    "finn": [
        {
            "id": "finn_legend_verify",
            "title": "전설의 검증",
            "description": "용 사냥 영웅이 떠난 길을 따라가 그의 운명을 확인해주오.",
            "reward": "음유시인의 노래 (경험치)",
            "trust_required": 40,
            "intro": "오 그대여, 운명이 그대를 이끌었나니. 부탁이 하나 있노라.",
        },
        {
            "id": "finn_lost_song",
            "title": "잊혀진 노래의 악보",
            "description": "옛 도시에 묻혀있다는 전설의 악보를 찾아 와주오.",
            "reward": "고대 마력의 결정",
            "trust_required": 65,
            "intro": "오 영웅이여, 그대에게만 부탁할 수 있는 일이 있노라.",
        },
    ],
    "bernhardt": [
        {
            "id": "bernhardt_supply_run",
            "title": "약초 보급",
            "description": "산 너머 마을에서 약초를 사 와 주십시오. 거래 대금은 미리 드립니다.",
            "reward": "금화 30닢",
            "trust_required": 35,
            "intro": "어서 오십시오. 마침 거래 제안 하나 드리지요.",
        },
        {
            "id": "bernhardt_secret_item",
            "title": "잡화상의 비밀 거래",
            "description": "오래된 친구에게 보내는 물건을 옛 도시까지 전해주십시오.",
            "reward": "희귀 잡화 + 금화",
            "trust_required": 60,
            "intro": "흠, 자네 같은 단골에게만 말씀드리는 거래가 하나 있지요.",
        },
    ],
}


class QuestTracker:
    """NPC별 quest 상태 추적.

    상태:
      - available: 아직 제안 안 함 (trust 부족 또는 다른 quest 진행 중)
      - offered:   제안됨, 플레이어가 수락하지 않은 상태
      - completed: 완료
    """
    def __init__(self):
        self._state: dict[str, str] = {}  # quest_id → status

    def status(self, quest_id: str) -> str:
        return self._state.get(quest_id, "available")

    def mark_offered(self, quest_id: str):
        self._state[quest_id] = "offered"

    def mark_completed(self, quest_id: str):
        self._state[quest_id] = "completed"

    def get_active_quests(self, npc: str) -> list[dict]:
        """NPC가 현재 제안 가능한 quest 목록 (available 상태)."""
        pool = NPC_QUEST_POOL.get(npc, [])
        return [q for q in pool if self.status(q["id"]) == "available"]

    def get_pickable_quest(self, npc: str, current_trust: int) -> dict | None:
        """현재 trust로 받을 수 있는 quest 1개 (가장 낮은 trust_required부터)."""
        candidates = [
            q for q in self.get_active_quests(npc)
            if current_trust >= q.get("trust_required", 0)
        ]
        if not candidates:
            return None
        # trust_required 가장 낮은 quest 우선 (난이도 순서)
        candidates.sort(key=lambda q: q.get("trust_required", 0))
        return candidates[0]

    def snapshot(self) -> dict:
        return dict(self._state)


# NPC별 첫 만남 greeting — F 키로 처음 대화 시작할 때 NPC가 먼저 한마디.
# 신뢰도 4등급(낯선 사람/지인/친구/절친) × 3 바리에이션 = 다양성 보장.
# LLM 호출 없이 즉시 표시 = 빠름.
NPC_GREETINGS = {
    "elias": {
        "낯선 사람": [
            "흠, 처음 보는 얼굴이오. 여기엔 무슨 일이오?",
            "낯선 자가 내 공방을 찾았구려. 무슨 용건이오?",
            "흠... 그대를 본 적이 없는 듯하오. 무엇이 궁금하시오?",
        ],
        "지인": [
            "어서 오시오 모험가 양반. 무슨 일로 오셨소?",
            "흠, 또 오셨구려. 오늘은 무엇이 궁금하시오?",
            "어서 오시오. 마침 한가하던 참이오.",
        ],
        "친구": [
            "오, 자네인가. 잘 오셨소.",
            "반갑소이다 친구여. 무슨 이야기를 나눠볼까.",
            "어서 오시오, 늘 환영하오.",
        ],
        "절친": [
            "오, 자네 왔는가! 마침 자네 생각을 하던 참이오.",
            "허허, 친애하는 벗이여. 무슨 좋은 소식이라도?",
            "내 가장 신뢰하는 자가 왔구려. 어서 앉으시오.",
        ],
    },
    "hermann": {
        "낯선 사람": [
            "어. 누구야 너.",
            "음... 처음 보는데. 뭐 살 거 있어?",
            "어. 모험가냐? 검 보러 왔어?",
        ],
        "지인": [
            "어. 또 왔네. 뭐 필요해?",
            "음, 너구나. 무슨 일이야.",
            "어. 어서 와.",
        ],
        "친구": [
            "오, 왔어? 마침 잘 됐다.",
            "어이, 친구. 검 손볼 거 있냐?",
            "어. 잘 왔다. 한 잔 할래?",
        ],
        "절친": [
            "야, 보고 싶었다. 잘 지냈냐?",
            "오! 내 절친 왔구나. 뭐 도와줄 거 있냐?",
            "왔구나. 너 없으니 심심하더라.",
        ],
    },
    "mathilda": {
        "낯선 사람": [
            "어머, 처음 뵙는 손님이네요? 어서 오세요!",
            "어머나, 새 얼굴이네요! 차 한 잔 드릴까요?",
            "어서 오세요, 손님! 처음이시죠?",
        ],
        "지인": [
            "어머, 또 오셨네요! 반가워요.",
            "어서 오세요~ 오늘은 뭘 드시러 오셨어요?",
            "아유, 잘 오셨어요. 들어와 앉으세요.",
        ],
        "친구": [
            "어머나, 우리 단골! 보고 싶었어요!",
            "오, 친구 왔네! 자, 오늘은 특별히 맛있는 거 준비했어요.",
            "어머 어서 와요, 자리 비워뒀어요.",
        ],
        "절친": [
            "어머! 내 사랑하는 친구! 어서 와요, 빨리!",
            "꺄, 보고 싶었어요! 오늘은 우리 둘이 수다 떨어요.",
            "아유 정말, 왜 이렇게 오랜만이에요! 자, 앉아요.",
        ],
    },
    "finn": {
        "낯선 사람": [
            "오, 낯선 그대여. 별빛이 새 운명을 이끌어왔구려.",
            "그대의 이름은 들어본 적 없거늘, 어인 일로 이곳에?",
            "처음 뵙는 분이로다. 그대의 이야기를 들려주오.",
        ],
        "지인": [
            "오 그대여, 다시 만났구려. 무슨 노래를 들으러 오셨소?",
            "다시 그대를 보니 반갑소이다. 어떤 이야기를 원하오?",
            "오, 익숙한 발걸음이여. 어서 오시오.",
        ],
        "친구": [
            "오 친애하는 벗이여, 그대를 위한 노래가 준비되어 있다오.",
            "그대를 위해 새 시를 지었거늘, 들어보겠소?",
            "오, 영웅이여. 그대의 모험담을 들려주오.",
        ],
        "절친": [
            "오 나의 영원한 벗이여! 그대의 이야기가 전설로 남으리라.",
            "그대 없는 마을은 노래 없는 술집 같았소이다.",
            "어서 오시오 진정한 벗이여, 별빛이 그대를 환영하오.",
        ],
    },
    "bernhardt": {
        "낯선 사람": [
            "어서 오시지요. 처음 뵙는 분 같은데, 뭘 찾으십니까?",
            "흠, 새로운 손님이군요. 약초나 잡화 필요하시면 말씀하시지요.",
            "어서 오십시오. 저희 가게가 처음이신가요?",
        ],
        "지인": [
            "어서 오시지요. 오늘은 어떤 게 필요하십니까?",
            "또 오셨군요, 반갑습니다. 무엇을 보여드릴까요?",
            "흠, 어서 오시지요. 이번엔 무엇을 찾으십니까?",
        ],
        "친구": [
            "오, 친구분 오셨군요. 특별 할인 가격으로 해드리지요.",
            "어서 오시지요, 단골손님. 좋은 물건 들어왔습니다.",
            "반갑습니다. 자주 찾아주시니 감사할 따름이지요.",
        ],
        "절친": [
            "오, 가장 소중한 단골손님! 어서 오시지요.",
            "흠, 그대 같은 분께는 무엇이든 최고 품질로 드리지요.",
            "친애하는 벗이여, 어서 오시오. 오늘은 특별한 거래가 가능하지요.",
        ],
    },
}


# NPC별 Quest 안내 template — LLM 응답 뒤에 자연스럽게 이어붙임.
# {title}/{description}/{reward}만 채우면 됨. 페르소나 어조 유지하므로 추가 LLM 호출 불필요 = 빠름.
NPC_QUEST_INTRO = {
    "elias": (
        " 흠... 한 가지 부탁이 있소. "
        "「{title}」 — {description} "
        "성공하면 {reward} 보답하오."
    ),
    "hermann": (
        " 너 모험가지? 한 건 도와줘봐. "
        "「{title}」 — {description} "
        "끝내면 {reward} 챙겨주마."
    ),
    "mathilda": (
        " 어머나, 마침 잘 왔어요! 부탁 하나 들어줄래요? "
        "「{title}」 — {description} "
        "해주시면 {reward} 드릴게요!"
    ),
    "finn": (
        " 오 그대여, 운명의 부름이 들리지 않으시오? "
        "「{title}」 — {description} "
        "그 길 끝에 {reward}의 영광이 있으리라."
    ),
    "bernhardt": (
        " 흠, 거래 하나 제안드리지요. "
        "「{title}」 — {description} "
        "성사되면 {reward} 지불해드리겠습니다."
    ),
}

PROMPT_FACT = (
    "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "사람 이름과 장소 이름은 절대 바꾸지 말고, 어조만 너답게 바꾸세요. "
    "다른 설명이나 라벨은 붙이지 마세요.\n\n"
    "사실: {memory}\n\n"
    "당신의 한 마디:"
)

QUEST_EXTRACT_PROMPT = """다음 NPC가 플레이어에게 어떤 행동을 제안(quest)하는지 분석하시오.

NPC: {npc} ({role})
플레이어 발언: {user_text}
NPC 응답: {response}
NPC가 떠올린 정보: {memory_text}

NPC가 플레이어에게 행동(조사·추적·거래·모험·확인 등)을 제안하면 quest로 추출.
단순 정보 제공이나 일상 대화면 has_quest: false.

JSON 한 개만 출력하시오 (다른 설명 절대 금지, 한국어로):
{{"has_quest": true, "title": "10자 내외 짧은 제목", "description": "20-40자 플레이어가 할 일", "reward": "보상 키워드"}}
또는
{{"has_quest": false}}

JSON:"""

PROMPT_DIALOGUE = (
    "플레이어가 너에게 다음과 같이 말했다. 이 말에 담긴 사실 정보를 다른 마을 사람에게 "
    "한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "사람 이름과 장소 이름은 절대 바꾸지 말고, 어조만 너답게 바꾸세요. "
    "사실 정보가 없거나 단순 인사면 빈 답변을 출력하세요.\n\n"
    "플레이어 발언: {memory}\n\n"
    "당신의 한 마디:"
)


class TrustTracker:
    """NPC별로 플레이어 신뢰도(0-100) 추적.

    - default 30 (지인)
    - 대화 1회 +1 (자연 증가)
    - 긍정 키워드 +1 추가
    - 부정 키워드 -5
    - quest 완수 +10
    """
    DEFAULT_TRUST = 30
    POSITIVE_KEYWORDS = [
        "감사", "고마", "도와", "도울", "부탁", "안녕", "수고",
        "잘 부탁", "친절", "고맙", "최고", "멋져", "훌륭",
    ]
    NEGATIVE_KEYWORDS = [
        "꺼져", "닥쳐", "바보", "멍청", "싫어", "거짓말", "사기",
    ]

    def __init__(self):
        self._trust: dict[str, int] = {}
        self._interactions: dict[str, int] = {}

    def get(self, npc: str) -> int:
        return self._trust.get(npc, self.DEFAULT_TRUST)

    def label(self, npc: str) -> str:
        t = self.get(npc)
        if t < 20: return "낯선 사람"
        if t < 50: return "지인"
        if t < 80: return "친구"
        return "절친"

    def disclosure_hint(self, npc: str) -> str:
        """system prompt에 들어갈 친밀도 지침 — 한 줄."""
        t = self.get(npc)
        if t < 20:
            return "친밀도 낮음 (낯선 사람). 표면적·공개 정보만. 개인사·비밀 절대 X."
        if t < 50:
            return "친밀도 보통 (지인). 일반 정보 자유롭게. 개인사는 자제."
        if t < 80:
            return "친밀도 높음 (친구). 개인사·과거 이야기 공유 가능."
        return "친밀도 매우 높음 (절친). 깊은 비밀·트라우마까지 털어놓을 수 있음."

    def on_player_turn(self, npc: str, user_text: str) -> int:
        """플레이어 발화 후 신뢰도 업데이트. 변화량(delta) 반환."""
        delta = 1  # 대화 1회 +1
        if any(k in user_text for k in self.POSITIVE_KEYWORDS):
            delta += 1
        if any(k in user_text for k in self.NEGATIVE_KEYWORDS):
            delta = -5
        new_val = max(0, min(100, self.get(npc) + delta))
        self._trust[npc] = new_val
        self._interactions[npc] = self._interactions.get(npc, 0) + 1
        return delta

    def on_quest_complete(self, npc: str) -> int:
        """Quest 완수 시 +10."""
        old = self.get(npc)
        new_val = min(100, old + 10)
        self._trust[npc] = new_val
        return new_val - old

    def set(self, npc: str, value: int):
        self._trust[npc] = max(0, min(100, value))

    def snapshot(self) -> dict:
        return {
            npc: {"trust": self.get(npc), "label": self.label(npc),
                  "interactions": self._interactions.get(npc, 0)}
            for npc in set(list(self._trust.keys()) + list(self._interactions.keys()))
        }


class NpcServer:
    def __init__(
        self,
        adapters_dir: Path,
        chroma_dir: Path,
        relations_path: Path | None = None,
        characters: list[str] | None = None,
        retrieval_k: int = 1,  # 3 → 1: 회상 컨텍스트 줄여 페르소나 안정화
        use_lora: bool = False,  # LoRA 폐기 결정 후 default False. ablation용으로 True 가능.
        use_memory: bool = False,  # 회상 비활성 default. 단계적 접근: 페르소나만 → 메모리 → 전파.
    ):
        self.characters = characters or DEFAULT_CHARACTERS
        self.retrieval_k = retrieval_k
        self.use_lora = use_lora
        self.use_memory = use_memory

        # use_lora=True일 때만 어댑터 검증
        if use_lora:
            adapter_paths = {npc: adapters_dir / npc for npc in self.characters}
            for npc, p in adapter_paths.items():
                if not p.exists():
                    raise FileNotFoundError(f"어댑터 없음: {p}")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("[engine] 토크나이저 + 베이스 모델 로딩...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if use_lora:
            first = self.characters[0]
            print(f"[engine] LoRA 로딩 ({len(self.characters)}종)...")
            self.model = PeftModel.from_pretrained(
                base, str(adapter_paths[first]), adapter_name=first
            )
            for npc in self.characters[1:]:
                self.model.load_adapter(str(adapter_paths[npc]), adapter_name=npc)
        else:
            print("[engine] LoRA 비활성: 베이스 EXAONE + system prompt만 사용")
            self.model = base
        self.model.eval()

        print("[engine] 메모리 store/retriever 초기화...")
        self.stores: dict[str, MemoryStore] = {}
        self.retrievers: dict[str, MemoryRetriever] = {}
        for npc in self.characters:
            store = MemoryStore(npc_name=npc, base_dir=chroma_dir / npc)
            self.stores[npc] = store
            self.retrievers[npc] = MemoryRetriever(store)
            print(f"  {npc}: 메모리 {store.count()}개")

        # 정보 전파 그래프
        if relations_path is None:
            relations_path = Path(__file__).resolve().parents[2] / "configs" / "relations.yaml"
        if relations_path.exists():
            self.graph = RelationGraph.load(relations_path)
            print(f"[engine] 관계 그래프 로드 ({len(self.graph.edges())} edges)")
        else:
            self.graph = None
            print("[engine] 관계 그래프 없음, propagation 비활성")

        self.day = 0
        self._transform_cache: dict = {}
        self.trust = TrustTracker()
        self.quests = QuestTracker()

        # 페르소나 정의 로드 (system prompt에 사용)
        personas_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "test_prompts.yaml"
        try:
            with personas_path.open(encoding="utf-8") as f:
                self.personas = yaml.safe_load(f).get("personas", {})
            print(f"[engine] 페르소나 정의 로드 ({len(self.personas)}종)")
        except Exception as e:
            self.personas = {}
            print(f"[engine] 페르소나 정의 로드 실패: {e}")

    def _build_system_prompt(self, npc: str) -> str:
        """NPC별 system prompt — 페르소나 마커 + 어휘 + 다른 NPC 직업.

        use_memory=True일 때만 회상 활용 안내 추가.

        설계 노트:
        - 너무 길면 출력 깨짐 (영어 leak, 템플릿 토큰 leak 회귀 발생함). 보수적으로 유지.
        - vocabulary 추가 — 페르소나 어휘 reinforce.
        """
        p = self.personas.get(npc, {})
        desc = p.get("description", "")
        m = p.get("markers", {})
        tone = ", ".join(m.get("tone", []))
        avoid = ", ".join(m.get("avoid", []))
        starts = ", ".join(m.get("speech_start", []))
        ends = ", ".join(m.get("speech_end", []))
        vocab = ", ".join(m.get("vocabulary", []))

        # 다른 NPC 직업명만 (description 첫 마디만 추출)
        role_brief = {
            n: self.personas[n].get("description", "").split(".")[0].strip()
            for n in self.personas if n != npc
        }
        others = ", ".join(f"{n}={role_brief[n]}" for n in role_brief if role_brief[n])

        memory_hint = (
            "사용자 메시지에 [참고 기억: ...] 이 있으면 그건 너의 회상이오. "
            "**플레이어의 '질문:' 부분에 직접 답하는 것이 최우선**. "
            "회상이 질문과 관련 있을 때만 슬쩍 언급하시오 (예: '...라더군'). "
            "관련 없으면 무시하고 질문에만 답하시오. "
            if self.use_memory else ""
        )

        strict_rule = NPC_STRICT_RULES.get(npc, "")
        trust_hint = self.trust.disclosure_hint(npc)

        return (
            f"당신은 {npc}입니다. {desc}\n"
            f"어조: {tone}. 피해야 할 것: {avoid}.\n"
            f"자주 쓰는 어휘: {vocab}.\n"
            f"{strict_rule}\n"
            f"다른 마을 사람: {others}. 이들의 이름과 직업을 절대 바꾸지 마시오 "
            "(예: mathilda를 마트닐라로 변형 금지).\n"
            f"{trust_hint}\n"
            f"{memory_hint}"
            "**반드시 한 문장**으로만 답하시오. 절대 길게 X. "
            "한국어로만 답하시오. 영어/외국어/이모지/특수문자(♪✨ 등) 절대 금지."
        )

    # ---------- 응답 생성 ----------
    def respond(
        self,
        npc: str,
        user_text: str,
        history: list[dict] | None = None,
        max_new_tokens: int = 200,
    ) -> dict:
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")

        t0 = time.time()
        if self.use_memory:
            # 플레이어 발화(DIALOGUE)도 회상 가능 — "내 이름은 ~" 같은 자기소개 기억.
            # 직전 turn의 자기 메모리 회상은 retriever의 min_similarity로 자연스럽게 필터됨.
            retrieved = self.retrievers[npc].search(
                user_text, k=self.retrieval_k
            )
            augmented = build_user_prompt(retrieved, user_text)
        else:
            retrieved = []
            augmented = user_text

        # NPC별 system prompt 주입 (history 없을 때만, 있으면 첫 system 유지)
        messages = list(history or [])
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": self._build_system_prompt(npc)})
        messages.append({"role": "user", "content": augmented})

        # NPC별 generation 파라미터
        gp = GEN_PARAMS.get(npc, {
            "temperature": 0.5, "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
        })

        if self.use_lora:
            self.model.set_adapter(npc)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=gp["max_new_tokens"],
                do_sample=True,
                temperature=gp["temperature"],
                top_p=0.9,
                top_k=50,
                repetition_penalty=gp["repetition_penalty"],
                no_repeat_ngram_size=gp["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        text = _clean_response(text, npc=npc)  # emoji/특수문자/NPC별 어미 정리
        latency_ms = int((time.time() - t0) * 1000)

        # ─────────────────────────────────────────────────────────
        # LLM 자동 Quest 생성 — 일단 보류 (2026-05-13).
        # 이유: NPC 응답 뒤에 quest template이 앞뒤 맥락 없이 갑자기 붙는 케이스가
        #       많아 어색함. 향후 quest pool + LLM 선택 하이브리드 방식으로 전환 예정.
        # 복원: 아래 주석 블록 해제 + quest=None 줄 제거.
        # ─────────────────────────────────────────────────────────
        quest = None
        # high_imp = any(m["importance"] >= 7 for m in retrieved) if retrieved else False
        # if high_imp:
        #     quest = self._extract_quest(npc, user_text, text, retrieved)
        #     if quest is not None:
        #         template = NPC_QUEST_INTRO.get(npc)
        #         stripped = text.rstrip()
        #         ends_with_question = stripped.endswith("?") or stripped.endswith("?")
        #         too_short = len(stripped) < 12
        #         if template and not ends_with_question and not too_short:
        #             quest_intro = template.format(
        #                 title=quest.get("title", ""),
        #                 description=quest.get("description", ""),
        #                 reward=quest.get("reward", "응당한") or "응당한",
        #             )
        #             text = text.rstrip() + quest_intro
        #             text = _clean_response(text, npc=npc)

        # 플레이어 발화를 NPC의 DIALOGUE 메모리로 저장 (다음 tick에서 전파 후보)
        if self.use_memory:
            self._save_player_turn(npc, user_text)

        # 신뢰도 업데이트 (응답 후, 다음 대화부터 영향)
        trust_delta = self.trust.on_player_turn(npc, user_text)

        return {
            "npc": npc,
            "text": text,
            "memories_used": [
                {
                    "text": m["text"],
                    "importance": m["importance"],
                    "source": m["metadata"].get("source", "unknown"),
                }
                for m in retrieved
            ],
            "quest": quest,
            "trust": self.trust.get(npc),
            "trust_label": self.trust.label(npc),
            "trust_delta": trust_delta,
            "latency_ms": latency_ms,
        }

    def complete_quest(self, npc: str, quest_id: str | None = None) -> dict:
        """Quest 완수 시 호출 — 신뢰도 +10. quest_id 명시되면 상태 'completed'."""
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")
        delta = self.trust.on_quest_complete(npc)
        if quest_id:
            self.quests.mark_completed(quest_id)
        return {
            "npc": npc,
            "quest_id": quest_id,
            "trust": self.trust.get(npc),
            "trust_label": self.trust.label(npc),
            "trust_delta": delta,
        }

    def get_greeting(self, npc: str) -> str:
        """현재 신뢰도에 맞는 NPC greeting 무작위 1개 반환.

        LLM 호출 없이 즉시 — F키로 처음 대화 시작 시 NPC가 먼저 한마디.
        """
        if npc not in self.characters:
            return ""
        label = self.trust.label(npc)
        greetings = NPC_GREETINGS.get(npc, {}).get(label, [])
        if not greetings:
            return ""
        return random.choice(greetings)

    def get_dialogue_opener(self, npc: str) -> dict:
        """대화 시작 시 NPC가 먼저 말하는 첫 대사.

        조건:
        - 줄 수 있는 quest(trust 충족 + available) 있으면 → quest intro + quest 정보
        - 없으면 → 일반 greeting
        반환: {text: str, quest: dict | None}
        """
        if npc not in self.characters:
            return {"text": "", "quest": None}

        current_trust = self.trust.get(npc)
        pickable = self.quests.get_pickable_quest(npc, current_trust)

        if pickable is not None:
            # Quest intro로 대화 시작 + quest 정보 표시
            intro = pickable.get("intro", "")
            template = NPC_QUEST_INTRO.get(npc, "「{title}」 — {description} 보상: {reward}.")
            quest_body = template.format(
                title=pickable.get("title", ""),
                description=pickable.get("description", ""),
                reward=pickable.get("reward", ""),
            )
            text = (intro + quest_body).strip()
            # offered로 상태 변경 (한 번 보여주면 다시는 자동 제안 X)
            self.quests.mark_offered(pickable["id"])
            return {
                "text": text,
                "quest": {
                    "id": pickable["id"],
                    "title": pickable.get("title", ""),
                    "description": pickable.get("description", ""),
                    "reward": pickable.get("reward", ""),
                    "giver": npc,
                },
            }

        # Quest 없으면 일반 greeting
        return {"text": self.get_greeting(npc), "quest": None}

    # ---------- NPC-NPC 자율 대화 (Park et al. 2023 스타일) ----------
    def _generate_for_npc(
        self,
        npc: str,
        messages: list[dict],
        max_new_tokens: int | None = None,
    ) -> str:
        """공통 generate 헬퍼. messages = [system, user, assistant, ...] chat 포맷."""
        gp = GEN_PARAMS.get(npc, {
            "temperature": 0.5, "max_new_tokens": 120,
            "repetition_penalty": 1.15, "no_repeat_ngram_size": 4,
        })
        if self.use_lora:
            self.model.set_adapter(npc)
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens or gp["max_new_tokens"],
                do_sample=True,
                temperature=gp["temperature"],
                top_p=0.9,
                top_k=50,
                repetition_penalty=gp["repetition_penalty"],
                no_repeat_ngram_size=gp["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        return _clean_response(text, npc=npc)

    def simulate_conversation(
        self,
        npc_a: str,
        npc_b: str,
        topic: str | None = None,
        num_turns: int = 3,
    ) -> dict:
        """두 NPC가 자율적으로 대화하고 결과를 각자 메모리에 저장.

        Park et al. (Generative Agents) 스타일 NPC-NPC 대화.

        - num_turns: 각 NPC가 발화하는 횟수 (총 발화 ≤ num_turns × 2)
        - topic: 대화 주제 시드. None이면 npc_a 메모리에서 1개 선정.
        - 결과 대화 전체를 양쪽 DIALOGUE 메모리에 저장.
        """
        if npc_a not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc_a}")
        if npc_b not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc_b}")
        if npc_a == npc_b:
            raise ValueError("같은 NPC끼리 대화 불가")

        # topic 자동 선정: npc_a 메모리 중 importance 높은 것 위주
        if topic is None:
            try:
                # importance 6+ memory 위주 검색
                cand = self.retrievers[npc_a].search("마을 사건 소식", k=3)
                if cand:
                    chosen = random.choice(cand)["text"]
                    # 플레이어 발화는 출처 명시해서 topic 구성 — NPC가 자기 말로 오인 방지
                    if chosen.startswith("플레이어가 말했다: "):
                        content = chosen[len("플레이어가 말했다: "):][:80]
                        topic = f"플레이어가 나한테 '{content}'라고 했던 일"
                    else:
                        topic = chosen
                else:
                    topic = "마을 근황"
            except Exception:
                topic = "마을 근황"

        # 한국어 NPC 이름 (system prompt 영문 leak 회피)
        ko_name = {
            "elias": "엘리아스", "hermann": "헤르만", "mathilda": "마틸다",
            "finn": "핀", "bernhardt": "베른하르트",
        }
        ko_a = ko_name.get(npc_a, npc_a)
        ko_b = ko_name.get(npc_b, npc_b)

        turns: list[dict] = []

        # 시작 turn: npc_a가 화제 던지기
        opener_prompt = (
            f"당신은 지금 마을에서 {ko_b}을(를) 만났습니다. "
            f"다음 화제에 대해 자연스럽게 한두 문장으로 말 거시오 "
            f"(인사 + 짧은 화제 제기). 어조는 페르소나대로.\n"
            f"화제: {topic}"
        )
        a_messages = [
            {"role": "system", "content": self._build_system_prompt(npc_a)},
            {"role": "user", "content": opener_prompt},
        ]
        first_text = self._generate_for_npc(npc_a, a_messages)
        turns.append({"speaker": npc_a, "speaker_ko": ko_a, "text": first_text})

        # 각 NPC 시점의 대화 history (chat format)
        a_history = [
            {"role": "system", "content": self._build_system_prompt(npc_a)},
            {"role": "user", "content": opener_prompt},
            {"role": "assistant", "content": first_text},
        ]
        b_history = [
            {"role": "system", "content": self._build_system_prompt(npc_b)},
            {"role": "user", "content": f"{ko_a}가 당신에게 말했다: \"{first_text}\""},
        ]

        last_speaker = npc_a
        # 남은 발화 횟수 = num_turns*2 - 1 (이미 1번 발화함)
        for _ in range(num_turns * 2 - 1):
            responder = npc_b if last_speaker == npc_a else npc_a
            other = npc_a if responder == npc_b else npc_b
            ko_responder = ko_b if responder == npc_b else ko_a
            ko_other = ko_a if responder == npc_b else ko_b

            hist = b_history if responder == npc_b else a_history
            response = self._generate_for_npc(responder, hist)
            turns.append({
                "speaker": responder, "speaker_ko": ko_responder, "text": response
            })

            # 양쪽 history 갱신
            if responder == npc_b:
                b_history.append({"role": "assistant", "content": response})
                a_history.append({
                    "role": "user",
                    "content": f"{ko_b}가 답했다: \"{response}\"",
                })
            else:
                a_history.append({"role": "assistant", "content": response})
                b_history.append({
                    "role": "user",
                    "content": f"{ko_a}가 답했다: \"{response}\"",
                })

            last_speaker = responder

        # 메모리 저장: 각 NPC가 본인 시점에서 대화를 기억
        # (LLM 요약 생략하고, 대화 일부를 그대로 저장 — 간단하게)
        convo_text_for_a = self._format_conversation_memory(turns, ko_other=ko_b)
        convo_text_for_b = self._format_conversation_memory(turns, ko_other=ko_a)

        entry_a = MemoryEntry(
            id=f"conv_{uuid.uuid4().hex[:8]}",
            text=convo_text_for_a,
            importance=6,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.CONVERSATION,
            metadata={
                "npc_conversation": True, "other_npc": npc_b,
                "day": self.day, "topic": topic[:80],
            },
        )
        self.stores[npc_a].add(entry_a)

        entry_b = MemoryEntry(
            id=f"conv_{uuid.uuid4().hex[:8]}",
            text=convo_text_for_b,
            importance=6,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.CONVERSATION,
            metadata={
                "npc_conversation": True, "other_npc": npc_a,
                "day": self.day, "topic": topic[:80],
            },
        )
        self.stores[npc_b].add(entry_b)

        return {
            "npc_a": npc_a,
            "npc_b": npc_b,
            "topic": topic[:120],
            "turns": turns,
            "memory_saved": True,
            "day": self.day,
        }

    @staticmethod
    def _format_conversation_memory(turns: list[dict], ko_other: str) -> str:
        """대화 전체를 한 메모리 텍스트로 압축. {ko_other}와 나눈 대화로 기록."""
        lines = []
        for t in turns:
            lines.append(f"{t['speaker_ko']}: {t['text'][:120]}")
        body = " / ".join(lines)
        return f"{ko_other}와 대화: {body}"

    def pick_random_pair(self) -> tuple[str, str] | None:
        """관계 그래프 edge 중 1쌍 무작위 선택 (NPC-NPC 자율 대화용).

        graph가 없으면 character list에서 2명 무작위 선정.
        """
        if self.graph is not None:
            edges = list(self.graph.edges())
            if edges:
                a, b, _freq = random.choice(edges)
                return a, b
        if len(self.characters) < 2:
            return None
        a, b = random.sample(self.characters, 2)
        return a, b

    def _extract_quest(
        self, npc: str, user_text: str, response: str, retrieved: list
    ) -> dict | None:
        """NPC 응답 + 회상 메모리에서 quest 객체 추출.

        LLM 별도 호출 (deterministic) → JSON 파싱 → quest dict 반환.
        실패하거나 has_quest=false면 None.
        """
        role_brief = (
            self.personas.get(npc, {}).get("description", "").split(".")[0].strip()
        )
        memory_text = " / ".join(m["text"][:80] for m in retrieved)

        prompt = QUEST_EXTRACT_PROMPT.format(
            npc=npc, role=role_brief,
            user_text=user_text, response=response,
            memory_text=memory_text,
        )

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=80,  # JSON 한 줄이면 충분
                do_sample=False,  # quest는 deterministic
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()

        # JSON 추출 — { ... } 패턴 찾기 (가장 큰 매칭)
        match = re.search(r"\{[^{}]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        if not parsed.get("has_quest", False):
            return None

        return {
            "title": str(parsed.get("title", ""))[:30].strip(),
            "description": str(parsed.get("description", ""))[:120].strip(),
            "reward": str(parsed.get("reward", ""))[:30].strip(),
            "giver": npc,
        }

    def _save_player_turn(self, npc: str, user_text: str) -> None:
        text = user_text.strip()
        if len(text) < 4:
            return  # 너무 짧은 감탄사만 제외

        # 질문문 vs 평서문 분기
        is_question = "?" in text or any(
            text.endswith(suf) for suf in ["어요", "지요", "나요", "까", "야"]
        )
        # 사실 보고 키워드 (강한 fact 신호)
        fact_kw = ["나타났", "사라졌", "잡았", "봤", "들었", "있었", "갔다", "왔다", "했다",
                   "됐다", "당했", "보았", "만났", "들었어", "가봤", "도와", "받았"]
        has_fact = any(kw in text for kw in fact_kw)
        # 자기소개/personal info: 이름·정체성 정보
        personal_kw = ["내 이름", "제 이름", "이름은", "이라고 해", "이라고 한다",
                       "라고 합니다", "라고 불러", "나는 ", "저는 ", "내가 ", "제가 "]
        has_personal = any(kw in text for kw in personal_kw)

        # importance 매핑 — 평서문은 모두 전파 후보(threshold 7+) 보장.
        # 프로젝트 핵심: "플레이어 → A 발화" → propagation → "B가 알고 대화 이어감".
        if has_personal:
            importance = 10  # 자기소개: 최우선 회상 + 강한 전파
        elif is_question:
            importance = 4   # 질문은 전파 X
        elif has_fact:
            importance = 9   # 사실 보고: 시드보다 강하게 전파
        elif len(text) >= 15:
            importance = 8   # 일반 평서문: 전파 후보 보장 (6→8)
        else:
            importance = 7   # 짧은 평서문: 전파 후보 진입 (5→7)

        entry = MemoryEntry(
            id=f"dlg_{uuid.uuid4().hex[:8]}",
            text=f"플레이어가 말했다: {text}",
            importance=importance,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.DIALOGUE,
            metadata={
                "player": True, "is_question": is_question,
                "has_fact": has_fact, "has_personal": has_personal,
            },
        )
        self.stores[npc].add(entry)

    # ---------- PropagationSimulator transformer 인터페이스 ----------
    def transform(
        self,
        sender_npc: str,
        memory_text: str,
        source: str = "observation",
        max_new_tokens: int = 50,  # 80→50: 한 줄 짧게 (속도 ↑)
    ) -> str:
        """sender NPC의 어조로 메모리를 다시 표현 (정보 전파 시 사용).

        source가 'dialogue'면 플레이어 발언에서 사실 정보 추출용 prompt,
        그 외에는 사실 그대로 전달용 prompt 사용.
        """
        cache_key = (sender_npc, memory_text, source)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        if self.use_lora:
            self.model.set_adapter(sender_npc)
        # 메모리 prefix 제거: 더 깨끗한 입력으로
        clean = memory_text
        if clean.startswith("플레이어가 말했다: "):
            clean = clean[len("플레이어가 말했다: "):]
        elif "한테 들었다: " in clean:
            clean = clean.split("한테 들었다: ", 1)[1]

        template = PROMPT_DIALOGUE if source == "dialogue" else PROMPT_FACT
        prompt = template.format(memory=clean)
        # use_lora=False: 페르소나 변형이 LoRA 없이 system prompt에만 의존하므로 추가
        # use_lora=True: LoRA가 페르소나 가중치 가지고 있어 system 없이도 작동 (기존)
        messages = []
        if not self.use_lora:
            messages.append({"role": "system", "content": self._build_system_prompt(sender_npc)})
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].strip().strip('"').strip("'")
        text = _clean_response(text, npc=sender_npc)  # NPC별 어미 정리
        if not text:
            text = clean  # fallback: 원문 그대로
        self._transform_cache[cache_key] = text
        return text

    # ---------- 시간 진행 (정보 전파 + NPC-NPC 자율 대화) ----------
    def tick(
        self,
        day: int | None = None,
        npc_conversation: bool = True,
        npc_conversation_turns: int = 1,
        fast: bool = True,  # 빠른 모드 — propagation transform 생략 (LLM 호출 ↓ 큰 속도)
    ) -> dict:
        """하루치 정보 전파 시뮬레이션 + 1쌍 NPC-NPC 자율 대화.

        - 1단계: propagation (전파). fast=True면 페르소나 변환 LLM 생략.
        - 2단계: graph 무작위 페어 → simulate_conversation (자율 대화는 유지).
        """
        if self.graph is None:
            return {"day": self.day, "events": [], "error": "관계 그래프 없음"}
        if day is None:
            self.day += 1
            day = self.day
        else:
            self.day = day

        # 1단계: propagation
        sim = PropagationSimulator(
            graph=self.graph,
            stores=self.stores,
            transformer=self,
            use_transform=not fast,  # fast 모드면 transform 생략
        )
        events = sim.tick(day)

        # 2단계: NPC-NPC 자율 대화 1쌍 (옵션)
        conversation_result = None
        if npc_conversation:
            pair = self.pick_random_pair()
            if pair is not None:
                try:
                    conversation_result = self.simulate_conversation(
                        pair[0], pair[1], num_turns=npc_conversation_turns
                    )
                except Exception as e:
                    print(f"[tick] NPC-NPC 대화 실패: {e}")

        # 3단계: 메모리 정리 — NPC별 최대 보유 메모리 제한 (속도 유지)
        # seed는 보존, importance 낮고 오래된 것부터 삭제.
        pruned_total = 0
        for npc in self.characters:
            try:
                pruned_total += self.stores[npc].prune(max_keep=60)
            except Exception as e:
                print(f"[tick] {npc} 메모리 정리 실패: {e}")
        if pruned_total > 0:
            print(f"[tick] 메모리 정리: 총 {pruned_total}개 삭제 (NPC당 60개 유지)")

        return {
            "day": day,
            "events": events,
            "conversation": conversation_result,
        }

    def memory_counts(self) -> dict[str, int]:
        return {npc: self.stores[npc].count() for npc in self.characters}
