"""5종 NPC를 한 프로세스에서 추론하는 통합 엔진.

베이스 EXAONE 1개 + 5종 LoRA 어댑터를 PEFT의 load_adapter / set_adapter로
스위칭하면서 사용한다. 메모리 store/retriever도 NPC별로 보유.
또한 NPC 간 정보 전파(시간 기반)도 동일 프로세스에서 수행한다.
"""

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
    # 영문 이름 — lookaround로 한글-영문 경계도 매칭 ("hermann님" 같은 leak 잡기)
    (re.compile(r"(?<![a-zA-Z])[Hh]err?mann(?![a-zA-Z])"), "헤르만"),
    (re.compile(r"(?<![a-zA-Z])[Mm]athilda(?![a-zA-Z])"), "마틸다"),
    (re.compile(r"(?<![a-zA-Z])[Bb]ernhardt(?![a-zA-Z])"), "베른하르트"),
    (re.compile(r"(?<![a-zA-Z])[Ee]lias(?![a-zA-Z])"), "엘리아스"),
    (re.compile(r"(?<![a-zA-Z])[Ff]inn(?![a-zA-Z])"), "핀"),
    # 한글 변형도 통일
    (re.compile(r"베르나르드|베르나르트|베른하르크|베른하르타르|베른하드"), "베른하르트"),
    (re.compile(r"마트닐라|마트일다|수학틸라|수학틸다|마틸달라|마틸 다라"), "마틸다"),
    (re.compile(r"헤르몬|헤른|헤르몽|허르만|허먼|헤르몰"), "헤르만"),
    # 핀 변형 — 조사까지 함께 정정 (핀나가 → 핀이)
    (re.compile(r"핀나가"), "핀이"),
    (re.compile(r"핀나는"), "핀은"),
    (re.compile(r"핀나를"), "핀을"),
    (re.compile(r"핀나"), "핀"),
    (re.compile(r"엘리어스|엘시아스|엘리아斯|엘리아\s+아찌|엘리아\s+아저씨|엘리아씨|엘리아\s+씨|엘리아스님이시군요"), "엘리아스"),
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
    # 어미 오타형: ~군오 → ~군요 ("떠나셨군오!")
    (re.compile(r"([가-힣])군오(?=[\s.,!?]|$)"), lambda m: m.group(1) + "군요"),
    # ~군요만 / ~더군요만 → ~군요 ("그러더군요만,")
    (re.compile(r"(더군요|군요)만(?=[\s.,!?]|$)"), lambda m: m.group(1)),
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
        # 응답 시작 어색 감탄사 정리 ("네/예/아호/아하" → "흠")
        (re.compile(r"^(아호|아하|어허|오호|에헴|네네|넵|예예|네|예)([\s,]+)"), r"흠\2"),
        # "(어휘)소이다만" → "(어휘)소만" (어색한 이중 어미)
        (re.compile(r"소이다만\b"), "소만"),
        (re.compile(r"([가-힣])소이다만\b"), lambda m: m.group(1) + "소만"),
        # "(어휘)이오나" / "(어휘)오나" → "(어휘)이오만" / "(어휘)오만"
        (re.compile(r"이오나\b"), "이오만"),
        (re.compile(r"([가-힣])오나\b(?!요)"), lambda m: m.group(1) + "오만"),
        # "당신이란 분께선" 또 leak 케이스
        (re.compile(r"당신이란\s+분께선\b"), "그대는"),
        # 격식체 한자어 표현 → Elias 자연 어조
        (re.compile(r"인지하였으나\b"), "들었으나"),
        (re.compile(r"인지하였소\b"), "알겠소"),
        (re.compile(r"인지하오\b"), "알겠소"),
        (re.compile(r"확인하였으나\b"), "들었으나"),
        (re.compile(r"한계가\s+있으니\b"), "어려우니"),
        (re.compile(r"한계가\s+있소\b"), "어렵소"),
        (re.compile(r"제공에는\s+한계가\s+있"), "어렵"),
        # "추가 정보 제공" / "추가 정보 부탁" → 자연
        (re.compile(r"추가\s+정보\s+제공"), "더 자세히 말씀"),
        (re.compile(r"부탁드리겠소"), "부탁하오"),
        # 사물에 "계시오" 잘못 사용 → "있소"
        (re.compile(r"(부분|일|것|점|곳|이야기|사실|정보|문제)이?\s+계시오"), lambda m: m.group(1) + "이 있소"),
        (re.compile(r"(부분|일|것|점|곳|이야기|사실|정보|문제)이?\s+계시는"), lambda m: m.group(1) + "이 있는"),
        # "본인의" 모호함 → "그대의" 명확 (Elias가 plyaer 칭함)
        (re.compile(r"본인의\s+이름은\b"), "그대의 이름은"),
        (re.compile(r"본인의\s+"), "그대의 "),
        # 한글 이름 + 띄어쓰기 + 어미 → 붙임 ("반욱현 이오" → "반욱현이오")
        (re.compile(r"([가-힣]{2,4})\s+(이오|이라|구려|소|오)\b"), lambda m: m.group(1) + m.group(2)),
        # "(어휘)셨는지" / "(어휘)셨으나" 잘못된 주체 사극체 (사용자가 주체)
        (re.compile(r"청하셨는지\b"), "청하시는지"),
        (re.compile(r"하셨는지\b"), "하시는지"),
        (re.compile(r"하셨으나\b"), "하시나"),
        # 응답 끝 "감사" 단독 미완 → "고맙소" 추가
        (re.compile(r"감사$"), "감사하오"),
        (re.compile(r"감사\.$"), "감사하오."),
        # "소였으나" / "소였소" 어색 패턴 (과거형 leak)
        (re.compile(r"([가-힣])소였으나\b"), lambda m: m.group(1) + "소만"),
        (re.compile(r"([가-힣])소였소\b"), lambda m: m.group(1) + "소"),
        # "당신이란 분께서" / "당신이란 분" 어색 표현 → "당신" / "그대"
        (re.compile(r"당신이란\s+분께서\b"), "그대"),
        (re.compile(r"당신이란\s+분\b"), "그대"),
        # "확인되오" → OK이지만 "사실 자체만큼은 확인되오" 같은 장황한 표현 정리
        (re.compile(r"사실\s+자체만큼은\b"), "사실은"),
        # "(어휘)겠다 하오" / "(어휘)겠다고 하오" → "(어휘)겠소"
        (re.compile(r"([가-힣])겠다\s+하오\b"), lambda m: m.group(1) + "겠소"),
        (re.compile(r"([가-힣])겠다고\s+하오\b"), lambda m: m.group(1) + "겠소"),
        # "보겠다 하오" 같은 일반 패턴
        (re.compile(r"보겠다\s+하오\b"), "보겠소"),
        (re.compile(r"하겠다\s+하오\b"), "하겠소"),
        # "(어휘)다 하오" 끝 정리 (이중 어미)
        (re.compile(r"([가-힣])다\s+하오\b"), lambda m: m.group(1) + "오"),
        # "맞소이까" / "맞소이다" → "맞소"
        (re.compile(r"맞소이까\b"), "맞소"),
        (re.compile(r"맞소이다\b"), "맞소"),
        # "주시리오시다" / "주시리오" 같은 삼중 어미
        (re.compile(r"주시리오시다\b"), "주시오"),
        (re.compile(r"([가-힣])시리오시다\b"), lambda m: m.group(1) + "시오"),
        # 일반 "(어미)시다" 사극체 leak (elias는 ~오/~이오/~구려만)
        (re.compile(r"([가-힣])시리오?시다\b"), lambda m: m.group(1) + "시오"),
        (re.compile(r"하오시다\b"), "하오"),
        (re.compile(r"되오시다\b"), "되오"),
        (re.compile(r"이오시다\b"), "이오"),
        # "이오 되오" / "이오 하오" 같은 어미 중복 leak
        (re.compile(r"이오\s+되오\b"), "이오"),
        (re.compile(r"이오\s+하오\b"), "이오"),
        (re.compile(r"하오\s+되오\b"), "하오"),
        (re.compile(r"하오\s+이오\b"), "하오"),
        (re.compile(r"([가-힣])오\s+오\b"), lambda m: m.group(1) + "오"),
        # NPC 이름 + "이오" 띄어쓰기 정리 ("엘리아스 이오" → "엘리아스이오")
        (re.compile(r"(엘리아스|헤르만|마틸다|핀|베른하르트)\s+이오\b"),
         lambda m: m.group(1) + "이오"),
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
        (re.compile(r"([가-힣])십니까\b"), lambda m: m.group(1) + "시는가"),
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
        # 흔한 중복 표현
        (re.compile(r"손님님\b"), "손님"),
        (re.compile(r"분님\b"), "분"),
        # 잘못된 주체 사극체 (사용자가 주체일 때 "(어휘)셨으니까")
        (re.compile(r"못하셨으니까\b"), "못했으니까"),
        (re.compile(r"못하셨으니\b"), "못했으니"),
        # 반말 leak → 친근 존댓말 (Mathilda는 ~어요/~네요/~죠)
        (re.compile(r"들었니\?"), "들으셨어요?"),
        (re.compile(r"있니\?"), "있어요?"),
        (re.compile(r"가니\?"), "가세요?"),
        (re.compile(r"오니\?"), "오세요?"),
        (re.compile(r"([가-힣])니\?"), lambda m: m.group(1) + "어요?"),
        # 응답 끝 반말 → "~에요/~어요"
        (re.compile(r"안심이야(?=[\s.,!?]|$)"), "안심이에요"),
        (re.compile(r"걱정이야(?=[\s.,!?]|$)"), "걱정이에요"),
        (re.compile(r"([가-힣])이야(?=[\s.,!?]|$)"), lambda m: m.group(1) + "이에요"),
        (re.compile(r"([가-힣])이지(?=[\s.,!?]|$)"), lambda m: m.group(1) + "이죠"),
        # "없대" / "있대" 같은 어색한 줄임 → 자연
        (re.compile(r"없대\s+보이는데"), "없어 보이는데"),
        (re.compile(r"있대\s+보이는데"), "있어 보이는데"),
        (re.compile(r"([가-힣])대\s+보이는데"), lambda m: m.group(1) + "어 보이는데"),
        # mathilda는 "~네요/~죠/~어요" 자연. "~십니까" 어색 → "~세요"
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "세요?"),
        (re.compile(r"([가-힣])십니까\b"), lambda m: m.group(1) + "세요"),
        # "~합니다" 너무 형식적 → "~해요"
        (re.compile(r"합니다\b"), "해요"),
        (re.compile(r"입니다\b"), "이에요"),
        (re.compile(r"있습니다\b"), "있어요"),
        (re.compile(r"없습니다\b"), "없어요"),
        # 사극체 leak → 친근 어조 (긴 패턴 먼저)
        (re.compile(r"모르오\b"), "몰라요"),
        (re.compile(r"모르겠소\b"), "잘 모르겠어요"),
        (re.compile(r"([가-힣])하오\b"), lambda m: m.group(1) + "해요"),
        (re.compile(r"([가-힣])시오\b"), lambda m: m.group(1) + "세요"),
        (re.compile(r"([가-힣])이오\b"), lambda m: m.group(1) + "이에요"),
        (re.compile(r"([가-힣])구려\b"), lambda m: m.group(1) + "네요"),
        (re.compile(r"([가-힣])소\b(?![이가오])"), lambda m: m.group(1) + "어요"),
        # 격식 표현 → 친근
        (re.compile(r"누구께서\b"), "누가"),
        (re.compile(r"누구신지\b"), "어느 분인지"),
        # "그렇소이까" / "그렇소" 같은 사극체 → "그래요"
        (re.compile(r"그렇소이까\b"), "그래요?"),
        (re.compile(r"그렇소\b"), "그래요"),
    ],
    "hermann": [
        # hermann은 반말. 존댓말 leak 시 반말로 강제.
        # 0) 응답 시작 정중 감탄사 → "어"
        (re.compile(r"^(네|예|네네|예예)([\s,]+)"), r"어\2"),
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
        # 1-2) "(어휘)했습니다/했어요" → "(어휘)했어"
        (re.compile(r"([가-힣])했습니다\b"), lambda m: m.group(1) + "했어"),
        (re.compile(r"([가-힣])했어요\b"), lambda m: m.group(1) + "했어"),
        (re.compile(r"([가-힣])했네요\b"), lambda m: m.group(1) + "했네"),
        # "할 겁니다 / 할 거예요" → "할 거야"
        (re.compile(r"할\s+겁니다\b"), "할 거야"),
        (re.compile(r"할\s+거예요\b"), "할 거야"),
        (re.compile(r"할\s+거에요\b"), "할 거야"),
        # "(동사)할게요" → "(동사)할게" (드릴게요는 별도 위에 있음)
        (re.compile(r"([가-힣])할게요\b"), lambda m: m.group(1) + "할게"),
        # "보시고요/봐요" → "봐"
        (re.compile(r"보시고요\b"), "봐"),
        (re.compile(r"보세요\b"), "봐"),
        # "골라보시죠 / ~보시죠" → "골라봐 / ~봐"
        (re.compile(r"([가-힣])보시죠\b"), lambda m: m.group(1) + "봐"),
        # "(어휘)준비돼 있으니/있으니까" → "(어휘)있어"
        (re.compile(r"준비돼\s+있으니까?\b"), "준비돼 있어"),
        # "말씀하십라" / "말씀하세요" — Hermann은 단순 "말해"
        (re.compile(r"말씀하십라\b"), "말해"),
        (re.compile(r"말씀하세요\b"), "말해"),
        (re.compile(r"말씀해\s*주십시오\b"), "말해줘"),
        # "추천드릴게요/추천할게요" → "추천할게"
        (re.compile(r"추천드릴게요\b"), "추천할게"),
        (re.compile(r"추천할게요\b"), "추천할게"),
        # "(어휘)네요" → "(어휘)네"
        (re.compile(r"([가-힣])네요\b"), lambda m: m.group(1) + "네"),
        # 부드러운 반말 어미 "(어휘)구나" → 거친 반말 "(어휘)다" / "(어휘)네"
        (re.compile(r"고맙구나\b"), "고맙다"),
        (re.compile(r"미안하구나\b"), "미안하다"),
        (re.compile(r"좋구나\b"), "좋네"),
        (re.compile(r"그렇구나\b"), "그렇네"),
        (re.compile(r"([가-힣])이구나\b"), lambda m: m.group(1) + "이네"),
        (re.compile(r"([가-힣])는구나\b"), lambda m: m.group(1) + "는다"),
        # 명사형 종결 (사실 보고체) → 반말 동사형
        (re.compile(r"진행\s+중임\b"), "진행 중이야"),
        (re.compile(r"([가-힣])\s+중임\b"), lambda m: m.group(1) + " 중이야"),
        (re.compile(r"([가-힣])많아짐\b"), lambda m: m.group(1) + "많아"),
        (re.compile(r"많아짐\b"), "많아졌어"),
        (re.compile(r"적어짐\b"), "적어졌어"),
        (re.compile(r"늘어남\b"), "늘었어"),
        (re.compile(r"줄어듦\b"), "줄었어"),
        (re.compile(r"([가-힣])됨\b"), lambda m: m.group(1) + "돼"),
        (re.compile(r"([가-힣])함\b(?!수)"), lambda m: m.group(1) + "해"),
        # 명사형 종결 + 마침표 ("진행함." 같은 패턴)
        (re.compile(r"([가-힣])됨\."), lambda m: m.group(1) + "돼."),
        (re.compile(r"([가-힣])함\."), lambda m: m.group(1) + "해."),
        # "(어휘)하길" / "(어휘)길" 어색한 명령형 → 거친 반말
        (re.compile(r"경계하길\b"), "조심해"),
        (re.compile(r"조심하길\b"), "조심해"),
        (re.compile(r"확인하길\b"), "확인해"),
        (re.compile(r"([가-힣])하길\b"), lambda m: m.group(1) + "해"),
        (re.compile(r"([가-힣])시길\b"), lambda m: m.group(1) + "해"),
        # 일반 "(어휘)길\.?" 어미 (마침표 앞)
        (re.compile(r"([가-힣])길(?=[\s.,!?]|$)"), lambda m: m.group(1) + "라"),
        # 사극체 leak (Hermann은 거친 반말)
        (re.compile(r"무슨 일인고\?"), "무슨 일이야?"),
        (re.compile(r"무슨 일인고\b"), "무슨 일이야"),
        (re.compile(r"([가-힣])인고\?"), lambda m: m.group(1) + "이야?"),
        (re.compile(r"([가-힣])인고\b"), lambda m: m.group(1) + "이야"),
        (re.compile(r"([가-힣])이오\b"), lambda m: m.group(1) + "이야"),
        (re.compile(r"([가-힣])하오\b"), lambda m: m.group(1) + "해"),
        (re.compile(r"([가-힣])소이다\b"), lambda m: m.group(1) + "어"),
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


# 자주 leak되는 영어 단어 → 한국어 치환
_ENGLISH_LEAK_MAP = {
    r"\btoday\b": "오늘",
    r"\bnow\b": "지금",
    r"\btomorrow\b": "내일",
    r"\byesterday\b": "어제",
    r"\bok\b": "좋소",
    r"\bokay\b": "좋소",
    r"\byes\b": "예",
    r"\bno\b": "아니오",
    r"\bhello\b": "",
    r"\bhi\b": "",
    r"\bthanks\b": "감사하오",
    r"\bsorry\b": "미안하오",
}
_ENGLISH_LEAK_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _ENGLISH_LEAK_MAP.items()]


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
    # 자주 leak되는 영어 단어 한국어로 치환
    for pattern, replacement in _ENGLISH_LEAK_COMPILED:
        text = pattern.sub(replacement, text)
    # 그 외 남은 영어 단어 (3자 이상 알파벳 연속) 제거
    text = re.sub(r"\s*\b[a-zA-Z]{3,}\b", "", text)
    # 이중 조사/어미 정리
    for pattern, replacement in _PARTICLE_FIX:
        text = pattern.sub(replacement, text)
    # NPC별 어미 후처리
    if npc and npc in _NPC_POSTFIX:
        for pattern, replacement in _NPC_POSTFIX[npc]:
            text = pattern.sub(replacement, text)
    # NPC 이름 뒤 조사 보정
    text = _fix_korean_particles(text)
    # 중복 공백 + 문장부호 앞 공백 정리
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    # 미완성 끝 문장 자르기
    text = _cut_to_last_sentence(text)
    return text.strip()

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import logging as hf_logging

# HF 경고/INFO 억제 — 시연 콘솔에 파이프라인 로그만 보이게
hf_logging.set_verbosity_error()

from ..memory import MemoryEntry, MemoryRetriever, MemorySource, MemoryStore
from ..memory.chat import build_user_prompt
from ..propagation.graph import RelationGraph
from ..propagation.simulator import PropagationSimulator
from . import pipeline_log

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
BASE_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"

DEFAULT_CHARACTERS = ["elias", "hermann", "mathilda", "finn", "bernhardt"]

# NPC별 강한 instruction — EXAONE의 정중한 baseline 깨고 페르소나 강제
# 형식 단순화: 모델이 어미 표시 대괄호를 응답에 포함시키는 부작용 회피.
# 직업 명시: NPC가 다른 NPC 영역 침범하는 것 방지.
# Quest hook: 회상 정보가 흥미로운 사건이면 페르소나에 맞게 흘림 (플레이어가 단서 놓치지 않도록).
NPC_STRICT_RULES = {
    "elias": (
        "**너의 정체: 마법사·학자**. 절대 잡화상/대장장이/술집주인/음유시인이 아니다. "
        "회상에 다른 직업 언급 있어도 너는 마법사임을 잊지 마라.\n"
        "마법·학문·검증 질문만 답하시오.\n"
        "차분한 학자 어조, 약간 회의적. **어미는 오직 ~오 / ~이오 / ~구려 3가지만**. "
        "다른 사극체(~나이다 ~소이다 ~옵소서 ~십시오 ~군요 ~네요) 절대 사용 금지. "
        "현대체(~니다 ~습니다) 절대 사용 금지.\n"
        "한 문장만. 흠 으로 시작 자주.\n"
        "예: 안녕하세요 -> 흠, 무슨 일이오?\n"
        "예: 뭐하세요? -> 흠, 옛 마법서를 읽고 있었구려.\n"
        "예: 마법 어디서 배웠어요 -> 옛 도시에서 50년 익혔구려.\n"
        "예: 광산은? -> 흠... 그 일이 마음에 걸리오."
    ),
    "hermann": (
        "**너의 정체: 대장장이**. 절대 마법사/잡화상/술집주인/음유시인이 아니다. "
        "회상에 다른 직업 언급 있어도 너는 대장장이임을 잊지 마라.\n"
        "검·쇠·도구 거래만. 약초·음식은 다른 NPC로 보내시오.\n"
        "반말 단답. 존댓말 절대 금지. 한 문장. "
        "어. 음. ... 같은 짧은 시작. 어미는 다 해 지.\n"
        "예: 안녕하세요 -> 어. 무슨 일.\n"
        "예: 뭐해? -> 음, 검 단조 중이야.\n"
        "예: 검 추천해줘 -> 음, 강철 단검부터 써봐."
    ),
    "mathilda": (
        "**너의 정체: 술집 주인**. 절대 마법사/대장장이/잡화상/음유시인이 아니다. "
        "회상에 다른 직업 언급 있어도 너는 술집 주인임을 잊지 마라.\n"
        "음식·음료·소문만. 검·약초는 다른 NPC로 보내시오.\n"
        "따뜻하고 수다스러움. 어머나 어머 아유 자주. "
        "어미는 ~어요 ~네요 ~죠. 한 문장.\n"
        "예: 안녕하세요 -> 어머나, 어서 와요!\n"
        "예: 뭐하세요? -> 아유, 손님 맞을 준비 중이에요.\n"
        "예: 무슨 소문 있어요 -> 어머, 광산 얘기 들었어요?"
    ),
    "finn": (
        "**너의 정체: 음유시인**. 절대 마법사/대장장이/잡화상/술집주인이 아니다. "
        "회상에 다른 직업 언급 있어도 너는 음유시인임을 잊지 마라.\n"
        "노래·이야기·전설만. 거래는 다른 NPC로 보내시오.\n"
        "시적이지만 짧게. 오 그대 자주. 어미는 ~이라 ~리라 ~노라 위주. "
        "절대 금지: ~사옵니다 ~이옵니다 너무 과한 사극체.\n"
        "한 문장.\n"
        "예: 안녕하세요 -> 오 그대, 별빛 같은 걸음이라.\n"
        "예: 뭐하세요? -> 새 노래의 가사를 짓고 있노라.\n"
        "예: 광산은? -> 그곳엔 22인의 혼이 잠들었노라."
    ),
    "bernhardt": (
        "**너의 정체: 잡화상**. 절대 마법사/대장장이/술집주인/음유시인이 아니다. "
        "회상에 다른 직업 언급 있어도 너는 잡화상임을 잊지 마라.\n"
        "약초·잡화만. 검·무기는 헤르만으로 보내시오.\n"
        "정중하고 거래 실용적. 어서 흠 같은 시작. "
        "어미는 ~지요 ~이오 ~습니다 짧게. 한 문장.\n"
        "예: 안녕하세요 -> 어서 오시지요, 뭐 찾으십니까?\n"
        "예: 뭐하세요? -> 흠, 약초 재고 정리 중이지요.\n"
        "예: 약초 있어요? -> 흠, 회복약이라면 셋 정도 있지요."
    ),
}

# NPC별 generation 파라미터 차별화
# - hermann/elias: 짧고 무뚝뚝/회의적 → 낮은 temp + 짧은 max_tokens
# - mathilda/finn: 수다스럽고 시적 → 약간 높은 temp + 긴 max_tokens
# - bernhardt: 정중한 거래상 → 중간
# 속도 우선 — 한두 문장이면 충분. 페르소나도 더 자연스러움.
GEN_PARAMS = {
    # max 50으로 약간 여유 — 여러 메모리 통합 응답 위해 + 한두 문장 가능.
    "hermann":   {"temperature": 0.35, "max_new_tokens": 50, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4, "top_k": 30},
    "elias":     {"temperature": 0.35, "max_new_tokens": 55, "repetition_penalty": 1.20, "no_repeat_ngram_size": 4, "top_k": 30},
    "mathilda":  {"temperature": 0.50, "max_new_tokens": 55, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4, "top_k": 30},
    "finn":      {"temperature": 0.45, "max_new_tokens": 55, "repetition_penalty": 1.18, "no_repeat_ngram_size": 3, "top_k": 30},
    "bernhardt": {"temperature": 0.40, "max_new_tokens": 50, "repetition_penalty": 1.18, "no_repeat_ngram_size": 4, "top_k": 30},
}

# NPC별 Quest Pool — 미리 정의된 quest. 조건(trust 등) 충족 시 NPC가 먼저 제안.
# id는 unique. trust_required: 이 trust 이상이어야 quest 제안.
# intro: NPC가 quest를 줄 때 첫 대사 (greeting 대체).
NPC_QUEST_POOL = {
    "elias": [
        {
            "id": "elias_wasp_nest",
            "title": "말벌집 제거",
            "description": "탑 근처에 자리 잡은 말벌집을 없애 주시오.",
            "reward": "마법 강화 물약",
            "trust_required": 30,  # 기본 친밀도로 바로 시작 가능 (입문 퀘스트)
            "intro": "마침 잘 왔소. 요즘 연구를 하는데 창밖에서 말벌들이 윙윙대 도무지 집중이 안 되오.",
        },
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
            "intro": "야, 너한테만 말하는 건데. 부탁이 하나 있다.",
        },
    ],
    "mathilda": [
        {
            "id": "mathilda_rumor_check",
            "title": "이상한 그림자 소문",
            "description": "밤마다 광장에 나타난다는 검은 그림자의 정체를 알아봐주세요.",
            "reward": "특별 양조 술",
            "trust_required": 35,
            "intro": "어머나 모험가 분, 마침 잘 오셨네요. 실은 부탁이 하나 있거든요.",
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
            "reward": "음유시인의 헌정 노래",
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

    def mark_accepted(self, quest_id: str):
        self._state[quest_id] = "accepted"

    def mark_completed(self, quest_id: str):
        self._state[quest_id] = "completed"

    def get_quest(self, npc: str, quest_id: str) -> dict | None:
        """NPC pool에서 quest_id로 정의 찾기."""
        for q in NPC_QUEST_POOL.get(npc, []):
            if q.get("id") == quest_id:
                return q
        return None

    def list_for_npc(self, npc: str, current_trust: int) -> list[dict]:
        """퀘스트 리스트 UI용 — NPC의 모든 quest + 상태 + 시작 가능 여부."""
        out = []
        for q in NPC_QUEST_POOL.get(npc, []):
            st = self.status(q["id"])
            out.append({
                "id": q["id"],
                "title": q.get("title", ""),
                "description": q.get("description", ""),
                "reward": q.get("reward", ""),
                "trust_required": q.get("trust_required", 0),
                "state": st,
                "eligible": st == "available" and current_trust >= q.get("trust_required", 0),
            })
        return out

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
            "오, 낯선 그대여. 별빛이 새 운명을 이끌어왔노라.",
            "그대의 이름 들어본 적 없거늘, 어인 일로 이곳에 왔노라?",
            "처음 보는 영혼이라. 그대의 이야기를 들려주오.",
        ],
        "지인": [
            "오 그대여, 다시 만났노라. 무슨 노래를 들으러 왔노라?",
            "그대 발걸음이 또 별빛 아래 닿았으니, 반갑노라.",
            "오, 익숙한 영혼이여. 어인 이야기를 청하노라?",
        ],
        "친구": [
            "오 친애하는 벗이여, 그대를 위한 노래가 준비되었노라.",
            "그대를 위해 새 시를 지었으니, 들어보지 않겠노라?",
            "오, 영웅이여. 그대의 모험담을 들려주오.",
        ],
        "절친": [
            "오 나의 영원한 벗이여! 그대의 이야기가 전설로 남으리라.",
            "그대 없는 마을은 노래 없는 밤과 같았노라.",
            "어서 오라 진정한 벗이여, 별빛이 그대를 환영하노라.",
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


# 플레이어 이름을 알 때의 greeting — {name} 자리에 이름 삽입.
# 조사 문제 회피를 위해 받침 영향 없는 위치({name} 님, {name} 양반, 쉼표 뒤)에만 사용.
NPC_GREETINGS_NAMED = {
    "elias": {
        "낯선 사람": [
            "흠, {name}. 무슨 용건이오?",
            "흠, 왔는가 {name}. 용건만 말하시오.",
        ],
        "지인": [
            "어서 오시오, {name} 양반. 무슨 일로 오셨소?",
            "흠, {name} 양반 또 오셨구려. 오늘은 무엇이 궁금하시오?",
            "{name} 양반, 어서 오시오. 마침 한가하던 참이오.",
        ],
        "친구": [
            "오, {name} 왔는가. 잘 오셨소.",
            "반갑소, {name} 양반. 무슨 이야기를 나눠볼까.",
            "어서 오시오 {name}, 늘 환영하오.",
        ],
        "절친": [
            "오, {name}! 마침 자네 생각을 하던 참이오.",
            "허허, {name} 왔는가. 무슨 좋은 소식이라도?",
            "내 가장 신뢰하는 벗 {name}, 어서 앉으시오.",
        ],
    },
    "hermann": {
        "낯선 사람": [
            "어. {name}. 무슨 일이야.",
            "음, {name} 맞지. 뭐 필요해?",
        ],
        "지인": [
            "어. {name} 또 왔네. 뭐 필요해?",
            "음, {name}구나. 무슨 일이야.",
            "어. 어서 와, {name}.",
        ],
        "친구": [
            "오, {name} 왔어? 마침 잘 됐다.",
            "어이 {name}. 검 손볼 거 있냐?",
            "{name} 왔구나. 한 잔 할래?",
        ],
        "절친": [
            "야, {name}! 보고 싶었다.",
            "오! {name} 왔구나. 뭐 도와줄까?",
            "{name}, 너 없으니 심심하더라.",
        ],
    },
    "mathilda": {
        "낯선 사람": [
            "어머, {name} 씨... 오늘은 무슨 일이세요?",
            "{name} 씨군요. 어서 오세요.",
        ],
        "지인": [
            "어머, {name} 님 또 오셨네요! 반가워요.",
            "어서 오세요 {name} 님~ 오늘은 뭘 드시러 오셨어요?",
            "아유, {name} 님 잘 오셨어요. 들어와 앉으세요.",
        ],
        "친구": [
            "어머나, 우리 {name} 님! 보고 싶었어요!",
            "오, {name} 님 왔네! 오늘 맛있는 거 준비했어요.",
            "어머 어서 와요 {name} 님, 자리 비워뒀어요.",
        ],
        "절친": [
            "어머! {name} 님! 어서 와요, 빨리!",
            "꺄, {name} 님 보고 싶었어요! 오늘은 우리 수다 떨어요.",
            "아유 {name} 님, 왜 이렇게 오랜만이에요! 자, 앉아요.",
        ],
    },
    "finn": {
        "낯선 사람": [
            "{name}... 그대의 운명이 다시 이곳에 닿았노라.",
            "오, {name}. 어인 일로 왔는가.",
        ],
        "지인": [
            "오 {name}, 다시 만났노라. 무슨 노래를 들으러 왔는가?",
            "{name}, 그대 발걸음이 별빛 아래 닿았으니 반갑노라.",
            "오, {name}. 어인 이야기를 청하는가?",
        ],
        "친구": [
            "오 친애하는 {name}, 그대를 위한 노래가 준비되었노라.",
            "{name}, 그대를 위해 새 시를 지었노라. 들어보겠는가?",
            "오, 영웅 {name}. 그대의 모험담을 들려주오.",
        ],
        "절친": [
            "오 나의 영원한 벗 {name}! 그대의 이야기가 전설로 남으리라.",
            "{name} 없는 마을은 노래 없는 밤과 같았노라.",
            "어서 오라 {name}, 별빛이 그대를 환영하노라.",
        ],
    },
    "bernhardt": {
        "낯선 사람": [
            "{name} 님이시군요. 어서 오시지요.",
            "흠, {name} 님. 무엇을 찾으십니까?",
        ],
        "지인": [
            "어서 오시지요, {name} 님. 오늘은 어떤 게 필요하십니까?",
            "{name} 님 또 오셨군요, 반갑습니다.",
            "흠, {name} 님. 이번엔 무엇을 찾으십니까?",
        ],
        "친구": [
            "오, {name} 님 오셨군요. 특별 할인 가격으로 해드리지요.",
            "어서 오시지요 {name} 님, 좋은 물건 들어왔습니다.",
            "{name} 님, 자주 찾아주시니 감사할 따름이지요.",
        ],
        "절친": [
            "오, 가장 소중한 단골 {name} 님! 어서 오시지요.",
            "{name} 님께는 무엇이든 최고 품질로 드리지요.",
            "친애하는 {name} 님, 오늘은 특별한 거래가 가능하지요.",
        ],
    },
}


# NPC 한글 표기 (인사/언급 template용)
KO_NPC_NAME = {
    "elias": "엘리아스", "hermann": "헤르만", "mathilda": "마틸다",
    "finn": "핀", "bernhardt": "베른하르트",
}

# 전파받은 플레이어 사건을 인사에서 언급하는 template.
# 규칙: 오늘(tick 직후) 받은 소식만 + 1회만 (mentioned 마커).
# {src} = 전해준 NPC, {content} = 플레이어 발화 원문.
# 조사 안전: 에게/한테/님/의 는 받침 무관.
NPC_EVENT_MENTION = {
    "elias": " 그나저나 {src}에게 들었소만, 그대가 '{content}'라 했다지.",
    "hermann": " 아 맞다, {src}한테 들었는데 너 '{content}'라며?",
    "mathilda": " 그나저나 {src} 님한테 들었는데, '{content}'라면서요?",
    "finn": " 바람결에 {src}의 이야기를 들었노라. 그대가 '{content}'라 하였다지.",
    "bernhardt": " 참, {src} 님께 들었습니다만 '{content}'라고요.",
}

# 자기소개(이름) 전파 언급 — 발화 원문 인용 대신 "성함은 들었다" 형태.
# ("'내 이름은 X야'라면서요?" 같은 어색한 직접 인용 회피)
NPC_NAME_HEARD_MENTION = {
    "elias": " 그대 이름은 {src}에게 들었소.",
    "hermann": " 네 이름은 {src}한테 들었다.",
    "mathilda": " 성함은 {src} 님한테 들었어요~",
    "finn": " 그대의 이름은 {src}에게 들어 알고 있노라.",
    "bernhardt": " 성함은 {src} 님께 들었습니다.",
}

# 오늘 한 자율 대화를 인사에서 언급하는 template (1회, 당일만).
# general = 그냥 대화했다 / about_player = 플레이어 얘기가 화제였다.
# {other} = 대화 상대 NPC. 조사 안전: '하고'·'님'은 받침 무관.
NPC_CHAT_MENTION = {
    "elias": {
        "general": " 방금 {other}하고 이야기를 나눴소만.",
        "about_player": " 방금 {other}하고 그대 이야기를 나눴소.",
    },
    "hermann": {
        "general": " 방금 {other}하고 얘기 좀 했다.",
        "about_player": " 아까 {other}하고 네 얘기 했었다.",
    },
    "mathilda": {
        "general": " 방금 {other} 님하고 수다 떨고 있었어요~",
        "about_player": " 방금 {other} 님하고 얘기하다가 그쪽 얘기도 나왔어요~",
    },
    "finn": {
        "general": " 방금 {other}하고 이야기를 나누고 온 참이노라.",
        "about_player": " 방금 {other}하고 그대의 이야기를 나눴노라.",
    },
    "bernhardt": {
        "general": " 방금 {other} 님하고 이야기를 나눴지요.",
        "about_player": " 방금 {other} 님하고 손님 이야기를 했지요.",
    },
}


# NPC별 Quest 본문 template — propose_quest에서 intro(퀘스트별 도입) 뒤에 붙음.
# 주의: 도입 인사/부탁 멘트는 pool의 intro가 담당 — 여기엔 퀘스트 핵심만
# (중복 도입 "부탁이 있소 + 부탁이 있소" 방지).
NPC_QUEST_INTRO = {
    "elias": (
        " 「{title}」 — {description} "
        "성공하면 {reward} 보답하겠소."
    ),
    "hermann": (
        " 「{title}」 — {description} "
        "끝내면 {reward} 챙겨주마."
    ),
    "mathilda": (
        " 「{title}」 — {description} "
        "해주시면 {reward} 드릴게요!"
    ),
    "finn": (
        " 「{title}」 — {description} "
        "보답으로 {reward} 약속하노라."
    ),
    "bernhardt": (
        " 「{title}」 — {description} "
        "성사되면 {reward} 지불해드리지요."
    ),
}

# 퀘스트 대사 마무리 — "해주겠나?" (플레이어 응답 대기 신호)
NPC_QUEST_ASK = {
    "elias": " 맡아주겠소?",
    "hermann": " 해줄래?",
    "mathilda": " 해주실래요?",
    "finn": " 그대, 맡아주겠는가?",
    "bernhardt": " 맡아주시겠습니까?",
}

# 수락 응답 — "고맙다, 역시 너라면 해줄 줄 알았다"
NPC_QUEST_ACCEPT_REPLY = {
    "elias": "고맙소. 역시 그대라면 맡아줄 줄 알았소.",
    "hermann": "고맙다. 역시 너라면 해줄 줄 알았다.",
    "mathilda": "어머, 고마워요! 역시 믿고 있었다니까요~",
    "finn": "오, 고맙노라! 그대의 용기가 곧 노래가 되리라.",
    "bernhardt": "감사합니다. 역시 믿을 만한 분이지요.",
}

# 거절 응답 — "흠 알겠네, 맘 바뀌면 다시 와줘"
NPC_QUEST_DECLINE_REPLY = {
    "elias": "흠, 알겠소. 마음이 바뀌면 다시 와주시오.",
    "hermann": "흠, 알겠다. 맘 바뀌면 다시 와라.",
    "mathilda": "아쉽네요~ 마음 바뀌면 언제든 다시 와요.",
    "finn": "그런가... 운명이 그대를 다시 부르면 오라.",
    "bernhardt": "알겠습니다. 마음 바뀌시면 다시 들러주시지요.",
}

# 애매한 응답 — "아직은 생각이 없으신가 보군" (거절 처리, 퀘스트는 다시 시작 가능)
NPC_QUEST_UNCLEAR_REPLY = {
    "elias": "흠, 아직 생각이 없는 모양이구려. 마음이 정해지면 다시 말해주시오.",
    "hermann": "음, 아직 생각 없나 보네. 정해지면 말해.",
    "mathilda": "아직 고민되시나 봐요~ 천천히 생각해보고 말해줘요.",
    "finn": "아직 마음의 준비가 안 되었나 보군. 때가 오면 말해주오.",
    "bernhardt": "아직 생각이 없으신가 보군요. 정해지면 말씀해주시지요.",
}

# 완료 반응 — "고맙다, 역시 너야" (완료 버튼 → 대화창 표시)
NPC_QUEST_COMPLETE_REPLY = {
    "elias": "오, 벌써 해결했는가. 역시 그대였소. 고맙소.",
    "hermann": "오, 다 끝냈구나. 역시 너야. 고맙다.",
    "mathilda": "어머! 벌써 해결했어요? 정말 대단해요, 고마워요!",
    "finn": "오 영웅이여! 그대의 공적이 노래로 남으리라. 고맙노라.",
    "bernhardt": "훌륭합니다. 약속한 보상을 드리지요. 감사합니다.",
}

# 퀘스트 응답 분류 키워드
_QUEST_POS_SHORT = {"예", "네", "넵", "응", "ㅇㅇ", "ㅇㅋ", "콜", "그래", "좋아", "좋지", "오케이", "ok", "okay", "당연"}
_QUEST_POS_SUB = ["할게", "할께", "해볼게", "해보겠", "해줄게", "알겠", "수락", "맡겨", "해야지", "도와줄", "받을게", "좋아요"]
_QUEST_NEG_SHORT = {"ㄴㄴ", "아니", "아뇨", "아니요", "노", "no", "싫어", "싫다", "패스"}
_QUEST_NEG_SUB = ["안할", "안 할", "못해", "못 해", "못하", "어려울", "어렵", "거절", "나중에", "다음에", "무리", "사양", "싫"]


def classify_quest_reply(text: str) -> str:
    """퀘스트 제안에 대한 플레이어 응답 분류.

    반환: "accepted" | "declined" | "unclear"
    - 부정 우선 검사 ("안할게"의 '할게' 오탐 방지)
    - 짧은 토큰(예/응/ㄴㄴ)은 공백·문장부호 제거 후 짧을 때만 매칭 ("예전에~" 오탐 방지)
    """
    t = re.sub(r"[\s.,!?~^;:]+", "", (text or "").strip().lower())
    if not t:
        return "unclear"
    # 1) 부정 — substring 먼저
    for k in _QUEST_NEG_SUB:
        if k.replace(" ", "") in t:
            return "declined"
    if t in _QUEST_NEG_SHORT or (len(t) <= 5 and any(t.startswith(k) for k in _QUEST_NEG_SHORT)):
        return "declined"
    # 2) 긍정
    for k in _QUEST_POS_SUB:
        if k in t:
            return "accepted"
    if t in _QUEST_POS_SHORT or (len(t) <= 5 and any(t.startswith(k) for k in _QUEST_POS_SHORT)):
        return "accepted"
    return "unclear"

PROMPT_FACT = (
    "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "사람 이름과 장소 이름은 절대 바꾸지 말고, 어조만 너답게 바꾸세요. "
    "다른 설명이나 라벨은 붙이지 마세요.\n\n"
    "사실: {memory}\n\n"
    "당신의 한 마디:"
)

REFLECTION_PROMPT = """다음은 {npc}({role})의 최근 기억들이오.

{memories}

위 기억들에서 가장 중요한 통찰 3가지를 한 문장씩 한국어로 추출하시오.
사람 이름·장소 이름은 절대 바꾸지 말고 그대로 사용.
각 문장은 20-40자 내외, 1인칭으로 짧고 명확하게.

형식:
1. ...
2. ...
3. ...

추출:"""


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
        retrieval_k: int = 2,  # 1→2: 여러 메모리 통합 응답 (시드 + reflection 활용도 ↑)
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
        self.player_name: str | None = None  # plyaer 이름 추적 (자기소개 후 설정)
        # 이름 인지 캐시 — _npc_knows_player_name의 full-scan을 응답마다 반복하지 않도록.
        # prune이 player/personal 메모리를 항상 보존하므로 "한번 알면 계속 앎" (단조 증가).
        # /memory/reset 시 함께 비워야 함.
        self._name_known_cache: set[str] = set()
        # _restore_player_name 정규식 캐시 (이름 바뀔 때만 재컴파일)
        self._name_regex_cache: tuple | None = None

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
            "[참고 기억]은 다른 사람들이 한 말의 기록임. **너 자신의 말이 아니다**. "
            "**절대 회상 속 다른 NPC의 직업·정체성을 너의 것으로 말하지 마라**. "
            "예: bernhardt가 '잡화점 운영'이라 했어도, 너는 잡화상이 아니다. "
            "회상은 '~라더군' '~한테 들었소' 형태로만 인용하시오. "
            "**플레이어 '질문:'에 직접 답하기가 최우선**. "
            "관련 없으면 회상 무시. "
            # 인식론 라벨별 확신도 (Hindsight 계열 — 기억 타입 분리)
            "기억 앞 라벨에 따라 확신도를 조절하시오: "
            "(직접 들음)·(아는 사실)은 확신 있게, "
            "(전해 들은 소문)은 '~라더라/~라 들었소' 전언 투로 불확실하게, "
            "(나의 생각)은 '~인 듯하오' 추정 투로. "
            # 흔한 해석 오류 방지
            "플레이어가 '제 이름' 또는 '내 이름' 묻거나 말하면 그건 **플레이어 본인의 이름**임. "
            "회상에 플레이어 이름이 있으면 그것을 답하고, 없으면 '아직 듣지 못했소' 식으로 답. "
            "절대 자기 이름을 답하지 말 것. "
            if self.use_memory else ""
        )

        strict_rule = NPC_STRICT_RULES.get(npc, "")
        trust_hint = self.trust.disclosure_hint(npc)
        # 이 NPC가 실제로 이름을 알 때만 system prompt에 주입
        # (전파 안 받은 NPC가 이름을 쓰면 시뮬레이션 모순)
        player_name_hint = (
            f"**플레이어(지금 대화 중인 모험가)의 이름은 정확히 '{self.player_name}'**. "
            f"이게 확정된 이름. 절대 다른 변형(반응현/반욱헌 등) 사용 X. "
            f"플레이어가 '아니고' '아니야' 같은 부정문으로 정정해도, "
            f"플레이어가 직접 알려준 '{self.player_name}'가 정답. "
            f"플레이어를 '{self.player_name}'으로만 부르고, 다른 사람 이름과 헷갈리지 X. "
            if self._npc_knows_player_name(npc) else ""
        )

        return (
            f"당신은 {npc}입니다. {desc}\n"
            f"어조: {tone}. 피해야 할 것: {avoid}.\n"
            f"자주 쓰는 어휘: {vocab}.\n"
            f"{strict_rule}\n"
            f"다른 마을 사람: {others}. 이들의 이름과 직업을 절대 바꾸지 마시오 "
            "(예: mathilda를 마트닐라로 변형 금지).\n"
            f"{player_name_hint}"
            f"{trust_hint}\n"
            f"{memory_hint}"
            "**반드시 한 문장**으로만 답하시오. 절대 길게 X. "
            "**모르는 정보는 짧게 '모르오/못 들었소' 식으로 답** — 절대 변명하거나 길게 늘리지 X. "
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

        # ① 발화 — 플레이어가 NPC에게 말함
        pipeline_log.log("utter", f'{npc} ← 플레이어: "{user_text[:60]}"',
                         npc=npc, text=user_text[:120])

        t0 = time.time()
        if self.use_memory:
            # 정체성/이름 query 감지 — semantic 매칭이 약해도 personal 메모리 우선 회상.
            identity_kw = ["이름", "누구", "누군지", "누구신지", "기억하", "기억나"]
            is_identity_query = any(kw in user_text for kw in identity_kw)

            retrieved = []
            if is_identity_query:
                # 플레이어 자기소개 메모리 2개 우선 추가 (최신 우선)
                personal = self.stores[npc].find_player_personal(limit=2)
                retrieved.extend(personal)

            # 일반 semantic retrieval (정체성 query면 보조, 아니면 메인)
            # game_day 전달 → 일화 기억은 망각곡선(retention) 기반 최신성
            semantic = self.retrievers[npc].search(
                user_text, k=self.retrieval_k, game_day=self.day
            )
            # 중복 제거 + 결합
            seen_ids = {r["id"] for r in retrieved}
            for r in semantic:
                if r["id"] not in seen_ids:
                    retrieved.append(r)
                    seen_ids.add(r["id"])

            # 키워드 기반 추가 검색 — query에 명사 키워드 있으면 그 단어 포함 메모리도 회상
            event_kw = ["곰", "용", "광산", "광장", "마법사", "약초", "검",
                        "위험", "사건", "사고", "곰의", "괴물"]
            query_kw = [kw for kw in event_kw if kw in user_text]
            if query_kw:
                all_data = self.stores[npc].all()
                ids, docs, metas = (all_data.get("ids", []),
                                    all_data.get("documents", []),
                                    all_data.get("metadatas", []))
                for i, mid in enumerate(ids):
                    if mid in seen_ids:
                        continue
                    doc = docs[i] if i < len(docs) else ""
                    meta = metas[i] if i < len(metas) else {}
                    if int(meta.get("importance", 5)) < 5:
                        continue
                    # 키워드 매칭
                    if any(kw in doc for kw in query_kw):
                        retrieved.append({
                            "id": mid, "text": doc,
                            "importance": int(meta.get("importance", 5)),
                            "metadata": meta,
                            "similarity": 0.7,  # 키워드 매칭은 강한 신호
                            "score": 0.7,
                        })
                        seen_ids.add(mid)
                        if len(retrieved) >= 3:
                            break

            # 회상 강화 (SAGE/RMM 계열) — 회상된 기억은 last_access 갱신 + recall_count 증가.
            # 자주 회상되는 기억일수록 반감기가 늘어나 오래 남음 (spaced repetition).
            if retrieved:
                try:
                    self.stores[npc].collection.update(
                        ids=[m["id"] for m in retrieved],
                        metadatas=[{
                            "last_access_day": self.day,
                            "recall_count": int(
                                m.get("metadata", {}).get("recall_count", 0) or 0
                            ) + 1,
                        } for m in retrieved],
                    )
                except Exception:
                    pass  # 강화 실패는 치명적이지 않음 (다음 회상에서 재시도됨)

            augmented = build_user_prompt(retrieved, user_text)

            # 회상 결과 로그 — 출처(누구에게 전파받았나) + 내용까지 상세 표시
            if retrieved:
                # source 한글 라벨
                src_label = {
                    "seed": "배경지식", "dialogue": "플레이어 발화",
                    "propagation": "전파", "conversation": "NPC대화",
                    "reflection": "통찰", "observation": "관찰",
                }
                recall_summary = []
                has_from_other = False
                for m in retrieved:
                    meta = m.get("metadata", {})
                    src = meta.get("source", "?")
                    label = src_label.get(src, src)
                    frm = meta.get("from")  # 전파/대화 출처 NPC
                    other = meta.get("other_npc")  # NPC-NPC 대화 상대
                    txt = m["text"][:55]
                    if frm:  # propagation — 누구에게 들었나
                        recall_summary.append(f"[{label}|{frm}한테] {txt}")
                        has_from_other = True
                    elif other:  # conversation
                        recall_summary.append(f"[{label}|{other}와] {txt}")
                        has_from_other = True
                    else:
                        recall_summary.append(f"[{label}] {txt}")

                pipeline_log.log(
                    "recall",
                    f"{npc} 회상 {len(retrieved)}개"
                    + ("  ← 다른 NPC에게서 전파받은 정보 활용" if has_from_other else ""),
                    npc=npc, memories=recall_summary,
                )
                # 각 회상 메모리를 별도 줄로 상세 표시 (출처 + 내용)
                for s in recall_summary:
                    pipeline_log.log("recall", f"   └ {s}", npc=npc, detail=True)
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
                top_k=gp.get("top_k", 30),
                repetition_penalty=gp["repetition_penalty"],
                no_repeat_ngram_size=gp["no_repeat_ngram_size"],
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        text = _clean_response(text, npc=npc)  # emoji/특수문자/NPC별 어미 정리
        # 플레이어 이름 변형 자동 복원 (예: "반욱헌" → "반욱현")
        text = self._restore_player_name(text)
        latency_ms = int((time.time() - t0) * 1000)

        # ④ 언급 — NPC 최종 응답
        pipeline_log.log("recall", f'{npc} → 플레이어: "{text[:60]}" ({latency_ms}ms)',
                         npc=npc, response=text[:120], latency_ms=latency_ms)

        # Quest는 응답에서 생성하지 않음 — 플레이어 주도 propose_quest 흐름이 정식 경로.
        # (LLM 자동 quest 생성은 맥락 없는 제안 문제로 제거됨, 2026-06)
        quest = None

        # 플레이어 발화를 NPC의 DIALOGUE 메모리로 저장 (다음 tick에서 전파 후보)
        if self.use_memory:
            self._save_player_turn(npc, user_text)

        # 신뢰도 업데이트 (응답 후, 다음 대화부터 영향)
        trust_before = self.trust.get(npc)
        label_before = self.trust.label(npc)
        trust_delta = self.trust.on_player_turn(npc, user_text)
        trust_after = self.trust.get(npc)
        label_after = self.trust.label(npc)
        # ♥ 친밀도 로그 (변화 있을 때만)
        if trust_delta != 0:
            sign = f"+{trust_delta}" if trust_delta > 0 else f"{trust_delta}"
            promote = "  ⭐ 등급 변화!" if label_before != label_after else ""
            pipeline_log.log(
                "trust",
                f"{npc} {trust_before} → {trust_after} ({sign}) [{label_after}]{promote}",
                npc=npc, before=trust_before, after=trust_after, delta=trust_delta,
                label=label_after,
            )

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
        """Quest 완수 — 신뢰도 +10 + 완료 경험담 기억 저장 (전파 → 평판 확산).

        반환에 reaction(NPC 완료 반응 대사) 포함 — Unity가 대화창에 표시.
        """
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")
        label_before = self.trust.label(npc)
        before = self.trust.get(npc)
        delta = self.trust.on_quest_complete(npc)
        if quest_id:
            self.quests.mark_completed(quest_id)

        # 완료 경험담을 NPC 기억에 — importance 9 → 다른 NPC로 전파 (평판)
        q = self.quests.get_quest(npc, quest_id) if quest_id else None
        title = q.get("title", "") if q else (quest_id or "부탁")
        try:
            entry = MemoryEntry(
                id=f"qdone_{uuid.uuid4().hex[:8]}",
                text=f"플레이어가 '{title}' 부탁을 해결해줬다. 정말 듬직한 모험가다",
                importance=9,
                timestamp=datetime.now(timezone.utc),
                source=MemorySource.OBSERVATION,
                metadata={"day": self.day, "quest_id": quest_id or ""},
            )
            self.stores[npc].add(entry)
        except Exception as e:
            print(f"[quest] 완료 경험담 저장 실패: {e}")

        # ♥ 친밀도 로그 — quest 완료 보상
        promote = "  ⭐ 등급 변화!" if label_before != self.trust.label(npc) else ""
        pipeline_log.log(
            "trust",
            f"{npc} {before} → {self.trust.get(npc)} (+{delta}) quest 완료!{promote}",
            npc=npc, delta=delta, reason="quest",
        )
        pipeline_log.log(
            "quest",
            f"{npc} 퀘스트 완료: 「{title}」 → 경험담 기억 저장 (전파 후보)",
            npc=npc, quest_id=quest_id or "",
        )

        return {
            "npc": npc,
            "quest_id": quest_id,
            "title": title,
            "reaction": NPC_QUEST_COMPLETE_REPLY.get(npc, "고맙네, 잘 해결해줬군."),
            "trust": self.trust.get(npc),
            "trust_label": self.trust.label(npc),
            "trust_delta": delta,
        }

    # ---------- 퀘스트 제안/수락 흐름 (플레이어 주도) ----------
    def propose_quest(self, npc: str, quest_id: str) -> dict:
        """플레이어가 퀘스트 리스트에서 시작 — NPC 퀘스트 대사 생성 (template, LLM X).

        대사 = pool intro + 퀘스트 안내 + "해주겠나?" 마무리.
        반환: {"text": 대사, "quest": {...}} 또는 {"error": ...}
        """
        if npc not in self.characters:
            return {"error": f"알 수 없는 NPC: {npc}"}
        q = self.quests.get_quest(npc, quest_id)
        if q is None:
            return {"error": f"알 수 없는 퀘스트: {quest_id}"}
        st = self.quests.status(quest_id)
        if st == "completed":
            return {"error": "이미 완료한 퀘스트입니다"}
        if st == "accepted":
            return {"error": "이미 진행 중인 퀘스트입니다"}
        if self.trust.get(npc) < q.get("trust_required", 0):
            return {"error": f"친밀도가 부족합니다 (필요: {q.get('trust_required', 0)})"}

        intro = q.get("intro", "")
        # 이름을 아는 NPC면 '모험가' 호칭을 이름으로
        if self._npc_knows_player_name(npc):
            intro = (intro
                     .replace("모험가 양반", f"{self.player_name} 양반")
                     .replace("모험가 분", f"{self.player_name} 님")
                     .replace("모험가", self.player_name))
        body = NPC_QUEST_INTRO.get(
            npc, " 「{title}」 — {description} 보상: {reward}."
        ).format(
            title=q.get("title", ""),
            description=q.get("description", ""),
            reward=q.get("reward", "") or "응당한",
        )
        ask = NPC_QUEST_ASK.get(npc, " 맡아주겠는가?")
        text = (intro + body + ask).strip()

        pipeline_log.log(
            "quest", f"{npc} 퀘스트 제안: 「{q.get('title', '')}」 — 응답 대기",
            npc=npc, quest_id=quest_id,
        )
        return {
            "text": text,
            "quest": {
                "id": quest_id,
                "title": q.get("title", ""),
                "description": q.get("description", ""),
                "reward": q.get("reward", ""),
                "giver": npc,
            },
        }

    def handle_quest_reply(self, npc: str, quest_id: str, user_text: str) -> dict:
        """제안 후 플레이어 응답 분류 → 수락/거절/애매 template 응답.

        - accepted: tracker 상태 변경 + NPC 기억 저장 (전파 → 평판)
        - declined/unclear: 상태 그대로 (available) — 나중에 다시 시작 가능
        """
        stage = classify_quest_reply(user_text)
        q = self.quests.get_quest(npc, quest_id)
        title = q.get("title", "") if q else quest_id

        if stage == "accepted":
            self.accept_quest(npc, quest_id)
            text = NPC_QUEST_ACCEPT_REPLY.get(npc, "고맙네. 부탁하지.")
        elif stage == "declined":
            text = NPC_QUEST_DECLINE_REPLY.get(npc, "알겠네. 마음 바뀌면 다시 오게.")
            pipeline_log.log(
                "quest", f"{npc} 퀘스트 거절됨: 「{title}」 (다시 시작 가능)",
                npc=npc, quest_id=quest_id,
            )
        else:  # unclear — 거절 뉘앙스, 퀘스트는 그대로
            text = NPC_QUEST_UNCLEAR_REPLY.get(npc, "아직 생각이 없는 모양이군.")
            pipeline_log.log(
                "quest", f"{npc} 퀘스트 응답 애매 → 보류: 「{title}」",
                npc=npc, quest_id=quest_id,
            )
        return {"stage": stage, "text": text}

    def accept_quest(self, npc: str, quest_id: str) -> None:
        """수락 확정 — 상태 변경 + '플레이어가 수락했다' 기억 저장 (전파 후보)."""
        self.quests.mark_accepted(quest_id)
        q = self.quests.get_quest(npc, quest_id)
        title = q.get("title", "") if q else quest_id
        # 수락 사실을 NPC 기억에 — importance 8이라 다른 NPC로 전파됨 (평판)
        try:
            entry = MemoryEntry(
                id=f"qacc_{uuid.uuid4().hex[:8]}",
                text=f"플레이어가 '{title}' 부탁을 수락했다",
                importance=8,
                timestamp=datetime.now(timezone.utc),
                source=MemorySource.OBSERVATION,
                metadata={"day": self.day, "quest_id": quest_id},
            )
            self.stores[npc].add(entry)
        except Exception as e:
            print(f"[quest] 수락 기억 저장 실패: {e}")
        pipeline_log.log(
            "quest",
            f"{npc} 퀘스트 수락됨: 「{title}」 → 기억 저장 (전파 후보, 평판)",
            npc=npc, quest_id=quest_id,
        )

    def _npc_knows_player_name(self, npc: str) -> bool:
        """이 NPC가 플레이어 이름을 아는가.

        전역 player_name만으로 판단하면 전파 시뮬레이션과 모순
        (말 안 한 NPC도 호명하는 치트). 그 NPC의 ChromaDB에
        이름 메모리가 실제로 있어야 함:
        - 직접 들음 (player=True + has_personal)
        - 전파로 받음 (player_origin=True + has_personal)
        → 호명 자체가 '전파가 도달했다'는 가시적 증거가 됨.
        """
        if not self.player_name or npc not in self.characters:
            return False
        # 캐시 hit — full-scan 생략 (응답마다 호출되는 hot path)
        if npc in self._name_known_cache:
            return True
        try:
            known = bool(self.stores[npc].find_player_personal(limit=1))
        except Exception:
            return False
        if known:
            self._name_known_cache.add(npc)
        return known

    def _consume_fresh_mention(self, npc: str) -> dict | None:
        """오늘(방금 tick) 전파받은 플레이어 사건 1개를 인사 언급용으로 소비.

        조건:
        - source=propagation + player_origin (플레이어 관련 전파 소식만)
        - meta.day == self.day  → 방금 tick에 받은 신선한 소식만 (하루 지나면 X)
        - mentioned 마커 없음   → 반환 즉시 마커 찍어 1회만 언급
        - 발화 인용부('...'라고 했어) 추출 가능한 것만

        여러 개면 importance 최고 1개. 없으면 None.
        """
        if npc not in self.characters or self.day <= 0:
            return None
        try:
            data = self.stores[npc].all()
        except Exception:
            return None
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        best = None
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if meta.get("source") != "propagation":
                continue
            if not meta.get("player_origin"):
                continue
            if meta.get("mentioned"):
                continue
            if int(meta.get("day", -1)) != self.day:
                continue  # 오늘 소식만 — 묵은 소문은 언급 안 함
            doc = docs[i] if i < len(docs) else ""
            m = re.search(r"'([^']{2,60})'라고 했어", doc)
            if not m:
                continue
            imp = int(meta.get("importance", 5))
            if best is None or imp > best["importance"]:
                best = {
                    "id": mid,
                    "content": m.group(1),
                    "src": meta.get("from", ""),
                    "importance": imp,
                    "has_personal": bool(meta.get("has_personal")),
                }

        if best is None:
            return None
        # 1회 언급 마커 — 재방문 시 반복 방지
        try:
            self.stores[npc].collection.update(
                ids=[best["id"]], metadatas=[{"mentioned": True}]
            )
        except Exception:
            pass
        best["src_ko"] = KO_NPC_NAME.get(best["src"], best["src"])
        return best

    def _consume_fresh_chat_mention(self, npc: str) -> dict | None:
        """오늘(방금 tick) 한 자율 대화 1개를 인사 언급용으로 소비 (1회).

        조건: source=conversation + meta.day == self.day + mentioned 마커 없음.
        화제에 플레이어가 등장하면 about_player=True (전용 template 사용).
        """
        if npc not in self.characters or self.day <= 0:
            return None
        try:
            data = self.stores[npc].all()
        except Exception:
            return None
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if meta.get("source") != "conversation":
                continue
            if meta.get("mentioned"):
                continue
            if int(meta.get("day", -1)) != self.day:
                continue  # 오늘 대화만
            other = meta.get("other_npc", "")
            if not other:
                continue
            topic = str(meta.get("topic", ""))
            doc = docs[i] if i < len(docs) else ""
            about_player = (
                "플레이어" in topic
                or bool(self.player_name and self.player_name in (topic + doc))
            )
            # 1회 언급 마커
            try:
                self.stores[npc].collection.update(
                    ids=[mid], metadatas=[{"mentioned": True}]
                )
            except Exception:
                pass
            return {
                "other": other,
                "other_ko": KO_NPC_NAME.get(other, other),
                "about_player": about_player,
            }
        return None

    def get_greeting(self, npc: str) -> str:
        """현재 신뢰도에 맞는 NPC greeting 무작위 1개 반환.

        LLM 호출 없이 즉시 — F키로 처음 대화 시작 시 NPC가 먼저 한마디.
        '이 NPC가' 플레이어 이름을 알 때만(직접 듣거나 전파받음) 이름으로 호명.
        """
        if npc not in self.characters:
            return ""
        label = self.trust.label(npc)
        # 이 NPC가 이름을 알 때만 호명 greeting ("어서 오시오, 반욱현 양반")
        if self._npc_knows_player_name(npc):
            named = NPC_GREETINGS_NAMED.get(npc, {}).get(label, [])
            if named:
                return random.choice(named).format(name=self.player_name)
        greetings = NPC_GREETINGS.get(npc, {}).get(label, [])
        if not greetings:
            return ""
        return random.choice(greetings)

    def get_dialogue_opener(self, npc: str) -> dict:
        """대화 시작 시 NPC가 먼저 말하는 첫 대사 (greeting + 당일 소식 언급).

        Quest는 여기서 자동 제안하지 않음 — 플레이어가 퀘스트 리스트에서
        직접 시작 (propose_quest 흐름)이 정식 경로.
        반환: {text: str, quest: None}
        """
        if npc not in self.characters:
            return {"text": "", "quest": None}

        # greeting + 오늘 소식 1개만 언급
        # (우선순위: 전파받은 플레이어 소식 > 자율 대화. 둘 다 당일 + 1회 소비)
        text = self.get_greeting(npc)
        if text:
            mention = self._consume_fresh_mention(npc)
            if mention:
                # 자기소개 전파는 "성함은 들었다" 전용 템플릿 (원문 인용 어색 회피)
                tmpl = (NPC_NAME_HEARD_MENTION if mention.get("has_personal")
                        else NPC_EVENT_MENTION).get(npc)
                if tmpl:
                    text = text + tmpl.format(
                        src=mention["src_ko"], content=mention["content"]
                    )
                    pipeline_log.log(
                        "recall",
                        f"{npc} 인사에서 전파 소식 언급 ({mention['src']}한테 들음): "
                        f"'{mention['content'][:30]}'",
                        npc=npc, src=mention["src"],
                    )
            else:
                chat = self._consume_fresh_chat_mention(npc)
                if chat:
                    tmpls = NPC_CHAT_MENTION.get(npc)
                    if tmpls:
                        key = "about_player" if chat["about_player"] else "general"
                        text = text + tmpls[key].format(other=chat["other_ko"])
                        pipeline_log.log(
                            "recall",
                            f"{npc} 인사에서 자율 대화 언급 (상대: {chat['other']}, "
                            f"플레이어 화제: {chat['about_player']})",
                            npc=npc, other=chat["other"],
                        )
        return {"text": text, "quest": None}

    # ---------- Reflection (Park et al. 2023 스타일 추상화) ----------
    def reflect(self, npc: str, recent_n: int = 25, min_importance_sum: int = 80) -> dict:
        """NPC의 최근 메모리에서 LLM으로 추상화된 통찰 3개 추출.

        Park et al. Generative Agents의 Reflection 컴포넌트.
        - 최근 N개 메모리의 importance 합계가 임계값 미만이면 skip
        - LLM에게 "핵심 통찰 3가지" 추출 요청
        - 추출된 통찰을 REFLECTION source, importance 9로 저장

        반환: {"npc": npc, "reflections": [list of str], "skipped": bool}
        """
        if npc not in self.characters:
            return {"npc": npc, "reflections": [], "skipped": True, "reason": "unknown npc"}

        # 최근 메모리 가져오기 (ChromaDB의 all() 사용)
        all_data = self.stores[npc].all()
        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        entries = []
        for i, mid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            # reflection 자체는 다시 reflect 대상에서 제외
            if meta.get("source") == "reflection":
                continue
            entries.append({
                "id": mid,
                "text": docs[i] if i < len(docs) else "",
                "importance": int(meta.get("importance", 5)),
                "timestamp": meta.get("timestamp", ""),
            })

        # 최신 N개 (timestamp 내림차순)
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        recent = entries[:recent_n]

        if not recent:
            return {"npc": npc, "reflections": [], "skipped": True, "reason": "no memories"}

        # importance 합 임계값 검사
        imp_sum = sum(e["importance"] for e in recent)
        if imp_sum < min_importance_sum:
            return {"npc": npc, "reflections": [], "skipped": True, "reason": f"imp_sum {imp_sum} < {min_importance_sum}"}

        # 메모리 텍스트 정리 (각 줄 한 메모리)
        mem_lines = []
        for i, e in enumerate(recent[:20], 1):
            mem_lines.append(f"{i}. {e['text'][:120]}")
        memories_text = "\n".join(mem_lines)

        role_brief = (
            self.personas.get(npc, {}).get("description", "").split(".")[0].strip()
        )
        prompt = REFLECTION_PROMPT.format(
            npc=npc, role=role_brief, memories=memories_text,
        )

        # LLM 호출
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=150,
                do_sample=False,  # 통찰 추출은 deterministic
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()

        # 응답 파싱: "1. ...", "2. ...", "3. ..." 패턴
        reflections = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^[1-9][\.\)]\s*(.+)$", line)
            if m:
                insight = m.group(1).strip()
                if 8 <= len(insight) <= 100:  # 길이 제한 (너무 짧/길면 잘못 추출)
                    reflections.append(insight)
            if len(reflections) >= 3:
                break

        # 메모리로 저장 + 통찰 로그
        cleaned_insights = []
        for i, insight in enumerate(reflections):
            insight_clean = _clean_response(insight, npc=npc)
            cleaned_insights.append(insight_clean)
            entry = MemoryEntry(
                id=f"refl_d{self.day}_{uuid.uuid4().hex[:8]}",
                text=insight_clean,
                importance=9,  # 추상 통찰은 importance 9
                timestamp=datetime.now(timezone.utc),
                source=MemorySource.REFLECTION,
                metadata={"day": self.day, "source_count": len(recent)},
            )
            self.stores[npc].add(entry)

        # ✦ 통찰 로그 — 최근 메모리에서 추출한 추상 통찰
        if cleaned_insights:
            pipeline_log.log(
                "reflect",
                f"{npc} — 최근 {len(recent)}개 메모리(imp합 {imp_sum}) → 통찰 {len(cleaned_insights)}개",
                npc=npc, imp_sum=imp_sum, insights=cleaned_insights,
            )
            for ins in cleaned_insights:
                pipeline_log.log("reflect", f"   └ {ins}", npc=npc, detail=True)

        return {
            "npc": npc,
            "reflections": reflections,
            "skipped": False,
            "imp_sum": imp_sum,
            "source_memories": len(recent),
        }

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
                top_k=gp.get("top_k", 30),
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
                # importance 6+ memory 위주 검색 (game_day → 신선한 기억이 화제로)
                cand = self.retrievers[npc_a].search(
                    "마을 사건 소식", k=3, game_day=self.day
                )
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

        # 화제 전처리 — 전파 메모리 prefix("X한테 들었다:")를 자연어로 변환.
        # 원문 그대로 넣으면 주어가 모호해져 화제 속 인물과 대화 상대를 혼동함.
        topic_clean = topic
        m_src = re.match(r"^(\w+)한테 들었다:\s*(.+)$", topic, re.DOTALL)
        if m_src:
            src_ko = KO_NPC_NAME.get(m_src.group(1), m_src.group(1))
            topic_clean = f"({src_ko}에게 들은 소문) {m_src.group(2)}"

        # 시작 turn: npc_a가 화제 던지기
        opener_prompt = (
            f"당신은 지금 마을에서 {ko_b}을(를) 만났습니다. "
            f"다음 화제로 자연스럽게 말을 거시오 (인사 + 화제 제기, 한두 문장).\n"
            f"규칙: 눈앞의 상대는 {ko_b}이고, 화제 속 인물과 혼동하지 마시오. "
            f"자기 자신({ko_a})을 3인칭으로 말하지 마시오.\n"
            f"화제: {topic_clean}"
        )
        a_messages = [
            {"role": "system", "content": self._build_system_prompt(npc_a)},
            {"role": "user", "content": opener_prompt},
        ]
        first_text = self._generate_for_npc(npc_a, a_messages, max_new_tokens=45)
        turns.append({"speaker": npc_a, "speaker_ko": ko_a, "text": first_text})

        # 각 NPC 시점의 대화 history (chat format)
        a_history = [
            {"role": "system", "content": self._build_system_prompt(npc_a)},
            {"role": "user", "content": opener_prompt},
            {"role": "assistant", "content": first_text},
        ]
        b_history = [
            {"role": "system", "content": self._build_system_prompt(npc_b)},
            {"role": "user", "content":
                f"{ko_a}가 당신에게 말했다: \"{first_text}\"\n"
                f"(한두 문장으로 답하시오. 자기 자신({ko_b})을 3인칭으로 말하지 마시오.)"},
        ]

        last_speaker = npc_a
        # 남은 발화 횟수 = num_turns*2 - 1 (이미 1번 발화함)
        for _ in range(num_turns * 2 - 1):
            responder = npc_b if last_speaker == npc_a else npc_a
            other = npc_a if responder == npc_b else npc_b
            ko_responder = ko_b if responder == npc_b else ko_a
            ko_other = ko_a if responder == npc_b else ko_b

            hist = b_history if responder == npc_b else a_history
            response = self._generate_for_npc(responder, hist, max_new_tokens=45)
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

        # ○ 자율대화 로그 — 두 NPC가 나눈 대화 + 양쪽 메모리 저장
        pipeline_log.log(
            "chat",
            f"Day{self.day}: {npc_a} ↔ {npc_b}  (주제: {topic[:40]})",
            npc_a=npc_a, npc_b=npc_b, day=self.day,
        )
        for t in turns:
            ko = t.get("speaker_ko", t["speaker"])
            pipeline_log.log("chat", f"   └ {ko}: {t['text'][:55]}", detail=True)
        pipeline_log.log("chat", f"   → {npc_a}, {npc_b} 양쪽 메모리에 저장", detail=True)

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

    def _save_player_turn(self, npc: str, user_text: str) -> None:
        text = user_text.strip()
        if len(text) < 4:
            return  # 너무 짧은 감탄사만 제외

        # 질문문 vs 평서문 분기
        # 핵심: "잡았어요"(평서) vs "잡았어요?"(질문) 구분.
        # "?" 가 있거나, 명백한 의문 어미로 끝날 때만 질문.
        # "어요/지요"는 평서문에도 흔해서 제외 (잡았어요·봤어요는 사실 보고).
        q_endings = ["나요", "까요", "까", "가요", "는가요", "니", "냐", "을까", "ㄹ까", "신지", "는지"]
        is_question = "?" in text or any(text.rstrip().endswith(suf) for suf in q_endings)
        # 사실 보고 키워드 (강한 fact 신호) — 다양한 활용형 포함
        fact_kw = ["나타났", "사라졌", "잡았", "봤", "들었", "있었", "갔다", "왔다", "했다",
                   "됐다", "당했", "보았", "만났", "들었어", "가봤", "도와", "받았",
                   "갔는", "있어요", "있더", "봤어", "갔었", "왔어", "혔어",  # 활용형
                   "곰", "용", "괴물", "광산", "광장", "사고", "사건",  # 사물/장소 키워드
                   "위험", "이상한", "수상한"]
        has_fact = any(kw in text for kw in fact_kw)
        # 자기소개/personal info — 자기소개 평서문 한정 (질문 제외).
        # "제 이름은 반욱현이에요" ✅, "제 이름 기억나시나요?" ❌ (질문)
        personal_kw = [
            "내 이름은", "제 이름은",        # 자기소개 평서문 (조사 '은' 포함)
            "이라고 해", "이라고 한다",       # 자기 호칭
            "라고 합니다", "라고 부른다", "라고 불러",
        ]
        # 추가: "나는 X이야" / "저는 X이에요" 패턴 (X는 단어)
        identity_patterns = [
            re.compile(r"^(나는|저는|내가|제가)\s+\S+(이야|이에요|입니다|예요|야)"),
        ]
        has_personal = (
            any(kw in text for kw in personal_kw)
            or any(p.search(text) for p in identity_patterns)
        )
        # **질문이면 personal로 분류 X** — "제 이름?" 같은 질문 잘못 저장 방지
        if is_question:
            has_personal = False

        # importance 매핑 — 평서문은 모두 전파 후보(threshold 7+) 보장.
        if has_personal:
            importance = 10  # 자기소개 평서문: 최우선 회상
        elif is_question:
            importance = 4   # 질문은 전파 X + personal X
        elif has_fact:
            importance = 9
        elif len(text) >= 15:
            importance = 8
        else:
            importance = 7

        entry = MemoryEntry(
            id=f"dlg_{uuid.uuid4().hex[:8]}",
            text=f"플레이어가 말했다: {text}",
            importance=importance,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.DIALOGUE,
            metadata={
                "player": True, "is_question": is_question,
                "has_fact": has_fact, "has_personal": has_personal,
                "day": self.day,  # 망각곡선 계산용 (game day 기준)
            },
        )
        self.stores[npc].add(entry)

        # ② 저장 — ChromaDB에 적재 (전파 후보면 강조)
        tags = []
        if has_personal: tags.append("자기소개")
        if has_fact: tags.append("사실보고")
        if is_question: tags.append("질문")
        tag_str = "/".join(tags) if tags else "일반"
        spread_note = " → 전파 후보" if importance >= 7 else " (전파 X)"
        pipeline_log.log(
            "store",
            f"{npc}.db ← imp:{importance} [{tag_str}]{spread_note}",
            npc=npc, importance=importance, source="dialogue",
            has_personal=has_personal, has_fact=has_fact, is_question=is_question,
        )

        # 플레이어 이름 자동 추출 → 응답 시 LLM에게 정확히 전달 + 후처리 자동 복원
        if has_personal and not is_question:
            extracted = self._extract_player_name(text)
            if extracted:
                self.player_name = extracted

    def _restore_player_name(self, text: str) -> str:
        """응답에서 플레이어 이름 변형(반욱헌, 반응현 등)을 정확한 이름으로 복원."""
        if not self.player_name or len(self.player_name) < 2:
            return text
        name = self.player_name
        n = len(name)
        import re as _re
        # 정규식 캐시 — 이름이 바뀔 때만 재컴파일 (매 응답 호출되는 경로)
        if self._name_regex_cache and self._name_regex_cache[0] == name:
            _, pattern, spaced_pattern = self._name_regex_cache
        else:
            # 한글 경계 (\b는 한글에 안 통함) — lookahead/lookbehind 사용
            pattern = _re.compile(rf"(?<![가-힣])([가-힣]{{{n}}})(?![가-힣])")
            spaced_inner = r"\s*".join(name)
            spaced_pattern = _re.compile(rf"(?<![가-힣])({spaced_inner})(?![가-힣])")
            self._name_regex_cache = (name, pattern, spaced_pattern)
        def _replace(m):
            cand = m.group(1)
            if cand == name:
                return cand
            # 1글자 차이 → 일반 명사가 아니면 정정 (이름 변형 추정)
            diff = sum(1 for a, b in zip(cand, name) if a != b)
            if diff == 1:
                # 일반 명사 후보 — 변형 같지만 다른 단어일 가능성. 보수적으로 정정.
                # 같은 위치 글자가 n-1개 일치하면 거의 이름 변형
                return name
            # 띄어쓰기로 분리된 변형 ("반응 현" 등)은 다른 곳에서 처리
            return cand
        out = pattern.sub(_replace, text)
        # 띄어쓰기로 분리된 이름 변형 ("반욱 현" 같은 leak) — 캐시된 패턴 사용
        out = spaced_pattern.sub(name, out)
        return out

    @staticmethod
    def _extract_player_name(text: str) -> str | None:
        """플레이어 자기소개에서 이름 추출. 2-6자 한글/영문 단어.

        "반욱현이야"처럼 어미가 붙은 형태에서 어미를 분리해 이름만 캡처.
        (lazy 매칭 + 어미 alternation + 경계 lookahead)
        """
        _SUFFIX = r"(?:이라고|라고|이에요|입니다|이다|이야|예요|이오|임|야)"
        patterns = [
            rf"(?:내|제)\s*이름은\s+([가-힣A-Za-z]{{2,6}}?){_SUFFIX}?(?=[\s.,!?~]|$)",
            rf"^(?:나|저)는\s+([가-힣A-Za-z]{{2,6}}?){_SUFFIX}(?=[\s.,!?~]|$)",
            rf"^(?:내가|제가)\s+([가-힣A-Za-z]{{2,6}}?){_SUFFIX}(?=[\s.,!?~]|$)",
            r"([가-힣A-Za-z]{2,6}?)(?:이|)라고\s+(?:해|합니다|불러)",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                name = m.group(1).strip()
                # 안전망: 그래도 어미가 붙어 잡혔으면 제거 (2자 이상 남을 때만)
                for suf in ("이라고", "라고", "이에요", "입니다",
                            "이다", "이야", "예요", "이오", "임", "야"):
                    if len(name) - len(suf) >= 2 and name.endswith(suf):
                        name = name[: -len(suf)]
                        break
                # 일반 명사 제외 (이름 아닌 단어)
                if name in {"학생", "모험가", "사람", "여기", "거기", "그냥", "한번"}:
                    continue
                return name
        return None

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

        # ③ 전파 로그 — 누가 누구에게 전달했는지 (player_origin 우선 강조)
        if events:
            player_events = [e for e in events if e.get("player_origin")]
            other_events = [e for e in events if not e.get("player_origin")]
            # 플레이어 발화 전파를 우선 표시 (시연 핵심)
            for e in player_events[:5]:
                pipeline_log.log(
                    "spread",
                    f"Day{day}: {e['from']} → {e['to']}  [플레이어 정보 전파] "
                    f"\"{e['transformed'][:45]}\"",
                    day=day, frm=e["from"], to=e["to"], player_origin=True,
                )
            # 그 외 전파는 요약만
            if other_events:
                pipeline_log.log(
                    "spread",
                    f"Day{day}: 그 외 마을 소식 {len(other_events)}건 전파",
                    day=day, count=len(other_events), player_origin=False,
                )

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

        # 3단계: Reflection — 매 tick 1 NPC씩 추상화 (rotating)
        # day % len(characters)로 순환. 5종 NPC면 5 tick에 한 사이클.
        reflection_result = None
        try:
            npc_to_reflect = self.characters[(day - 1) % len(self.characters)]
            reflection_result = self.reflect(npc_to_reflect)
        except Exception as e:
            print(f"[tick] reflection 실패: {e}")

        # 4단계: 메모리 정리 — NPC별 최대 보유 메모리 제한
        pruned_total = 0
        for npc in self.characters:
            try:
                # current_day 전달 → retention(망각곡선) 낮은 기억부터 잊혀짐
                pruned_total += self.stores[npc].prune(max_keep=80, current_day=day)
            except Exception as e:
                print(f"[tick] {npc} 메모리 정리 실패: {e}")
        if pruned_total > 0:
            print(f"[tick] 메모리 정리: 총 {pruned_total}개 삭제")

        return {
            "day": day,
            "events": events,
            "conversation": conversation_result,
            "reflection": reflection_result,
        }

    def memory_counts(self) -> dict[str, int]:
        return {npc: self.stores[npc].count() for npc in self.characters}
