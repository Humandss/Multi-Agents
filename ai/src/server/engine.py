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
    예: "권장드립니다지만, 일반적으로 검류는 헤르만에게 문의하시길 권하노" → "권장드립니다지만, 일반적으로 검류는 헤르만에게 문의하시길"
        (이 경우 마침표 없으면 그대로 — 추가 처리 필요할 수 있음)
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
        # "~십니까?" → "~시오?" (사극체 학자 어조 강제)
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "시오?"),
        (re.compile(r"([가-힣])십니까\b"), lambda m: m.group(1) + "시오"),
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
    ],
    "hermann": [
        # hermann은 반말. 존댓말 leak 시 반말로 강제.
        (re.compile(r"하세요\b"), "해"),
        (re.compile(r"드릴게요\b"), "줄게"),
        (re.compile(r"드립니다\b"), "준다"),
        (re.compile(r"있습니다\b"), "있어"),
        (re.compile(r"없습니다\b"), "없어"),
        (re.compile(r"합니다\b"), "해"),
        (re.compile(r"입니다\b"), "이야"),
        (re.compile(r"됩니다\b"), "돼"),
        (re.compile(r"([가-힣])십니까\?"), lambda m: m.group(1) + "냐?"),
        (re.compile(r"(이|있|없)어요\b"), lambda m: m.group(1) + "어"),
        (re.compile(r"([가-힣])세요\b"), lambda m: m.group(1) + "해"),
    ],
    "bernhardt": [
        # bernhardt는 "~지요/~습니다" 자연. 일부 패턴만.
        # "~죠" → "~지요" (더 정중하게)
        (re.compile(r"하죠\b"), "하지요"),
        (re.compile(r"있죠\b"), "있지요"),
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
        "차분한 학자 어조, 약간 회의적. 어미는 ~오 ~이오 ~구려 위주. "
        "절대 금지: ~니다 ~습니다 ~소이옵니다 ~옵소서 같은 과한 사극체.\n"
        "한 문장으로 답하시오. 흠 으로 시작 자주.\n"
        "예: 안녕하세요 -> 흠, 무슨 일이오?\n"
        "예: 마법 어디서 배웠어요 -> 옛 도시에서 50년 익혔구려.\n"
        "예: 광산은 안전한가요 -> 흠... 그 일이 마음에 걸리오."
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

# NPC별 Quest 안내 template — LLM 응답 뒤에 자연스럽게 이어붙임.
# {title}/{description}/{reward}만 채우면 됨. 페르소나 어조 유지하므로 추가 LLM 호출 불필요 = 빠름.
NPC_QUEST_INTRO = {
    "elias": (
        " 흠... 한 가지 부탁이 있소이다. "
        "「{title}」 — {description} "
        "성공하면 {reward} 보답하리오."
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
            # 같은 NPC와 대화 중에는 자기가 들은 플레이어 발화(DIALOGUE)를 회상하지 않음.
            # (전파를 거쳐 다른 NPC가 PROPAGATION으로 다시 받게 되면 그건 회상 가능)
            retrieved = self.retrievers[npc].search(
                user_text, k=self.retrieval_k, exclude_sources={"dialogue"}
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

        # Quest 추출은 중요 회상이 있을 때만 (latency 절약)
        # importance 7+: 시드 또는 propagation으로 강조된 사건만 quest 후보.
        quest = None
        high_imp = any(m["importance"] >= 7 for m in retrieved) if retrieved else False
        if high_imp:
            quest = self._extract_quest(npc, user_text, text, retrieved)
            # Quest가 추출되면 NPC 페르소나 template으로 안내문 자동 첨부
            # LLM 응답 (짧음) + Quest template (페르소나 어조) = 자연스럽고 빠름
            if quest is not None:
                template = NPC_QUEST_INTRO.get(npc)
                if template:
                    quest_intro = template.format(
                        title=quest.get("title", ""),
                        description=quest.get("description", ""),
                        reward=quest.get("reward", "응당한") or "응당한",
                    )
                    text = text.rstrip() + quest_intro

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

    def complete_quest(self, npc: str) -> dict:
        """Quest 완수 시 호출 — 신뢰도 +10."""
        if npc not in self.characters:
            raise ValueError(f"알 수 없는 NPC: {npc}")
        delta = self.trust.on_quest_complete(npc)
        return {
            "npc": npc,
            "trust": self.trust.get(npc),
            "trust_label": self.trust.label(npc),
            "trust_delta": delta,
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
                    topic = random.choice(cand)["text"]
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
        if len(text) < 8:
            return  # 인사·감탄사는 저장 X (전파 가치 없음)

        # 질문문 vs 평서문 분기 — 질문은 fact가 아니므로 전파 후보에서 제외 (importance < threshold)
        is_question = "?" in text or any(
            text.endswith(suf) for suf in ["어요", "지요", "나요", "까", "야"]
        )
        # 사실 보고 키워드: 있으면 전파 가치 ↑
        fact_kw = ["나타났", "사라졌", "잡았", "봤", "들었", "있었", "갔다", "왔다", "했다", "됐다", "당했"]
        has_fact = any(kw in text for kw in fact_kw)

        if is_question:
            importance = 4   # 전파 X, 자유 대화 컨텍스트로만 사용
        elif has_fact and len(text) >= 12:
            importance = 9   # 평서문 + 사실 키워드 → 시드보다 강하게 전파
        elif len(text) >= 30:
            importance = 7
        elif len(text) >= 15:
            importance = 6
        else:
            importance = 5

        entry = MemoryEntry(
            id=f"dlg_{uuid.uuid4().hex[:8]}",
            text=f"플레이어가 말했다: {text}",
            importance=importance,
            timestamp=datetime.now(timezone.utc),
            source=MemorySource.DIALOGUE,
            metadata={"player": True, "is_question": is_question, "has_fact": has_fact},
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
        npc_conversation_turns: int = 1,  # 2→1: 각 NPC 1회 발화 (속도 ↑)
    ) -> dict:
        """하루치 정보 전파 시뮬레이션 + 1쌍 NPC-NPC 자율 대화.

        - 1단계: propagation (전파)
        - 2단계: graph 무작위 페어 → simulate_conversation (Park et al. style)
        - 두 결과 묶어 반환
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

        return {
            "day": day,
            "events": events,
            "conversation": conversation_result,
        }

    def memory_counts(self) -> dict[str, int]:
        return {npc: self.stores[npc].count() for npc in self.characters}
