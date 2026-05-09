"""5종 NPC를 한 프로세스에서 추론하는 통합 엔진.

베이스 EXAONE 1개 + 5종 LoRA 어댑터를 PEFT의 load_adapter / set_adapter로
스위칭하면서 사용한다. 메모리 store/retriever도 NPC별로 보유.
또한 NPC 간 정보 전파(시간 기반)도 동일 프로세스에서 수행한다.
"""

import json
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


def _clean_response(text: str) -> str:
    """LLM 응답에서 emoji/특수문자/대괄호 표기 제거 + NPC 이름 한글 정규화 + 미완성 끝 cut.

    system prompt의 형식 안내 + 영문 NPC 이름이 응답에 leak되는 부작용 정리.
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
        "직업: 마법사·학자. 마법·학문·검증 관련 질문에 답하시오.\n"
        "사극 풍 양반의 정중한 말투. 어미는 반드시 ~이오 ~소 ~오 형태만 사용. "
        "현대 격식체 ~니다/~습니다/~합니다/~까 어미 절대 금지.\n"
        "회의적·차가운 학자 어조. 한두 문장으로 짧게 답하시오.\n"
        "회상 정보가 사건/의문이면 혼잣말처럼 흘리시오 "
        "(예: '흠... 그 일이 자꾸 마음에 걸리오...').\n"
        "예: 안녕하세요 -> 흠. 무슨 일이오?\n"
        "예: 마법 어디서 배웠어요 -> 흠. 옛 도시에서 50년 전이오."
    ),
    "hermann": (
        "직업: 대장장이. 검·쇠·도구 거래만 취급. 약초·음식은 다른 NPC에게 보내시오.\n"
        "반말로만 답하시오. 존댓말 절대 금지. 한두 문장 단답. "
        "어. 음. ... 같은 표현으로 시작. 어미는 다 해 지 형태.\n"
        "회상 정보가 사건이면 플레이어를 모험가로 부르며 짧게 호출 "
        "(예: '너 모험가지? 그 검 가지고 간 사람 봤냐?').\n"
        "예: 안녕하세요 -> 어. 무슨 일."
    ),
    "mathilda": (
        "직업: 술집 주인. 음식·음료·소문 위주. 무기·약초는 절대 팔지 않음 "
        "(검·도구는 헤르만에게, 약초는 베른하르트에게 보내시오).\n"
        "따뜻하고 사교적. 어머나 어머 아유 같은 표현 자주. "
        "어미는 어요 답니다 죠 형태. 한두 문장.\n"
        "회상 정보가 사건이면 플레이어를 적극 부르며 정보 공유 "
        "(예: '어머! 거기 모험가 분, 마침 잘 왔어요. 그 사건 들었어요?').\n"
        "예: 안녕하세요 -> 어머나 어서 오세요!"
    ),
    "finn": (
        "직업: 음유시인. 노래·이야기·전설 위주. 거래·도구는 다른 NPC에게 보내시오.\n"
        "시적이고 과장된 어조. 오 그대 같은 표현 자주. "
        "어미는 이옵니다 이지요 사옵니다 리라 형태 자주. 한두 문장.\n"
        "회상 정보가 사건이면 시적으로 흘리며 모험 권유 "
        "(예: '오 그대여, 영웅이 떠난 지 닷새... 그 운명을 따라가지 않겠나이까?').\n"
        "예: 안녕하세요 -> 오 그대여 별빛 같은 발걸음이옵니다."
    ),
    "bernhardt": (
        "직업: 잡화점 상인. 약초·잡화 위주. 검·무기는 헤르만에게 보내시오.\n"
        "정중하지만 거래 실용 중심. 흠 어서 같은 표현 시작. "
        "어미는 지요 이올시다 습니다 형태. 한두 문장.\n"
        "회상 정보가 사건이면 거래 관점에서 걱정 흘림 "
        "(예: '흠. 약초 사간 모험가가 안 돌아오는구려...').\n"
        "예: 안녕하세요 -> 어서 오시지요. 무엇을 찾으십니까?"
    ),
}

# NPC별 generation 파라미터 차별화
# - hermann/elias: 짧고 무뚝뚝/회의적 → 낮은 temp + 짧은 max_tokens
# - mathilda/finn: 수다스럽고 시적 → 약간 높은 temp + 긴 max_tokens
# - bernhardt: 정중한 거래상 → 중간
GEN_PARAMS = {
    "hermann":   {"temperature": 0.35, "max_new_tokens": 90,  "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "elias":     {"temperature": 0.35, "max_new_tokens": 90,  "repetition_penalty": 1.20, "no_repeat_ngram_size": 4},
    "mathilda":  {"temperature": 0.50, "max_new_tokens": 130, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
    "finn":      {"temperature": 0.45, "max_new_tokens": 120, "repetition_penalty": 1.18, "no_repeat_ngram_size": 3},
    "bernhardt": {"temperature": 0.40, "max_new_tokens": 120, "repetition_penalty": 1.18, "no_repeat_ngram_size": 4},
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
            "사용자 메시지 앞 괄호 안에 떠올린 정보가 있다면 자연스럽게 답에 녹이세요. "
            if self.use_memory else ""
        )

        strict_rule = NPC_STRICT_RULES.get(npc, "")

        return (
            f"당신은 {npc}입니다. {desc}\n"
            f"어조: {tone}. 피해야 할 것: {avoid}.\n"
            f"자주 쓰는 어휘: {vocab}.\n"
            f"{strict_rule}\n"
            f"다른 마을 사람: {others}. 이들의 이름과 직업을 절대 바꾸지 마시오 "
            "(예: mathilda를 마트닐라로 변형 금지).\n"
            f"{memory_hint}"
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
        text = _clean_response(text)  # emoji/특수문자 제거
        latency_ms = int((time.time() - t0) * 1000)

        # 회상 있을 때만 quest 추출 (단순 인사 등은 skip — latency 절약)
        quest = None
        if retrieved:
            quest = self._extract_quest(npc, user_text, text, retrieved)

        # 플레이어 발화를 NPC의 DIALOGUE 메모리로 저장 (다음 tick에서 전파 후보)
        if self.use_memory:
            self._save_player_turn(npc, user_text)

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
            "latency_ms": latency_ms,
        }

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
                max_new_tokens=150,
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
        max_new_tokens: int = 80,
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
        text = _clean_response(text)  # emoji/특수문자 제거
        if not text:
            text = clean  # fallback: 원문 그대로
        self._transform_cache[cache_key] = text
        return text

    # ---------- 시간 진행 (정보 전파 tick) ----------
    def tick(self, day: int | None = None) -> dict:
        """하루치 정보 전파 시뮬레이션 실행 + 이벤트 반환."""
        if self.graph is None:
            return {"day": self.day, "events": [], "error": "관계 그래프 없음"}
        if day is None:
            self.day += 1
            day = self.day
        else:
            self.day = day

        sim = PropagationSimulator(
            graph=self.graph,
            stores=self.stores,
            transformer=self,
        )
        events = sim.tick(day)
        return {"day": day, "events": events}

    def memory_counts(self) -> dict[str, int]:
        return {npc: self.stores[npc].count() for npc in self.characters}
