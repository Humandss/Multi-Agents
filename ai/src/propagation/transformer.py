"""페르소나별 정보 변형.

각 NPC가 자신의 LoRA 어댑터를 통해 정보를 자신의 어조로 다시 표현한다.
- hermann: 단답·사실
- mathilda: 사교적·풀어쓰기
- finn: 시적·과장
- bernhardt: 실용적·계산
- elias: 회의적·검증

베이스 모델 1개에 5종 어댑터를 로드해서 set_adapter()로 스위칭.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"

TRANSFORM_PROMPT = (
    "다음 사실을 다른 마을 사람에게 한 마디로 전달한다면 어떻게 말할지 한 줄로만 답하세요. "
    "다른 설명이나 라벨은 붙이지 마세요.\n\n"
    "사실: {memory}\n\n"
    "당신의 한 마디:"
)


class PersonaTransformer:
    """5종 NPC LoRA를 동적으로 스위칭하며 정보 변형."""

    def __init__(self, adapter_paths: dict[str, Path]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_REVISION,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # 첫 어댑터를 default로 로드
        names = list(adapter_paths.keys())
        first = names[0]
        self.model = PeftModel.from_pretrained(
            base, str(adapter_paths[first]), adapter_name=first
        )
        for name in names[1:]:
            self.model.load_adapter(str(adapter_paths[name]), adapter_name=name)
        self.model.eval()

        self._cache = {}

    def transform(self, sender_npc: str, memory_text: str, max_new_tokens: int = 80) -> str:
        cache_key = (sender_npc, memory_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self.model.set_adapter(sender_npc)
        prompt = TRANSFORM_PROMPT.format(memory=memory_text)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()
        # 첫 줄만 + 따옴표 정리
        text = text.split("\n")[0].strip().strip('"').strip("'")
        if not text:
            text = memory_text  # fallback

        self._cache[cache_key] = text
        return text
