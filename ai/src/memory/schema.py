"""메모리 엔트리 스키마."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemorySource(str, Enum):
    OBSERVATION = "observation"     # 직접 관찰
    DIALOGUE = "dialogue"           # 플레이어와 대화
    CONVERSATION = "conversation"   # NPC-NPC 대화 (회상 활용 가능)
    PROPAGATION = "propagation"     # 다른 NPC로부터 전달받음
    SEED = "seed"                   # 초기 시드 (배경 지식)
    REFLECTION = "reflection"       # LLM이 저수준 메모리에서 추출한 추상 통찰 (Park et al.)


class MemoryEntry(BaseModel):
    id: str
    text: str
    importance: int = Field(default=5, ge=1, le=10)
    timestamp: datetime
    source: MemorySource = MemorySource.OBSERVATION
    metadata: dict[str, Any] = Field(default_factory=dict)
