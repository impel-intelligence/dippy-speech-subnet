from enum import Enum
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field
import math

CREATIVITY_STEEPNESS = 5
CREATIVITY_THRESHOLD = 0.2
ALPHA_SCORE_WEIGHT = 0.82  # weight of the qualitative score in the total score
MODEL_SIZE_SCORE_WEIGHT = 0.06  # weight of the model size score in the total score
GAMMA_SCORE_WEIGHT = 0.06  # weight of the latency score in the total score
BETA_SCORE_WEIGHT = 0.06  # weight of the vibe score in the total score

class StrEnum(str, Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"{value} is not a valid {cls.__name__}")


class StatusEnum(StrEnum):
    QUEUED = "QUEUED"
    PRECHECK = "PRECHECK"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RUNNING = "RUNNING"
    NO_METADATA = "NO_METADATA"
    ERROR = "ERROR"

class Scores(BaseModel):
    human_similarity_score: float = Field(default=0, description="How similar the voice generated is to being human like")

