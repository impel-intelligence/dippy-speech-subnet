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
    total_score: float = Field(default=0, description="The total score of the evaluation")
    alpha_score: float = Field(default=0, description="The coherence score of the text")
    beta_score: float = Field(default=0, description="The vibe score of the text") 
    gamma_score: float = Field(default=0, description="The creativity score")
    llm_size_score: float = Field(default=0, description="The model_size score of the text")
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")

    @staticmethod
    def adjusted_q_score(
        initial_score: float,
    ):
        adjusted_score = initial_score
        return adjusted_score

    def from_response(self, response: Dict[str, Any]):
        if response is None or len(response) < 1:
            self.total_score = 0
            return self
        self.llm_size_score = response.get("model_size_score", 0)
        self.gamma_score = response.get("gamma_score", 0)
        self.alpha_score = response.get("alpha_score", 0)
        self.beta_score = response.get("beta_score", 0)
        return self

    def calculate_total_score(self, adjust_coherence: bool = False) -> float:
        q_score = self.adjusted_q_score(self.alpha_score)
        total_score = 0
        total_score += ALPHA_SCORE_WEIGHT * q_score
        total_score += MODEL_SIZE_SCORE_WEIGHT * self.llm_size_score
        total_score += BETA_SCORE_WEIGHT * self.beta_score
        total_score += GAMMA_SCORE_WEIGHT * self.gamma_score
        return total_score

    def calculate_new_total_score(self, adjust_coherence: bool = False) -> float:
        return self.calculate_total_score()
