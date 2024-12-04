from enum import Enum
from typing import Optional, Any, Dict

from pydantic import BaseModel, Field
import math

HUMAN_SIMILARITY_WEIGHT = 1


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
    human_similarity_score: float = Field(
        default=0, description="How similar the voice generated is to being human-like"
    )
    status: str = Field(default=StatusEnum.QUEUED, description="The current status of the scoring process")

    def from_response(self, response: Dict[str, Any]):
        """Populate the Scores instance from a response dictionary."""
        if not response:
            self.total_score = 0
            return self

        self.human_similarity_score = response.get("total_score", 0)
        return self

    def calculate_total_score(self, adjust_coherence: bool = False) -> float:
        """Calculate the total score and optionally adjust it for coherence."""
        base_score = self.human_similarity_score
        total_score = HUMAN_SIMILARITY_WEIGHT * base_score

        # Optionally update the instance's total_score if needed
        self.total_score = total_score
        return total_score
