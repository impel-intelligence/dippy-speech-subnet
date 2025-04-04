from typing import Optional
from pydantic import BaseModel

# Constants
MAX_GENERATION_LEEWAY = 0.5  # should be between 0 and 1. This is the percentage of tokens that the model can generate more than the last user message
MAX_GENERATION_LENGTH = 200  # maximum number of tokens that the model can generate
LENGTH_DIFF_PENALTY_STEEPNESS = 2  # the steepness of the exponential decay of the length difference penalty
MAX_AVG_LATENCY = 10000  # in milliseconds
CREATIVITY_SCALE_FACTOR = 5

SAMPLE_SIZE = 1024  # number of samples to evaluate the model from the dataset
EVALUATION_DATASET_SAMPLE_SIZE = 4096  # number of samples to evaluate the model from the dataset
PROB_TOP_K = 10  # the correct token should be in the top n tokens, else a score of 0 is given to that token
# TODO: this will truncate the sequence to MAX_SEQ_LEN tokens. This is a temporary fix to make the evaluation faster.
MAX_SEQ_LEN = (
    4096  # maximum sequence length that should be allowed because eval gets really slow with longer sequences than this
)

DATASET_DIR = "evalsets"
MODEL_CACHE_DIR = "./model_cache_dir"

class EvaluateModelRequest(BaseModel):
    repo_namespace: str
    repo_name: str
    config_template: str = "default"
    hash: str
    revision: Optional[str] = "main"
    competition_id: Optional[str] = "d1"
    admin_key: Optional[str] = "admin_key"
    hotkey: Optional[str] = ""
    block: Optional[int] = 0
    tokenizer: Optional[str] = "llama"

    def to_args(self) -> str:
        return " ".join([self.repo_name, self.repo_namespace, self.config_template, self.hash])

