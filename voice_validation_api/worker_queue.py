import gc
import time
import os
import multiprocessing
import logging
import random
import argparse
from typing import List, Optional, Dict, Any

import torch
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError, GatedRepoError
from dotenv import load_dotenv

from voice_validation_api.evaluator import Evaluator, HumanSimilarityScore, RunError
from voice_validation_api.pg_persistence import Persistence
from common.scores import StatusEnum, Scores
from common.validation_utils import regenerate_hash
from common.repo_details import get_model_size, check_model_repo_details, ModelRepo
from voice_validation_api.duplicate import duplicate
from common.event_logger import EventLogger
from scoring.common import EvaluateModelRequest

logger = logging.getLogger(__name__)

class WorkerQueue:
    GPU_ID_MAP = {
        0: "0",
        1: "0",
        2: "4,5",
        3: "6,7",
        4: "8,9"
    }

    def __init__(self, image_name: str = "ubuntu:20.04", stub: bool = False):
        self.db_client = Persistence("postgresql://vapi:vapi@localhost:5432/vapi")
        self.event_logger = EventLogger()
        self.stub = stub
        self.image_name = image_name

    def model_evaluation_queue(self, queue_id: int) -> None:
        """Main queue processing loop for a single worker"""
        try:
            while True:
                self._model_evaluation_step(queue_id)
                time.sleep(5)
        except Exception as e:
            self.event_logger.error("queue_error", queue_id=queue_id, error=str(e))
            logger.error(f"queue_error, queue_id={queue_id}, error={str(e)}")

    def start_staggered_queues(self, num_queues: int, stagger_seconds: int) -> List[multiprocessing.Process]:
        if not self.db_client.ensure_connection():
            raise Exception("Could not connect to database")
        """Start multiple queue processes with staggered timing"""
        processes: List[multiprocessing.Process] = []
        for i in range(num_queues):
            p = multiprocessing.Process(target=self.model_evaluation_queue, args=(i,))
            processes.append(p)
            p.start()
            logger.info(f"Started queue {i}")
            time.sleep(stagger_seconds + i)
        return processes

    def _model_evaluation_step(self, queue_id: int, duplicate: bool = False) -> None:
        """Single step of model evaluation processing"""
        time.sleep(random.random())
        logger.info("Checking for models to evaluate")
        request = self._get_next_model_to_eval()
        if request is None:
            logger.info("No more models to evaluate. Sleep before checking again.")
            return

        self.event_logger.info(f"Model evaluation queued: {request} {queue_id}")
        try:
            result = self._evaluate_model(request, queue_id)
            if result is None:
                result = {"note": "incoherent model"}    
            self.event_logger.info("model_eval_queue_complete", result=result, request=request)
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            self.event_logger.info("model_eval_queue_error", error=str(e))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def _get_next_model_to_eval(self) -> Optional[EvaluateModelRequest]:
        """Get the next model from the queue to evaluate"""
        response = self.db_client.get_next_model_to_eval()
        
        if response is None:
            return None
        return EvaluateModelRequest(
            repo_namespace=response.repo_namespace,
            repo_name=response.repo_name,
            config_template="default",  # Using default since HashEntry doesn't contain config_template
            hash=response.hash,
        )

    def _evaluate_model(self, request: EvaluateModelRequest, queue_id: int) -> Optional[Dict[str, Any]]:
        """Evaluate a model based on the model size and quality"""
        self.db_client.update_leaderboard_status(
            request.hash,
            "RUNNING",
            "Model evaluation in progress starting with human similarity score",
        )
        evaluator = Evaluator(image_name=self.image_name, trace=True, gpu_ids=self.GPU_ID_MAP[queue_id])

        # Run evaluation scoring
        try:
            if self.stub:
                human_similarity_result = HumanSimilarityScore(human_similarity_score=0.1)
                time.sleep(30)
            else:
                human_similarity_result = evaluator.human_similarity_score(request=request)
            if isinstance(human_similarity_result, RunError):
                raise Exception(human_similarity_result.error)
        except Exception as e:
            error_string = f"Error calling human_similarity_score job with message: {e}"
            self.db_client.update_leaderboard_status(request.hash, StatusEnum.FAILED, error_string)
            raise RuntimeError(error_string)

        # Calculate final scores
        
        scores_data = Scores()
        scores_data.human_similarity_score = human_similarity_result.human_similarity_score

        try:
            self._upsert_scores(request.hash, {
                "total_score": scores_data.human_similarity_score,
                "status": StatusEnum.COMPLETED,
                "notes": "",
            })
        except Exception as e:
            failure_reason = str(e)
            logger.error(f"Updating leaderboard to FAILED: {failure_reason}")
            self.db_client.update_leaderboard_status(request.hash, StatusEnum.FAILED, failure_reason)
            raise RuntimeError("Error updating leaderboard: " + failure_reason)

        return {"full_score_data": scores_data}

    def _upsert_scores(self, hash_id: str, data: Dict[str, Any]) -> None:
        """Update scores in the database"""
        data["hash"] = hash_id
        self.db_client.upsert_row(data)

def main():
    parser = argparse.ArgumentParser(description="Run the worker instance")
    parser.add_argument(
        "--queues",
        type=int,
        default=1,
        help="Specify the number of queues to start (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    worker = WorkerQueue(stub=True)
    processes = []
    try:
        logger.info(f"Starting {args.queues} evaluation threads")
        processes = worker.start_staggered_queues(args.queues, stagger_seconds=2)
        logger.info("Worker queues started successfully")
        
        # Keep the main process running until interrupted
        while True:
            for process in processes:
                if not process.is_alive():
                    logger.error("A worker process died unexpectedly")
                    raise Exception("Worker process died")
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
    finally:
        logger.info("Stopping evaluation threads")
        for process in processes:
            process.terminate()
            process.join()

if __name__ == "__main__":
    main()