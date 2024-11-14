import gc
import time
import os
import multiprocessing
import logging
from typing import List, Optional

import uvicorn

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Header, Request, Response
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError, GatedRepoError
from dotenv import load_dotenv
import random
from pydantic import BaseModel
from typing import Dict, Any

from voice_validation_api.evaluator import Evaluator, RunError
from voice_validation_api.pg_persistence import Persistence
from common.scores import StatusEnum, Scores
from common.validation_utils import (
    regenerate_hash,
)
from common.repo_details import (
    get_model_size,
    check_model_repo_details,
    ModelRepo,
)
from voice_validation_api.duplicate import duplicate
from common.event_logger import EventLogger
from scoring.common import EvaluateModelRequest, chat_template_mappings
from dotenv import load_dotenv
from huggingface_hub import HfApi, list_models



def model_evaluation_queue(queue_id):
    try:
        while True:
            _model_evaluation_step(queue_id)
            time.sleep(5)
    except Exception as e:
        app.state.event_logger.error("queue_error", queue_id=queue_id, error=e)


def start_staggered_queues(num_queues: int, stagger_seconds: int):
    processes: List[multiprocessing.Process] = []
    for i in range(num_queues):
        p = multiprocessing.Process(target=model_evaluation_queue, args=(i,))
        processes.append(p)
        p.start()
        logger.info(f"Started queue {i}")
        time.sleep(stagger_seconds + i)
    return processes


def _model_evaluation_step(queue_id, duplicate: bool = False):
    time.sleep(random.random())

    request = get_next_model_to_eval()
    if request is None:  # Sentinel value to exit the process
        logger.info("No more models to evaluate. Sleep for 15 seconds before checking again.")
        return
    app.state.event_logger.info(f"Model evaluation queued: {request} {queue_id}")
    try:
        # if duplicate:
        #     _duplicate_model(request)
        result = _evaluate_model(request, queue_id)
        if result is None:
            result = {"note": "incoherent model"}    
        app.state.event_logger.info("model_eval_queue_complete", result=result, request=request)
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        app.state.event_logger.info("model_eval_queue_error", error=e)
    finally:
        gc.collect()  # Garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Empty CUDA cache


def get_next_model_to_eval():
    response = supabaser.get_next_model_to_eval()
    
    if response is None:
        return None
    return EvaluateModelRequest(
        repo_namespace=response["repo_namespace"],
        repo_name=response["repo_name"],
        chat_template_type=response["chat_template_type"],
        hash=response["hash"],
    )


def _duplicate_model(request: EvaluateModelRequest):
    try:
        duplicate(request.repo_namespace, request.repo_name)
    except Exception as e:
        supabaser.update_leaderboard_status(
            request.hash,
            "FAILED",
            f"model error : {e}",
        )

GPU_ID_MAP = {
    0: "0",
    1: "0",
    2: "4,5",
    3: "6,7",
    4: "8,9"
}


def _evaluate_model(
    request: EvaluateModelRequest,
    queue_id: int,
):
    """
    Evaluate a model based on the model size and the quality of the model.
    """
    supabaser.update_leaderboard_status(
        request.hash,
        "RUNNING",
        "Model evaluation in progress starting with inference score",
    )

    evaluator = Evaluator(gpu_ids=GPU_ID_MAP[queue_id])
    try:
        inference_response = evaluator.inference_score(request)
        if isinstance(inference_response, RunError):
            raise Exception(inference_response.error)
        vibe_score = inference_response.vibe_score
        coherence_score = inference_response.coherence_score

        if coherence_score < 0.95:
            supabaser.update_leaderboard_status(request.hash, StatusEnum.COMPLETED, "Incoherent model submitted")
            return None
        upsert_row_supabase(
        {
            "hash": request.hash,
            "vibe_score": vibe_score,
            "coherence_score": coherence_score,
            "notes": "Now computing evaluation score",
        }
    )
    except Exception as e:
        error_string = f"Error calling inference_score job with message: {e}"
        supabaser.update_leaderboard_status(request.hash, StatusEnum.FAILED, error_string)
        raise RuntimeError(error_string)

    try:
        eval_score_result = evaluator.eval_score(request)
        if isinstance(eval_score_result, RunError):
            raise Exception(eval_score_result.error)
    except Exception as e:
        error_string = f"Error calling eval_score job with message: {e}"
        supabaser.update_leaderboard_status(
            request.hash,
            StatusEnum.FAILED,
            error_string,
        )
        raise RuntimeError(error_string)

    eval_score = eval_score_result.eval_score
    latency_score = eval_score_result.latency_score
    model_size_score = eval_score_result.eval_model_size_score
    creativity_score = eval_score_result.creativity_score

    if eval_score is None or latency_score is None or model_size_score is None or vibe_score is None:
        raise HTTPException(
            status_code=500,
            detail="Error calculating scores, one or more scores are None",
        )

    full_score_data = Scores()
    full_score_data.qualitative_score = eval_score
    full_score_data.llm_size_score = model_size_score
    full_score_data.coherence_score = coherence_score
    full_score_data.creativity_score = creativity_score
    full_score_data.vibe_score = vibe_score
    full_score_data.latency_score = latency_score

    try:
        upsert_row_supabase(
            {
                "hash": request.hash,
                "model_size_score": full_score_data.llm_size_score,
                "qualitative_score": full_score_data.qualitative_score,
                "creativity_score": full_score_data.creativity_score,
                "latency_score": full_score_data.latency_score,
                "total_score": full_score_data.calculate_total_score(),
                "status": StatusEnum.COMPLETED,
                "notes": "",
            }
        )
    except Exception as e:
        failure_reason = str(e)
        logger.error(f"Updating leaderboard to FAILED: {failure_reason}")
        supabaser.update_leaderboard_status(request.hash, StatusEnum.FAILED, failure_reason)
        raise RuntimeError("Error updating leaderboard: " + failure_reason)
    result = {
        "full_score_data": full_score_data,
    }
    return result




def start():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument(
        "--queues",
        type=int,
        default=1,
        help="Specify the number of queues to start (default: 1)",
    )
    args = parser.parse_args()
    num_queues = args.queues
    MAIN_API_PORT = args.main_api_port
    try:
        event_logger = EventLogger()
        app.state.event_logger = event_logger
        app.state.event_logger_enabled = True
    except Exception as e:
        logger.warning(f"Failed to create event logger: {e}")

    db_url = f"postgresql://vapi:vapi@localhost:5432/vapi"
    print(f"f{app.state}")
    try:
        db_client = Persistence()
        app.state.db_client = db_client
    except Exception as e:
        logger.warning(f"Failed to create db client: {e}")
        return

    processes = []
    stagger_seconds = 2
    try:
        from voice_validation_api.worker_queue import start_staggered_queues
        logger.info(f"Starting {num_queues} evaluation threads")
        processes = start_staggered_queues(num_queues, stagger_seconds)
        logger.info("Starting API server")
        uvicorn.run(app, host="0.0.0.0", port=MAIN_API_PORT)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
    finally:
        logger.info("Stopping evaluation thread")
        for process in processes:
            process.terminate()
            process.join()


if __name__ == "__main__":
    start()