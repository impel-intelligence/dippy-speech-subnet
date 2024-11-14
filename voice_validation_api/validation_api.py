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

load_dotenv()

BLOCK_RATE_LIMIT = 14400  # Every 14400 blocks = 48 hours
app = FastAPI()
logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.ERROR)

admin_key = os.environ["ADMIN_KEY"]
hf_api = HfApi()

def repository_exists(repo_id):
    try:
        hf_api.repo_info(repo_id)  # 'username/reponame'
        return True
    except RepositoryNotFoundError:
        return False
    except GatedRepoError:
        # If we get a GatedRepoError, it means the repo exists but is private
        return False
    except Exception as e:
        app.state.event_logger.error("hf_repo_error", error=e)
        return False


@app.post("/telemetry_report")
async def telemetry_report(
    request: Request,
    git_commit: str = Header(None, alias="Git-Commit"),
    bittensor_version: str = Header(None, alias="Bittensor-Version"),
    uid: str = Header(None, alias="UID"),
    hotkey: str = Header(None, alias="Hotkey"),
    coldkey: str = Header(None, alias="Coldkey"),
):
    request_details = {
        "git_commit": git_commit,
        "bittensor_version": bittensor_version,
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
    }
    if request is not None:
        try:
            payload = await request.json()
            request_details = {**payload, **request_details}
        except Exception as e:
            if app.state.event_logger_enabled:
                app.state.event_logger.info("failed_telemetry_request", extra=request_details)

    # log incoming request details
    if app.state.event_logger_enabled:
        app.state.event_logger.info("telemetry_request", extra=request_details)
    return Response(status_code=200)
class EventData(BaseModel):
    commit: str
    btversion: str
    uid: str
    hotkey: str
    coldkey: str
    payload: Dict[Any, Any]
    signature: Dict[str, Any]

@app.post("/event_report")
async def event_report(event_data: EventData):
    try:
        if app.state.event_logger_enabled:
            app.state.event_logger.info("event_request", extra=event_data)
        return Response(status_code=200)
    except Exception as e:
        return Response(status_code=400, content={"error": str(e)})



class MinerboardRequest(BaseModel):
    uid: int
    hotkey: str
    hash: str
    block: int
    admin_key: Optional[str] = "admin_key"


@app.post("/minerboard_update")
def minerboard_update(
    request: MinerboardRequest,
):
    if request.admin_key != admin_key:
        return Response(status_code=403)

    app.state.db_client.update_minerboard_status(
        minerhash=request.hash,
        uid=request.uid,
        hotkey=request.hotkey,
        block=request.block,
    )
    return Response(status_code=200)


@app.get("/minerboard")
def get_minerboard():
    entries = app.state.db_client.minerboard_fetch()
    if len(entries) < 1:
        return []
    results = []
    for entry in entries:
        flattened_entry = {**entry.pop("leaderboard"), **entry}
        results.append(flattened_entry)
    return results

def hash_check(request: EvaluateModelRequest) -> bool:
    hash_matches = int(request.hash) == regenerate_hash(
        request.repo_namespace,
        request.repo_name,
        request.chat_template_type,
        request.competition_id,
    )
    hotkey_hash_matches = int(request.hash) == regenerate_hash(
        request.repo_namespace,
        request.repo_name,
        request.chat_template_type,
        request.hotkey,
    )
    if hash_matches or hotkey_hash_matches:
        return True
    return False


@app.post("/evaluate_model")
def evaluate_model(
    request: EvaluateModelRequest,
    git_commit: str = Header(None, alias="Git-Commit"),
    bittensor_version: str = Header(None, alias="Bittensor-Version"),
    uid: str = Header(None, alias="UID"),
    hotkey: str = Header(None, alias="Hotkey"),
    coldkey: str = Header(None, alias="Coldkey"),
    signed_payload: str = Header(None, alias="signed_payload"),
    miner_hotkey: str = Header(None, alias="miner_hotkey"),
):
    request_details = {
        "git_commit": git_commit,
        "bittensor_version": bittensor_version,
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
        "signed_payload": signed_payload,
        "miner_hotkey": miner_hotkey,
    }
    # log incoming request details
    if app.state.event_logger_enabled:
        app.state.event_logger.info("incoming_evaluate_request", extra=request_details)
    # verify hash
    hash_verified =  hash_check(request)
    if not hash_verified:
        raise HTTPException(status_code=400, detail="Hash does not match the model details")

    return supabaser.get_json_result(request.hash)


def update_failure(new_entry, failure_notes):
    # noop if already marked failed
    if new_entry["status"] == StatusEnum.FAILED:
        return new_entry
    new_entry["status"] = StatusEnum.FAILED
    new_entry["notes"] = failure_notes
    return new_entry


INVALID_BLOCK_START = 3840700
INVALID_BLOCK_END = 3933300

@app.post("/get_or_create_model")
def get_or_create_model(
    request: EvaluateModelRequest,
):
    # verify hash
    hash_verified = hash_check(request)
    if not hash_verified:
        raise HTTPException(status_code=400, detail="Hash does not match the model details")
    if request.admin_key != admin_key:
        raise HTTPException(status_code=403, detail="invalid key")

    early_failure = False
    failure_notes = ""
    # validate the repo exists
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    if not repository_exists(repo_id):
        failure_notes = f"Huggingface repo not public: {repo_id}"
        early_failure = True


    # only check if the model already exists in the hash_entries
    # update state if repo not public
    # This needs to be a virtually atomic operation
    try:
        current_entry = app.state.db_client.get_from_hash(request.hash)
        if current_entry is not None and early_failure:
            logger.error(failure_notes)
            internal_entry = app.state.db_client.get_internal_result(request.hash)
            internal_entry = update_failure(internal_entry, failure_notes)
            return app.state.db_client.upsert_and_return(internal_entry, request.hash)
        if current_entry is not None:
            return current_entry
        
    except Exception as e:
        logger.error(f"error while fetching request {request} : {e}")
        return None
    logger.info(f"COULD NOT FIND EXISTING MODEL for {request} : QUEUING NEW MODEL")

    # add the model to leaderboard with status QUEUED
    new_entry_dict = {
        "hash": request.hash,
        "repo_namespace": request.repo_namespace,
        "repo_name": request.repo_name,
        "total_score": 0,
        "timestamp": pd.Timestamp.utcnow(),
        "status": StatusEnum.QUEUED,
        "notes": failure_notes,
    }

    if early_failure:
        logger.error(failure_notes)
        updated_entry = update_failure(new_entry_dict, failure_notes)
        return app.state.db_client.upsert_and_return(updated_entry, request.hash)
        

    logger.info("QUEUING: " + str(new_entry_dict))

    last_model = app.state.db_client.last_uploaded_model(request.hotkey)
    if last_model is not None:
        last_model_status = StatusEnum.from_string(last_model["leaderboard"]["status"])
        if last_model_status != StatusEnum.FAILED:
            last_block = last_model.get("block", request.block)
            current_block = request.block
            # eg block 3001 - 2001 = 1000
            if abs(current_block - last_block) < BLOCK_RATE_LIMIT and abs(current_block - last_block) > 0:
                failure_notes = f"""
                Exceeded rate limit. 
                Last submitted model was block {last_block} with details {last_model}.
                Block difference between last block {last_block} and current block {current_block} is {(current_block - last_block)}
                Current submission {current_block} exceeds minimum block limit {BLOCK_RATE_LIMIT}"""
                logger.error(failure_notes)
                new_entry_dict = update_failure(new_entry_dict, failure_notes)

    # validate the repo exists
    repo_id = f"{request.repo_namespace}/{request.repo_name}"
    if not repository_exists(repo_id):
        failure_notes = f"Huggingface repo not public: {repo_id}"
        logger.error(failure_notes)
        new_entry_dict = update_failure(new_entry_dict, failure_notes)
        return app.state.db_client.upsert_and_return(new_entry_dict, request.hash)
    

    return app.state.db_client.upsert_and_return(new_entry_dict, request.hash)


@app.get("/hc")
def hc():
    return {"g": True, "k": False}


def start():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--main-api-port", type=int, default=8000, help="Port for the main API")
    parser.add_argument(
        "--queues",
        type=int,
        default=0,
        help="Specify the number of queues to start (default: 1)",
    )
    args = parser.parse_args()
    num_queues = args.queues
    MAIN_API_PORT = args.main_api_port
    app.state.event_logger_enabled = False
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