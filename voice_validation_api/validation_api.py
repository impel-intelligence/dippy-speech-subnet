import os
import logging
from typing import Optional, Dict, Any

import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request, Response, APIRouter, Depends, Security
from huggingface_hub import HfApi
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError, GatedRepoError
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.status import HTTP_403_FORBIDDEN

from voice_validation_api.pg_persistence import Persistence
from common.scores import StatusEnum
from common.validation_utils import regenerate_hash
from common.event_logger import EventLogger
from scoring.common import EvaluateModelRequest

load_dotenv()

BLOCK_RATE_LIMIT = 14400  # Every 14400 blocks = 48 hours
logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.ERROR)

class EventData(BaseModel):
    commit: str
    btversion: str
    uid: str
    hotkey: str
    coldkey: str
    payload: Dict[Any, Any]
    signature: Dict[str, Any]

class MinerboardRequest(BaseModel):
    uid: int
    hotkey: str
    hash: str
    block: int
    admin_key: Optional[str] = "admin_key"

class AdminKeyMiddleware:
    def __init__(self, admin_key: str):
        self.admin_key = admin_key
        self.api_key_header = APIKeyHeader(name="admin-key", auto_error=False)

    async def __call__(self, api_key: str = Security(APIKeyHeader(name="admin-key"))):
        if not api_key or api_key != self.admin_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid or missing admin key"
            )
        return api_key

class ValidationAPI:
    def __init__(self, retries: int = 3):
        self.app = FastAPI()
        self.router = APIRouter()
        self.admin_key = os.environ.get("ADMIN_KEY", "admin_key")
        self.hf_api = HfApi()
        self.event_logger_enabled = False
        self.retries = retries
        self.setup_routes()
        self.setup_state()

    def setup_state(self):
        """Initialize API state"""
        try:
            self.event_logger = EventLogger()
            self.event_logger_enabled = True
        except Exception as e:
            logger.warning(f"Failed to create event logger: {e}")

        try:
            self.db_client = Persistence(connection_string=os.environ.get("POSTGRES_URL"))
            self.db_client.run_migrations()
        except Exception as e:
            logger.warning(f"Failed to create db client: {e}")

    def setup_routes(self):
        """Register all routes with the router"""
        # Initialize admin middleware
        admin_auth = AdminKeyMiddleware(self.admin_key)
        
        # Protected admin routes
        self.router.add_api_route(
            "/minerboard_update", 
            self.minerboard_update, 
            methods=["POST"],
            dependencies=[Depends(admin_auth)]
        )
        self.router.add_api_route(
            "/ensure_model", 
            self.ensure_model, 
            methods=["POST"],
            dependencies=[Depends(admin_auth)]
        )
        
        # Unprotected routes
        self.router.add_api_route("/telemetry_report", self.telemetry_report, methods=["POST"])
        self.router.add_api_route("/event_report", self.event_report, methods=["POST"])
        self.router.add_api_route("/minerboard", self.get_minerboard, methods=["GET"])
        self.router.add_api_route("/recent_entries", self.get_recent_entries, methods=["GET"])
        self.router.add_api_route("/model_submission_details", self.get_model_submission_details, methods=["GET"])
        self.router.add_api_route("/hc", self.hc, methods=["GET"])
        self.router.add_api_route("/", self.health, methods=["GET"])
        self.router.add_api_route("/routes", self.get_routes, methods=["GET"])
        
        # Include router in app
        self.app.include_router(self.router)

    def repository_exists(self, repo_id: str) -> bool:
        # Retry 3 times while checking repo availability
        for retried in rang(3):
            try:
                self.hf_api.repo_info(repo_id)
                return True
            except Exception as e:
                if self.event_logger_enabled:
                    self.event_logger.error("hf_repo_error", error=e)
                # No need to sleep in the 3rd time error
                if retried < 2:
                    time.sleep(retried + 1) # linear growth
        # Repo is not public after 3 times retried
        return False
    @staticmethod
    def hash_check(request: EvaluateModelRequest) -> bool:
        hash_matches = int(request.hash) == regenerate_hash(
            request.repo_namespace,
            request.repo_name,
            request.config_template,
            request.hotkey,
        )
        hotkey_hash_matches = int(request.hash) == regenerate_hash(
            request.repo_namespace,
            request.repo_name,
            request.config_template,
            request.hotkey,
        )
        return hash_matches or hotkey_hash_matches

    def update_failure(self, new_entry: dict, failure_notes: str) -> dict:
        if new_entry["status"] == StatusEnum.FAILED:
            return new_entry
        new_entry["status"] = StatusEnum.FAILED
        new_entry["notes"] = failure_notes
        return new_entry

    async def telemetry_report(
        self,
        request: Request,
        git_commit: str = Header(None, alias="Git-Commit"),
        bittensor_version: str = Header(None, alias="Bittensor-Version"),
        uid: str = Header(None, alias="UID"),
        hotkey: str = Header(None, alias="Hotkey"),
        coldkey: str = Header(None, alias="Coldkey"),
    ):
        """POST /telemetry_report - Endpoint for receiving telemetry data from clients"""
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
                if self.event_logger_enabled:
                    self.event_logger.info("failed_telemetry_request", extra=request_details)

        if self.event_logger_enabled:
            self.event_logger.info("telemetry_request", extra=request_details)
        return Response(status_code=200)

    async def event_report(self, event_data: EventData):
        """POST /event_report - Endpoint for receiving event reports"""
        try:
            if self.event_logger_enabled:
                self.event_logger.info("event_request", extra=event_data)
            return Response(status_code=200)
        except Exception as e:
            return Response(status_code=400, content={"error": str(e)})

    def minerboard_update(self, request: MinerboardRequest):
        """POST /minerboard_update - Protected endpoint for updating miner board status"""
        self.db_client.update_minerboard_status(
            hash_entry=request.hash,
            uid=request.uid,
            hotkey=request.hotkey,
            block=request.block,
        )
        return Response(status_code=200)

    def get_minerboard(self):
        """GET /minerboard - Endpoint for retrieving all miner entries"""
        entries = self.db_client.fetch_all_miner_entries()
        if entries is None:
            return []
        if len(entries) < 1:
            return []
        return entries
    
    """GET /model_submission_details - Endpoint for retrieving model submission details"""
    def get_model_submission_details(
            self,
        repo_namespace: str,
        repo_name: str,
        chat_template_type: str,
        hash: str,
        competition_id: Optional[str] = None,
        hotkey: Optional[str] = None,
        ):
        
        request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        config_template=chat_template_type,
        hash=hash,
        hotkey=hotkey,
    )
        # verify hash
        hash_verified = self.hash_check(request)
        if not hash_verified:
            raise HTTPException(status_code=400, detail="Hash does not match the model details")

        return self.db_client.get_json_result(request.hash)


    """POST /ensure_model - Protected endpoint for retrieving or creating a model entry"""
    def ensure_model(self, request: EvaluateModelRequest):
        
        if not self.hash_check(request):
            raise HTTPException(status_code=400, detail="Hash does not match the model details")

        early_failure = False
        failure_notes = ""
        repo_id = f"{request.repo_namespace}/{request.repo_name}"
        
        if not self.repository_exists(repo_id):
            failure_notes = f"Huggingface repo not public: {repo_id}"
            early_failure = True
        try:
            retries = self.retries
            current_entry = None
            while retries > 0 and current_entry is None:
                current_entry = self.db_client.get_from_hash(request.hash)
                retries -= 1
            if current_entry is not None and early_failure:
                logger.error(failure_notes)
                internal_entry = self.db_client.get_internal_result(request.hash)
                internal_entry = self.update_failure(internal_entry, failure_notes)
                return self.db_client.upsert_and_return(internal_entry, request.hash)
            if current_entry is not None:
                return current_entry
            
        except Exception as e:
            logger.error(f"error while fetching request {request} : {e}")
            return None

        logger.info(f"COULD NOT FIND EXISTING MODEL for {request} : QUEUING NEW MODEL")

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
            updated_entry = self.update_failure(new_entry_dict, failure_notes)
            return self.db_client.upsert_and_return(updated_entry, request.hash)

        logger.info("QUEUING: " + str(new_entry_dict))

        last_model = self.db_client.last_uploaded_model(request.hotkey)
        # Temp guard against rate limiting
        if last_model is not None and self.retries > 10:
            last_model_status = StatusEnum.from_string(last_model["status"])
            if last_model_status != StatusEnum.FAILED:
                last_block = last_model.get("block", request.block)
                current_block = request.block
                if abs(current_block - last_block) < BLOCK_RATE_LIMIT and abs(current_block - last_block) > 0:
                    failure_notes = f"""
                    Exceeded rate limit. 
                    Last submitted model was block {last_block} with details {last_model}.
                    Block difference between last block {last_block} and current block {current_block} is {(current_block - last_block)}
                    Current submission {current_block} exceeds minimum block limit {BLOCK_RATE_LIMIT}"""
                    logger.error(failure_notes)
                    new_entry_dict = self.update_failure(new_entry_dict, failure_notes)


        inserted = self.db_client.insert(new_entry_dict, request.hash)
        status = StatusEnum.QUEUED if inserted else StatusEnum.FAILED 
        return {"status":status}

    def health(self):
        """GET / - Health check endpoint for kubernetes"""
        return {"g": True, "k": False}

    def hc(self):
        """GET /hc - Health check endpoint"""
        return {"g": True, "k": False}

    def get_routes(self):
        """GET /routes - Endpoint for retrieving all available routes"""
        routes = []
        for route in self.router.routes:
            routes.append({
                "path": route.path,
                "methods": route.methods,
                "name": route.name
            })
        return routes

    def start(self, main_api_port: int = 8000):
        try:
            logger.info("Starting API server")
            uvicorn.run(self.app, host="0.0.0.0", port=main_api_port)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        except Exception as e:
            logger.error(f"An exception occurred: {e}")
        finally:
            logger.info("Stopping evaluation thread")

    def get_recent_entries(self):
        """GET /recent_entries - Endpoint for retrieving 256 most recent hash entries"""
        entries = self.db_client.fetch_recent_entries(limit=256)
        if entries is None:
            return []
        if len(entries) < 1:
            return []
        return entries

def start():
    import argparse
    parser = argparse.ArgumentParser(description="Run the server")
    parser.add_argument("--main-api-port", type=int, default=7777, help="Port for the main API")
    args = parser.parse_args()
    
    api = ValidationAPI()
    api.start(args.main_api_port)

if __name__ == "__main__":
    start()
