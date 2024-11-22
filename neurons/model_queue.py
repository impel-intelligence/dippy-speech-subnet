# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
from dataclasses import dataclass
import datetime as dt
import time
import argparse
import requests
import aiohttp
from asyncio import Semaphore


import math
import typing
import constants
import traceback
import bittensor as bt

from typing import Dict, Optional

from common.scores import StatusEnum, Scores
from common.model_id import ModelId
from common.validation_utils import LocalMetadata
import os

from common.event_logger import EventLogger
from bittensor.core.subtensor import Subtensor
from bittensor.core.metagraph import Metagraph
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

l = LocalMetadata(commit="x", btversion="x")


def push_minerboard(
    hash: str,
    uid: int,
    hotkey: str,
    block: int,
    config,
    local_metadata: LocalMetadata,
    retryWithRemote: bool = False,
) -> None:
    if config.use_local_validation_api and not retryWithRemote:
        validation_endpoint = f"http://localhost:{config.local_validation_api_port}/minerboard_update"
    else:
        validation_endpoint = f"{constants.VALIDATION_SERVER}/minerboard_update"

    # Construct the payload with the model name and chat template type
    payload = {
        "hash": hash,
        "uid": uid,
        "hotkey": hotkey,
        "block": block,
    }

    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }
    if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
        payload["admin_key"] = os.environ["ADMIN_KEY"]

    # Make the POST request to the validation endpoint
    try:
        response = requests.post(validation_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except Exception as e:
        print(e)

@dataclass
class MinerInfo:
    hotkey: str
    metadata: Optional[Dict] = None

class ModelQueue:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--immediate",
            action="store_true",
            help="Triggers run step immediately. NOT RECOMMENDED FOR PRODUCTION",
        )
        parser.add_argument("--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID.")
        parser.add_argument(
            "--use-local-validation-api",
            action="store_true",
            help="Use a local validation api",
        )
        parser.add_argument(
            "--local-validation-api-port",
            type=int,
            default=8000,
            help="Port for local validation api",
        )

        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self):
        config = ModelQueue.config()
        
        self.config = config
        self.netuid = self.config.netuid or 231
        # network = self.config.subtensor.network or "test"
        network = "test"
        self.subtensor = Subtensor(network)
        
        self.metagraph = Metagraph(netuid=self.netuid, network=network, lite=True, sync=True)
        self.metagraph.sync(subtensor=self.subtensor)
        
        logfilepath = "/tmp/modelq/{time:UNIX}.log"
        self.logger = EventLogger(
            filepath=logfilepath,
            level="INFO",
            stderr=True,
        )
        self.logger.info(f"Starting model queue with config: {self.config}")

    # Every x minutes
    def forever(self):
        while True:
            now = dt.datetime.now()
            # Calculate the next 5 minute mark
            minutes_until_next_epoch = 5 - (now.minute % 5)
            next_epoch_minute_mark = now + dt.timedelta(minutes=minutes_until_next_epoch)
            next_epoch_minute_mark = next_epoch_minute_mark.replace(second=0, microsecond=0)
            sleep_time = (next_epoch_minute_mark - now).total_seconds()
            self.logger.info(f"sleeping for {sleep_time}")
            if not self.config.immediate:
                time.sleep(sleep_time)
            
            try:
                asyncio.run(self.load_latest_metagraph())
            except Exception as e:
                self.logger.error(f"failed to queue {e}")

    async def check_uid(self, uid: int, miner_info_map: typing.Dict[int, MinerInfo], semaphore: Semaphore) -> StatusEnum:
        async with semaphore:  # Limit concurrent executions
            try:
                miner_info = miner_info_map[uid]
                hotkey = miner_info.hotkey
                metadata = miner_info.metadata
                if metadata is None or "model_id" not in metadata:
                    self.logger.info(f"NO_METADATA : uid: {uid} hotkey : {hotkey}")
                    return StatusEnum.NO_METADATA
                    
                model_id = metadata["model_id"]
                block = metadata["block"]
                result = await self.ensure_model(
                    namespace=model_id.namespace,
                    name=model_id.name,
                    hash=model_id.hash,
                    template=model_id.config_template,
                    block=block,
                    hotkey=hotkey,
                    config=self.config,
                    retryWithRemote=True,
                )
                stats = f"{result.status} : uid: {uid} hotkey : {hotkey} block: {block} model_metadata : {model_id}"
                self.logger.info(stats)
                
                await self.push_minerboard(
                    hash=model_id.hash,
                    uid=uid,
                    hotkey=hotkey,
                    block=block,
                    local_metadata=l,
                    config=self.config,
                    retryWithRemote=True,
                )
                
                return result.status
                
            except Exception as e:
                self.logger.error(f"exception for uid {uid} : {e}")
                return StatusEnum.ERROR

    async def load_latest_metagraph(self, max_tasks: int = 32):
        self.logger.info(f"fetching metagraph for netuid={self.netuid}")
        # latest_metagraph = self.subtensor.metagraph(self.netuid)
        self.metagraph.sync(subtensor=self.subtensor)
        latest_metagraph = self.metagraph
        all_uids = latest_metagraph.uids.tolist()
        if len(all_uids) < 1:
            self.logger.info("empty metagraph")
            return
        hotkeys = latest_metagraph.hotkeys
        
        # Create mapping of uid -> MinerInfo
        miner_info_map: Dict[int, MinerInfo] = {}
        for uid in all_uids:
            hotkey = hotkeys[uid]
            try:
                metadata = bt.core.extrinsics.serving.get_metadata(self.subtensor, self.netuid, hotkey)
                if not metadata:
                    raise RuntimeError(f"no metadata exists for {uid}") 
                
                commitment = metadata["info"]["fields"][0]
                hex_data = commitment[list(commitment.keys())[0]][2:]
                chain_str = bytes.fromhex(hex_data).decode()
                model_id = ModelId.from_compressed_str(chain_str)
                metadata = {
                    "model_id": model_id,
                    "block": metadata["block"],
                }
                miner_info_map[uid] = MinerInfo(hotkey=hotkey, metadata=metadata)
            except Exception as e:
                miner_info_map[uid] = MinerInfo(hotkey=hotkey, metadata=None)
                print(f"could not fetch data for uid {uid}")

        queued = failed = no_metadata = completed = error = running = precheck = 0
        
        # Create semaphore to limit concurrent tasks
        semaphore = Semaphore(max_tasks)
        
        # Create tasks with semaphore
        tasks = [self.check_uid(uid, miner_info_map, semaphore) for uid in all_uids]
        results = await asyncio.gather(*tasks)
            
        for status in results:
            if status == StatusEnum.QUEUED:
                queued += 1
            elif status == StatusEnum.FAILED:
                failed += 1 
            elif status == StatusEnum.NO_METADATA:
                no_metadata += 1
            elif status == StatusEnum.COMPLETED:
                completed += 1
            elif status == StatusEnum.ERROR:
                error += 1
            elif status == StatusEnum.RUNNING:
                running += 1
            elif status == StatusEnum.PRECHECK:
                precheck += 1
            
        self.logger.info(
            f"Status counts:\n"
            f"NO_METADATA: {no_metadata}\n"
            f"QUEUED: {queued}\n" 
            f"FAILED: {failed}\n"
            f"COMPLETED: {completed}\n"
            f"ERROR: {error}\n"
            f"RUNNING: {running}\n"
            f"PRECHECK: {precheck}\n"
            f"TOTAL: {len(all_uids)}"
        )

    async def ensure_model(
        self,
        namespace,
        name,
        hash,
        template,
        block,
        hotkey,
        config,
        retryWithRemote: bool = False,
    ) -> Scores:
        if config.use_local_validation_api and not retryWithRemote:
            validation_endpoint = f"http://localhost:{config.local_validation_api_port}/ensure_model"
        else:
            validation_endpoint = f"{constants.VALIDATION_SERVER}/ensure_model"

        payload = {
            "repo_namespace": namespace,
            "repo_name": name,
            "hash": hash,
            "chat_template_type": template,
            "block": block,
            "hotkey": hotkey,
        }
        score_data = Scores()

        headers = {}

        if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
            headers["admin-key"] = os.environ["ADMIN_KEY"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(validation_endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if result is None:
                        raise RuntimeError(f"no leaderboard entry exists at this time for {payload}")
                    if "status" in result:
                        status = StatusEnum.from_string(result["status"])
                        score_data.status = status

        except Exception as e:
            self.logger.error(e)
            self.logger.error(f"Failed to get score and status for {namespace}/{name}")
            score_data.status = StatusEnum.FAILED
        return score_data

    async def push_minerboard(
        self,
        hash: str,
        uid: int,
        hotkey: str,
        block: int,
        config,
        local_metadata: LocalMetadata,
        retryWithRemote: bool = False,
    ) -> None:
        if config.use_local_validation_api and not retryWithRemote:
            validation_endpoint = f"http://localhost:{config.local_validation_api_port}/minerboard_update"
        else:
            validation_endpoint = f"{constants.VALIDATION_SERVER}/minerboard_update"

        payload = {
            "hash": hash,
            "uid": uid,
            "hotkey": hotkey,
            "block": block,
        }

        headers = {
            "Git-Commit": str(local_metadata.commit),
            "Bittensor-Version": str(local_metadata.btversion),
            "UID": str(local_metadata.uid),
            "Hotkey": str(local_metadata.hotkey),
            "Coldkey": str(local_metadata.coldkey),
        }
        if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
            headers["admin-key"] = os.environ["ADMIN_KEY"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(validation_endpoint, json=payload, headers=headers) as response:
                    response.raise_for_status()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    q = ModelQueue()
    q.forever()