# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const
import argparse
import asyncio
import copy
import datetime as dt
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from shlex import split
from typing import Dict, List, Optional,Any, Tuple, cast

import bittensor as bt
import numpy as np
import requests
import torch
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from huggingface_hub import get_safetensors_metadata
from rich.console import Console
from rich.table import Table
from scipy import optimize
from threadpoolctl import threadpool_limits

import constants
from common.data import ModelId, ModelMetadata
from common.scores import Scores, StatusEnum
from utilities import utils
from utilities.compete import iswin
from utilities.event_logger import EventLogger
from utilities.miner_registry import MinerEntry
from utilities.validation_utils import regenerate_hash
from importlib.metadata import version as pkg_version

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

from bittensor.core.chain_data import (
    decode_account_id,
)

def extract_raw_data(data):
    try:
        # Navigate to the fields tuple
        fields = data.get('info', {}).get('fields', ())
        
        # The first element should be a tuple containing a dictionary
        if fields and isinstance(fields[0], tuple) and isinstance(fields[0][0], dict):
            # Find the 'Raw' key in the dictionary
            raw_dict = fields[0][0]
            raw_key = next((k for k in raw_dict.keys() if k.startswith('Raw')), None)
            
            if raw_key and raw_dict[raw_key]:
                # Extract the inner tuple of integers
                numbers = raw_dict[raw_key][0]
                # Convert to string
                result = ''.join(chr(x) for x in numbers)
                return result
                
    except (IndexError, AttributeError):
        pass
    
    return None

os.environ["TOKENIZERS_PARALLELISM"] = "false"
INVALID_BLOCK_START = 4200000
INVALID_BLOCK_END = 4200000
NEW_EPOCH_BLOCK = 5623610 # Models submitted before this block will get weights set to zero after block # 5,170,108

SUBNET_REGISTERED_UID = 155
SUBNET_EMISSION_BURN_RATE = 1


@dataclass
class LocalMetadata:
    """Metadata associated with the local validator instance"""

    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""


def local_metadata() -> LocalMetadata:
    """Extract the version as current git commit hash"""
    commit_hash = ""
    try:
        result = subprocess.run(
            split("git rev-parse HEAD"),
            check=True,
            capture_output=True,
            cwd=constants.ROOT_DIR,
        )
        commit = result.stdout.decode().strip()
        assert len(commit) == 40, f"Invalid commit hash: {commit}"
        commit_hash = commit[:8]
    except Exception as e:
        commit_hash = "unknown"

    bittensor_version = version("bittensor")
    return LocalMetadata(
        commit=commit_hash,
        btversion=bittensor_version,
    )


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=100,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--wait_for_inclusion",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--immediate",
            action="store_true",
            help="Triggers run step immediately. NOT RECOMMENDED FOR PRODUCTION",
        )
        parser.add_argument("--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID.")
        parser.add_argument(
            "--genesis",
            action="store_true",
            help="Don't sync to consensus, rather start evaluation from scratch",
        )
        parser.add_argument(
            "--do_sample",
            action="store_true",
            help="Sample a response from each model (for leaderboard)",
        )
        parser.add_argument(
            "--num_samples_per_eval",
            type=int,
            default=64,
            help="Number of samples to evaluate per UID",
        )
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
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self, local_metadata: LocalMetadata):
        self.config = Validator.config()
        bt.logging(config=self.config)

        bt_version = pkg_version("bittensor")
        bt.logging.warning(f"Starting validator with config: {self.config}, bittensor version: {bt_version}")

        network_name = self.config.subtensor.network or "finney"
        try:
            netuid = int(self.config.netuid)
        except (ValueError, TypeError):
            netuid = 58
        netuid = netuid or 58
        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)

        self.subtensor = Subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=netuid, lite=False)

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # === Running args ===
        torch_metagraph = torch.from_numpy(self.metagraph.S)

        self.weights = torch.zeros_like(torch_metagraph)
        self.alt_weights = torch.zeros_like(torch_metagraph)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        # Sync to consensus
        if not self.config.genesis:
            torch_consensus = torch.from_numpy(self.metagraph.C)
            self.weights.copy_(torch_consensus)

        validator_uid = 0
        if not self.config.offline:
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Set up local metadata for stats collection
        self.local_metadata = LocalMetadata(
            commit=local_metadata.commit,
            btversion=local_metadata.btversion,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            uid=validator_uid,
        )
        bt.logging.warning(f"dumping localmetadata: {self.local_metadata}")


        # eventlog_path = "/tmp/sn11_event_logs/event_{time}.log"
        eventlog_path = "/dev/null"
        self.use_event_logger = False
        if os.getenv("SN11_LOG_PATH") is not None:
            eventlog_path = os.getenv("SN11_LOG_PATH")
        try:
            self.event_logger = EventLogger(filepath=eventlog_path)
            self.use_event_logger = True
        except Exception as e:
            bt.logging.error(
                f"Could not initialize event logger: {e}. Event logging is optional and used for diagnostic purposes. If you do not know what this is for, that's ok."
            )

        # == Initialize the update thread ==
        self.stop_event = threading.Event()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            
        try:
            self.subtensor.close()
            bt.logging.warning("Successfully closed subtensor.")
        except Exception as e:
            bt.logging.error(f"Failed to close subtensor: {e}", exc_info=True)

    def _event_log(self, msg: str, **kwargs):
        try:
            if self.use_event_logger:
                self.event_logger.info(msg, **kwargs)
        except Exception as e:
            bt.logging.error(e)

        return

    def _with_decoration(self, metadata: LocalMetadata, keypair, payload):
        signature = sign_request(
            keypair,
            payload=metadata.hotkey,
        )
        combined_payload = {
            "signature": signature,
            "payload": payload,
            "commit": str(metadata.commit),
            "btversion": str(metadata.btversion),
            "uid": str(metadata.uid),
            "hotkey": str(metadata.hotkey),
            "coldkey": str(metadata.coldkey),
        }
        return combined_payload

    def _remote_log(self, payload):
        event_report_endpoint = f"{constants.VALIDATION_SERVER}/event_report"
        try:
            response = requests.post(event_report_endpoint, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            bt.logging.warning(f"successfully sent event_report with payload {payload}")
        except Exception as e:
            bt.logging.error(f"could not remote log: {e}. This error is ok to ignore if you are a validator")

    async def set_weights_with_wait(self, weights, netuid, wallet, uids):
        retries = 5
        backoff = 1.5
        msg = None
        success = False
        for attempt in range(retries):
            try:
                success, msg = self.subtensor.set_weights(
                        netuid=netuid,
                        wallet=wallet,
                        uids=uids,
                        weights=weights,
                        wait_for_inclusion=False,
                        wait_for_finalization=True,
                        version_key=constants.weights_version_key,
                )
                if msg:
                    bt.logging.error(
                        f"called set_weights and received msg {msg}"
                    )
                if success:
                    return True
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                wait_time = backoff**attempt

                bt.logging.error(
                    f"Failed to set weights {msg} (attempt {attempt+1}/{retries}). Retrying in {wait_time:.1f}s..."
                )
                self.close_subtensor()
                self.subtensor = Validator.new_subtensor()
                time.sleep(wait_time)
        return False

    async def _try_set_weights(self, debug: bool = False) -> Tuple[bool, Optional[str]]:
        weights_success = False
        error_str = None
        try:
            cpu_weights = self.weights
            adjusted_weights = cpu_weights
            self.weights.nan_to_num(0.0)
            try:
                netuid = int(self.config.netuid)
                weights_success = await asyncio.wait_for(
                    self.set_weights_with_wait(
                        weights=adjusted_weights,
                        netuid=netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                    ),
                    timeout=600  # 10 minutes
                )
            except asyncio.TimeoutError:
                bt.logging.error("Setting weights timed out after 10 minutes")
                weights_success = False

            weights_report = {"weights": {}}
            for uid, score in enumerate(self.weights):
                weights_report["weights"][uid] = score
            self._event_log("set_weights_complete", weights=weights_report)
            bt.logging.warning(f"successfully_set_weights")
            weights_success = True
        except Exception as e:
            bt.logging.error(f"failed_set_weights error={e}\n{traceback.format_exc()}")
            error_str = f"failed_set_weights error={e}\n{traceback.format_exc()}"
            return weights_success, error_str

        # Log weight state
        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="All Weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Weight setting status
        status_table = Table(title="Weight Setting Status")
        status_table.add_column("Status", style="cyan")
        status_table.add_column("Value", style="magenta")
        status_table.add_row("successfully_set_weights", str(weights_success))
        weights_failed = not weights_success
        status_table.add_row("failed_set_weights", str(weights_failed))
        console.print(status_table)
        return weights_success, error_str

    async def try_set_weights(self, ttl: int) -> Tuple[bool, Optional[str]]:
        if self.config.offline:
            return False, None

        weights_set_success = False
        error_msg = None
        exception_msg = None
        try:
            bt.logging.debug("Setting weights.")
            weights_set_success, error_msg = await asyncio.wait_for(self._try_set_weights(), ttl)
            bt.logging.debug("Finished setting weights.")
        except asyncio.TimeoutError:
            error_msg = f"Failed to set weights after {ttl} seconds"
            bt.logging.error(error_msg)
        except Exception as e:
            exception_msg = f"Error setting weights: {e}\n{traceback.format_exc()}"
            bt.logging.error(exception_msg)
        finally:
            payload = {
                "time": str(dt.datetime.now(dt.timezone.utc)),
                "weights_set_success": weights_set_success,
                "error": error_msg,
                "exception_msg": exception_msg,
                "weights_version": constants.weights_version_key,
            }   
            logged_payload = self._with_decoration(self.local_metadata, self.wallet.hotkey, payload)
            self._remote_log(logged_payload)
        return weights_set_success, error_msg


    async def try_sync_metagraph(self, ttl: int = 120) -> bool:
        try:
            bt.logging.warning(f"attempting sync with network {self.subtensor.network}")
            self.metagraph = Metagraph(netuid=int(self.config.netuid), subtensor=self.subtensor, lite=False, sync=True, )
            return True
        except Exception as e:
            metagraph_failure_payload = {
                    "initial_metagraph_sync_success": False,
                    "failure_str": str(e),
                    "stacktrace": traceback.format_exc(),
                    "network":self.subtensor.network,
                }
            logged_payload = self._with_decoration(
                    self.local_metadata, self.wallet.hotkey, payload=metagraph_failure_payload
                )
            self._remote_log(logged_payload)
            bt.logging.error(f"could not sync metagraph {e} using network {self.subtensor.network}. Starting retries. If this issue persists please restart the validator script")
            self.close_subtensor()
            self.subtensor = Validator.new_subtensor()

        def sync_metagraph(attempt):
            try:
                self.metagraph.sync(block=None, lite=False, subtensor=self.subtensor)
            except Exception as e:
                bt.logging.error(f"{e}")
                # Log failure to sync metagraph
                metagraph_failure_payload = {
                    "metagraph_sync_success": False,
                    "failure_str": str(e),
                    "attempt": attempt,
                    "stacktrace": traceback.format_exc(),
                }
                logged_payload = self._with_decoration(
                    self.local_metadata, self.wallet.hotkey, payload=metagraph_failure_payload
                )
                self._remote_log(logged_payload)
                self.close_subtensor()
                self.subtensor = Validator.new_subtensor()
                raise e

        for attempt in range(3):
            try:
                sync_metagraph(attempt)
                return True
            except Exception as e:
                bt.logging.error(f"could not sync metagraph {e}")
                if attempt == 2:
                    return False

        bt.logging.success("Synced metagraph")
        self._event_log("metagraph_sync_success")
        return True

    async def try_run_step(self, ttl: int) -> Optional[bool]:
        async def _try_run_step():
            success = await self.run_step()
            logged_payload = self._with_decoration(
                self.local_metadata, self.wallet.hotkey, {"run_step_success": success}
            )
            self._remote_log(logged_payload)
            return success

        try:
            bt.logging.warning(f"Running Validator Version - DTAO V0.0.9")
            bt.logging.warning(f"Running step with ttl {ttl}")
            step_success = await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.warning("Finished running step.")
            return step_success
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")
            return False
        except Exception as e:
            bt.logging.error(f"Failed to run step : {e} {traceback.format_exc()}")
            return False

    @staticmethod
    def new_subtensor():
        network = random.choice(["finney", "subvortex"])
        subtensor = Subtensor(network=network)
        bt.logging.warning(f"subtensor retry initialized with Subtensor(): {subtensor}")
        return subtensor

    def close_subtensor(self):
        status = ""
        try:
            self.subtensor.close()
            status = "subtensor_closed"
        except Exception as e:
            status = f"{str(e)}\n{traceback.format_exc()}"
        payload = {"subtensor_close_status": status}
        logged_payload = self._with_decoration(
                    self.local_metadata, self.wallet.hotkey, payload=payload
                )
        self._remote_log(logged_payload)


    async def run_step(self):
        if hasattr(self, 'metagraph') and self.metagraph is not None:
            torch_metagraph = torch.from_numpy(self.metagraph.S)
        else:
            torch_metagraph = torch.zeros(256)
        
        new_weights = torch.zeros_like(torch_metagraph)
        
        # Set all weight to the subnet registered UID
        if SUBNET_REGISTERED_UID < len(new_weights):
            new_weights[SUBNET_REGISTERED_UID] = 1.0
            bt.logging.info(f"Successfully allocated all weights to UID {SUBNET_REGISTERED_UID}")
        else:
            bt.logging.error(f"Target UID {SUBNET_REGISTERED_UID} is out of range for weight tensor of size {len(new_weights)}")
            return False
        
        # Update weights directly without moving average since we're always setting the same value
        self.weights = new_weights
        self.weights = self.weights.nan_to_num(0.0)
        
        return True

    async def run(self):
        try:
            current_time = dt.datetime.now(dt.timezone.utc)
            bt.logging.success(f"Running step at {current_time.strftime('%H:%M')}")

            success = await self.try_run_step(ttl=60 * 40)
            weights_set_success = False
            self.global_step += 1

            if success:
                weights_set_success, error_msg = await self.try_set_weights(ttl=120)
                bt.logging.warning(f"weights_set_success {weights_set_success} error_msg {error_msg}")

            metagraph_synced = await self.try_sync_metagraph(ttl=120)
            bt.logging.warning(f"metagraph_synced {metagraph_synced}")

        except KeyboardInterrupt:
            bt.logging.warning("KeyboardInterrupt caught")
            exit()
        except Exception as e:
            bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")
            # Add a small delay before retrying in case of continuous errors
            await asyncio.sleep(5)


def telemetry_report(local_metadata: LocalMetadata, payload=None):
    telemetry_endpoint = f"{constants.VALIDATION_SERVER}/telemetry_report"

    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }

    # Make the POST request to the validation endpoint
    try:
        response = requests.post(telemetry_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except Exception as e:
        bt.logging.error(e)
    return


import base64


def sign_request(keypair, payload: str):

    signed_payload = keypair.sign(data=payload)
    signed_payload_base64 = base64.b64encode(signed_payload).decode("utf-8")

    return {
        "payload_signed": signed_payload_base64,
        "payload": payload,
    }



if __name__ == "__main__":
    metadata = local_metadata()
    asyncio.run(Validator(metadata).run())
