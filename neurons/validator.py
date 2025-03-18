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
NEW_EPOCH_BLOCK = 5163462 # Models submitted before this block will get a score of 0. -- Block # 5,163,462

SUBNET_REGISTERED_UID = 155
SUBNET_EMISSION_BURN_RATE = 0.75
def compute_wins(
    miner_registry: Dict[int, MinerEntry],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    This function iterates through each miner in the registry, comparing losses to determine the number of wins
    and then calculates the win rate for each miner.

    Parameters:
        miner_registry (Dict[int, MinerEntry]): A dictionary with miner UIDs as keys and MinerEntry objects as values.

    Returns:
        Tuple[Dict[int, int], Dict[int, float]]: A tuple containing two dictionaries:
            - The first dictionary maps miner IDs to their respective number of wins.
            - The second dictionary maps miner IDs to their win rate, calculated as the number of wins divided by the total comparisons.
    """
    uids = miner_registry.keys()
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = miner_registry[uid_i].block
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = miner_registry[uid_j].block
            score_i = miner_registry[uid_i].total_score
            score_j = miner_registry[uid_j].total_score
            if block_i < NEW_EPOCH_BLOCK:
                score_i = 0
            if block_j < NEW_EPOCH_BLOCK:
                score_j = 0

            wins[uid_i] += 1 if iswin(score_i, score_j, block_i, block_j) else 0
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

        if miner_registry[uid_i].invalid or miner_registry[uid_i].total_score == 0:
            win_rate[uid_i] = float("-inf")

    return wins, win_rate


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


    def build_commit_data(self) -> Dict[str, Any]:
        max_retries = 10
        base_delay = 1.5  # seconds
        commitments = {}
        raw_commmitments = None
        for attempt in range(max_retries):
            try:
                # First try using self.subtensor
                try:
                    raw_commmitments = self.subtensor.query_map(
                        module="Commitments",
                        name="CommitmentOf",
                        params=[self.config.netuid])
                except Exception as e:
                    bt.logging.warning(f"Failed to fetch metadata with self.subtensor: {e}, trying dedicated subtensor")
                    # Fall back to dedicated subtensor
                    dedicated_subtensor = None
                    try:
                        network = "finney"
                        dedicated_subtensor = Subtensor(network=network)
                        bt.logging.warning(f"Created dedicated subtensor for metadata fetch: {dedicated_subtensor} ")
                        raw_commmitments = dedicated_subtensor.query_map(
                        module="Commitments",
                        name="CommitmentOf",
                        params=[self.config.netuid])
                    finally:
                        # Ensure we close the dedicated subtensor
                        if dedicated_subtensor is not None:
                            try:
                                dedicated_subtensor.close()
                            except Exception as close_error:
                                bt.logging.error(f"Error closing dedicated subtensor: {close_error}")
            except Exception as e:
                delay = base_delay**attempt
                if attempt < max_retries - 1:  # Don't log "retrying" on the last attempt
                    bt.logging.error(f"Attempt {attempt + 1}/{max_retries} failed to fetch data : {e}")
                    bt.logging.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    bt.logging.error(f"All attempts failed to fetch data : {e}")
                    raise e

        if raw_commmitments is None:
            raise Exception("Failed to fetch raw commitments from chain")
        commitments = {}
        for key, value in raw_commmitments:
            try:
                hotkey = decode_account_id(key[0])
                body = cast(dict, value.value)
                chain_str = extract_raw_data(body)
                commitments[str(hotkey)] = {"block": body["block"], "chain_str": chain_str}
            except Exception as e:
                bt.logging.error(f"Failed to decode commitment for hotkey {hotkey}: {e}")
                continue

        return commitments

    async def build_registry(
        self, all_uids: List[int], current_block: int, max_concurrent: int = 32
    ) -> Tuple[int, MinerEntry]:
        miner_registry: Dict[int, MinerEntry] = {uid: MinerEntry() for uid in all_uids}
        commitments = self.build_commit_data()
        invalid_uids = []

        async def process_uid(uid):
            hotkey = self.metagraph.hotkeys[uid]
            miner_registry[uid].hotkey = hotkey
            bt.logging.debug(f"now checking for uid={uid} and hotkey {hotkey}")
            try:
                raw_miner_data = commitments[hotkey] if hotkey in commitments else None
                if raw_miner_data is None:
                    invalid_uids.append(uid)
                    bt.logging.error(f"skip uid={uid} no model_data")
                    return
                miner_model_id = ModelId.from_compressed_str(raw_miner_data["chain_str"])
                miner_block = raw_miner_data["block"]
                model_data = MinerEntry()
                model_data.block = miner_block
                model_data.miner_model_id = miner_model_id
                if model_data.miner_model_id is None:
                    invalid_uids.append(uid)
                    bt.logging.warning(f"skip uid={uid} no model_id available")
                    return
                # Skip model submitted after run step has begun
                if model_data.block > current_block:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip uid={uid} submitted on {model_data.block} after {current_block}")
                    return

                miner_registry[uid].block = model_data.block
                miner_registry[uid].miner_model_id = model_data.miner_model_id

                signed_payload = sign_request(
                    self.wallet.hotkey,
                    hotkey,
                )
                _score_data = _get_model_score(
                    namespace=model_data.miner_model_id.namespace,
                    name=model_data.miner_model_id.name,
                    hash=model_data.miner_model_id.hash,
                    template=model_data.miner_model_id.config_template,
                    hotkey=model_data.miner_model_id.hotkey,
                    config=self.config,
                    local_metadata=self.local_metadata,
                    signatures=signed_payload,
                )

                # if _score_data.status != StatusEnum.COMPLETED:
                #     _score_data = _get_model_score(
                #         miner_registry[uid].miner_model_id,
                #         self.config,
                #         self.local_metadata,
                #         signed_payload,
                #         True,
                #     )
                bt.logging.warning(
                    f"_score_data for uid={uid} on block {miner_registry[uid].block} : {miner_registry[uid].miner_model_id} {_score_data}"
                )

                if _score_data.status == StatusEnum.QUEUED or _score_data.status == StatusEnum.RUNNING:
                    invalid_uids.append(uid)
                    bt.logging.info(f"skip uid={uid} status is {_score_data.status}")
                    return

                if _score_data.status == StatusEnum.COMPLETED:
                    miner_registry[uid].total_score = _score_data.calculate_total_score()

                elif _score_data.status == StatusEnum.FAILED:
                    miner_registry[uid].total_score = 0

            except Exception as e:
                bt.logging.error(f"could not update for uid={uid}:{hotkey} {e}")
                bt.logging.error(f"Traceback: {traceback.format_exc()}")
                invalid_uids.append(uid)
                return

        # Process UIDs in batches of max_concurrent size
        for i in range(0, len(miner_registry), max_concurrent):
            batch_uids = list(miner_registry.keys())[i : i + max_concurrent]
            batch_tasks = [process_uid(uid) for uid in batch_uids]
            await asyncio.gather(*batch_tasks)

        return invalid_uids, miner_registry

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
            bt.logging.warning(f"Running Validator Version - DTAO - V0.0.2")
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

    def model_id_matches_hotkey(self, model_id: ModelId, hotkey: str) -> bool:
        original_hash = model_id.hash or ""
        hotkey_hash = regenerate_hash(
            namespace=model_id.namespace, name=model_id.name, chat_template=model_id.chat_template, hotkey=hotkey
        )
        hotkey_matches = str(original_hash) == str(hotkey_hash)

        return hotkey_matches

    def get_metadata_with_retry(self, hotkey: str):
        """
        Retrieves metadata for a given hotkey with retry logic.

        Args:
            hotkey (str): The hotkey identifier.

        Returns:
            Optional[MinerEntry]: The fetched MinerEntry or None if failed.
        """
        max_retries = 5
        backoff_multiplier = 10  # Base wait time in seconds
        backoff_cap = 60  # Maximum wait time between retries

        for attempt in range(1, max_retries + 1):
            try:
                bt.logging.warning(f"Attempt {attempt}: Trying to get model metadata for {hotkey}")
                result = bt.core.extrinsics.serving.get_metadata(
                    self=self.subtensor, netuid=self.config.netuid, hotkey=hotkey
                )

                bt.logging.warning(f"Results: {result}")
                return result  # Successful result, exit function
            except Exception as e:
                bt.logging.error(f"Error fetching metadata for hotkey {hotkey} (Attempt {attempt}): {e}")
                if attempt == max_retries:  # If final attempt, re-raise the exception
                    raise
                # Wait before retrying (exponential backoff with cap)
                backoff_time = min(backoff_multiplier * (2 ** (attempt - 1)), backoff_cap)
                bt.logging.warning(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)

        return None

    def fetch_model_data(self, uid: int, hotkey: str) -> Optional[MinerEntry]:
        max_retries = 10
        base_delay = 1.5  # seconds
        for attempt in range(max_retries):
            try:
                # First try using self.subtensor
                try:
                    metadata = bt.core.extrinsics.serving.get_metadata(
                        self=self.subtensor,
                        netuid=self.config.netuid,
                        hotkey=hotkey
                    )
                except Exception as e:
                    bt.logging.warning(f"Failed to fetch metadata with self.subtensor: {e}, trying dedicated subtensor")
                    # Fall back to dedicated subtensor
                    dedicated_subtensor = None
                    try:
                        network = "finney"
                        dedicated_subtensor = Subtensor(network=network)
                        bt.logging.warning(f"Created dedicated subtensor for metadata fetch: {dedicated_subtensor} for {uid}")
                        
                        metadata = bt.core.extrinsics.serving.get_metadata(
                            self=dedicated_subtensor,
                            netuid=self.config.netuid,
                            hotkey=hotkey
                        )
                    finally:
                        # Ensure we close the dedicated subtensor
                        if dedicated_subtensor is not None:
                            try:
                                dedicated_subtensor.close()
                            except Exception as close_error:
                                bt.logging.error(f"Error closing dedicated subtensor: {close_error} for {uid}")

                if metadata is None:
                    return None

                commitment = metadata["info"]["fields"][0]
                hex_data = commitment[list(commitment.keys())[0]][2:]
                chain_str = bytes.fromhex(hex_data).decode()

                model_id = ModelId.from_compressed_str(chain_str)
                model_id.hotkey = hotkey

                block = metadata["block"]
                entry = MinerEntry()
                entry.block = block
                entry.miner_model_id = model_id
                return entry

            except Exception as e:
                delay = base_delay ** attempt
                if attempt < max_retries - 1:  # Don't log "retrying" on the last attempt
                    bt.logging.error(f"Attempt {attempt + 1}/{max_retries} failed to fetch data for {hotkey}: {e}")
                    bt.logging.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    bt.logging.error(f"All attempts failed to fetch data for {hotkey}: {e}")
                    return None

        return None

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
    @staticmethod
    def adjusted_temperature_multipler(current_block: int) -> float:
        CHANGE_BLOCK = 4247000
        # currently force static 0.15 temperature
        if current_block > CHANGE_BLOCK:
            return 15
        diff = current_block - CHANGE_BLOCK
        # Map block difference to temperature value between 1-15
        # Scale linearly up to NEW_EPOCH_BLOCK
        if diff <= 7200:
            return 1.0

        # Linear scaling: (diff / max_diff) * (max_temp - min_temp) + min_temp
        temp = (diff / CHANGE_BLOCK) * 14 + 1

        # Cap at max temperature of 0.15
        return min(temp, 15)

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Iterate through blockchain state to find miner entries for models.
        2. Fetches model scoring data from separate evaluation instance.
        3. Applies elimination logic to better calulate model scoring.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Logs all relevant data for the step, including model IDs, scores, and win rates.
        """

        # Update self.metagraph
        synced = await self.try_sync_metagraph(ttl=60)
        if not synced:
            return False
        current_block = self.metagraph.block.item()
        competition_parameters = constants.COMPETITION_SCHEDULE[0]
        # telemetry_report(self.local_metadata)
        all_uids = self.metagraph.uids.tolist()
        # Avoid biasing lower value uids when making calls
        random.shuffle(all_uids)
        # Prepare evaluation
        bt.logging.warning(
            f"Computing metrics for {len(all_uids)} UIDS, for competition {competition_parameters.competition_id}"
        )

        invalid_uids, miner_registry = await self.build_registry(all_uids=all_uids, current_block=current_block)
        bt.logging.warning(f"invalid_uids : {invalid_uids}")

        try:
            for uid1, entry1 in miner_registry.items():
                if entry1.invalid or entry1.miner_model_id is None:
                    continue
                for uid2, entry2 in miner_registry.items():
                    if uid1 == uid2 or entry2.invalid or entry2.miner_model_id is None:
                        continue
                    entry1_repo_id = f"{entry1.miner_model_id.namespace}/{entry1.miner_model_id.name}"
                    entry2_repo_id = f"{entry2.miner_model_id.namespace}/{entry2.miner_model_id.name}"

                    hash_matches = entry1.miner_model_id.hash == entry2.miner_model_id.hash
                    repo_details_matches = entry1_repo_id == entry2_repo_id

                    # Check if the model hashes are the same
                    if hash_matches or repo_details_matches:
                        # If blocks are different, mark the one with greater block as invalid
                        if entry1.block > entry2.block:
                            invalid_uids.append(uid1)
                            bt.logging.warning(f"Marked uid={uid1} as invalid due to duplicate model with newer block")

                            break
                        elif entry2.block > entry1.block:
                            invalid_uids.append(uid2)
                            bt.logging.warning(f"Marked uid={uid2} as invalid due to duplicate model with newer block")
        except Exception as e:
            bt.logging.error(f"could not perform hash check {e}")

        bt.logging.warning(
            f"all_uids : {len(miner_registry)} invalid uids: {len(invalid_uids)} cutoff_block : {current_block}"
        )
        # Mark uids that do not have a proper score
        for uid in invalid_uids:
            if uid not in miner_registry:
                miner_registry[uid] = MinerEntry()
            miner_registry[uid].invalid = True
            miner_registry[uid].total_score = 0

        wins, win_rate = compute_wins(miner_registry)
        sorted_uids = sorted(miner_registry.keys())

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor([win_rate[uid] for uid in sorted_uids], dtype=torch.float32)


        target_uid = SUBNET_REGISTERED_UID
        target_weight_ratio = SUBNET_EMISSION_BURN_RATE
        
        try:
            target_idx = sorted_uids.index(target_uid)
            temperature = constants.temperature
            
            initial_weights = torch.softmax(model_weights / temperature, dim=0)
            
            # Scale down non-target weights to make room for target weight
            initial_weights = initial_weights * (1 - target_weight_ratio)
            
            initial_weights[target_idx] = target_weight_ratio
            
            step_weights = initial_weights
            
        except ValueError:
            # Leave original method in case of rollback
            temperature = constants.temperature
            step_weights = torch.softmax(model_weights / temperature, dim=0)

        step_weights = step_weights / step_weights.sum()

        # Update weights based on moving average.
        torch_metagraph = torch.from_numpy(self.metagraph.S)
        self.weights = torch.zeros_like(torch_metagraph)
        new_weights = torch.zeros_like(torch_metagraph)
        for i, uid_i in enumerate(sorted_uids):
            new_weights[uid_i] = step_weights[i]
        new_weights *= 1 / new_weights.sum()
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

        # Log to screen.
        self.log_step(
            miner_registry,
            wins,
            win_rate,
        )
        return True

    def log_step(
        self,
        miner_registry: Dict[int, MinerEntry],
        wins,
        win_rate,
    ):
        sorted_uids = sorted(miner_registry.keys())
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "uids": sorted_uids,
            "uid_data": {},
            "step": self.epoch_step,
        }
        for i, uid in enumerate(sorted_uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": miner_registry[uid].block,
                "score": miner_registry[uid].total_score,
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("score", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in sorted_uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["score"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.debug(f"Step results: {step_log}")
        scores_per_uid = {}
        for uid in sorted_uids:
            scores_per_uid[uid] = miner_registry[uid].total_score
        self._event_log("log_scores", scores=scores_per_uid, step=self.epoch_step)

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


def _get_model_score(
    namespace: str,
    name: str,
    hash: str,
    template: str,
    hotkey: str,
    config,
    local_metadata: LocalMetadata,
    signatures: Dict[str, str],
    retryWithRemote: bool = False,
    debug: bool = False,
) -> Scores:
    # Status:
    # QUEUED, RUNNING, FAILED, COMPLETED
    # return (score, status)
    if config.use_local_validation_api and not retryWithRemote:
        validation_endpoint = f"http://localhost:{config.local_validation_api_port}/model_submission_details"
    else:
        validation_endpoint = f"{constants.VALIDATION_SERVER}/model_submission_details"

    # Construct the payload with the model name and chat template type
    payload = {
        "repo_namespace": namespace,
        "repo_name": name,
        "hash": hash,
        "chat_template_type": template,
        "hotkey": hotkey,
    }
    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }
    headers.update(signatures)
    if os.environ.get("ADMIN_KEY", None) not in [None, ""]:
        payload["admin_key"] = os.environ["ADMIN_KEY"]

    score_data = Scores()

    try:
        response = requests.get(validation_endpoint, params=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response JSON
        result = response.json()
        if debug:
            console = Console()
            console.print(f"Payload: {payload}")

        if result is None or "status" not in result:
            score_data.status = StatusEnum.FAILED
            return score_data

        status = StatusEnum.from_string(result["status"])
        score_data.status = status

        if "score" in result:
            score_data.from_response(result["score"])

    except Exception as e:
        score_data.status = StatusEnum.FAILED
        bt.logging.error(e)
        bt.logging.error(f"Failed to get score and status for {namespace}/{name}")

    bt.logging.debug(f"Model {namespace}/{name} has score data {score_data}")
    return score_data




if __name__ == "__main__":
    metadata = local_metadata()
    asyncio.run(Validator(metadata).run())
