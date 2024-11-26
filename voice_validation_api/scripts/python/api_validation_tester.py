import argparse
import base64
import datetime as dt
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version
from shlex import split
from typing import Dict

import bittensor as bt
import requests
from bittensor.core.subtensor import Subtensor
from rich.console import Console

from common.scores import Scores, StatusEnum

"""
uv run python api_validation_tester.py     --device cuda     --blocks_per_epoch 100     --dont_set_weights     --wait_for_inclusion     --offline     --immediate     --netuid 231     --genesis     --dtype bfloat16     --do_sample     --num_samples_per_eval 64     --use-local-validation-api     --local-validation-api-port 8000     --wandb-key WANDBKEY     --wallet.name coldkey3     --wallet.hotkey hotkey3     --subtensor.network test     --wallet.path "/home/ubuntu/.bittensor/wallets"
"""


# Define constants
class constants:
    VALIDATION_SERVER = "https://example.com"  # Replace with the actual server if needed
    SUBNET_UID = 231


# Define LocalMetadata class
@dataclass
class LocalMetadata:
    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""


# Define sign_request function
def sign_request(keypair, payload: str):
    # For testing, return a dummy signature
    signed_payload = keypair.sign(data=payload)
    signed_payload_base64 = base64.b64encode(signed_payload).decode("utf-8")
    return {
        "payload_signed": signed_payload_base64,
        "payload": payload,
    }


# Define Config class
@staticmethod
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
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
        "--dtype",
        type=str,
        default="bfloat16",
        help="datatype to load model in, either bfloat16 or float16",
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
    parser.add_argument(
        "--wandb-key",
        type=str,
        default="",
        help="A WandB API key for logging purposes",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config


# Function to get local metadata
def local_metadata() -> LocalMetadata:
    """Extract the version as current git commit hash"""
    commit_hash = ""
    try:
        result = subprocess.run(
            split("git rev-parse HEAD"),
            check=True,
            capture_output=True,
            cwd=os.getcwd(),
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


# The _get_model_score function
import os
from typing import Dict

import requests

# Ensure you have the necessary imports for Scores, StatusEnum, LocalMetadata, constants, bt, and Console
# from your_project import Scores, StatusEnum, LocalMetadata, constants, bt
# from rich.console import Console


def get_model_score(
    namespace: str,
    name: str,
    hash: str,
    template: str,
    hotkey: str,
    config,
    local_metadata: "LocalMetadata",
    signatures: Dict[str, str],
    retryWithRemote: bool = False,
    debug: bool = False,
) -> "Scores":
    """
    Retrieves the model score by making an HTTP POST request to the validation endpoint.

    Args:
        namespace (str): Namespace of the repository.
        name (str): Name of the repository.
        hash (str): Commit hash.
        template (str): Chat template type.
        hotkey (str): Hotkey identifier.
        config: Configuration object containing settings.
        local_metadata (LocalMetadata): Metadata about the local environment.
        signatures (Dict[str, str]): Signature headers.
        retryWithRemote (bool, optional): Whether to retry with the remote API. Defaults to False.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Returns:
        Scores: An object containing the score and status.
    """
    # Determine the validation endpoint
    if config.use_local_validation_api and not retryWithRemote:
        validation_endpoint = f"http://localhost:{config.local_validation_api_port}/evaluate_model"
    else:
        validation_endpoint = f"{constants.VALIDATION_SERVER}/evaluate_model"

    # Construct the payload with the model details
    payload = {
        "repo_namespace": namespace,
        "repo_name": name,
        "hash": hash,
        "chat_template_type": template,
        "hotkey": hotkey,
    }

    # Construct the headers with metadata and signatures
    headers = {
        "Git-Commit": str(local_metadata.commit),
        "Bittensor-Version": str(local_metadata.btversion),
        "UID": str(local_metadata.uid),
        "Hotkey": str(local_metadata.hotkey),
        "Coldkey": str(local_metadata.coldkey),
    }
    headers.update(signatures)

    # Include admin key if available
    admin_key = os.environ.get("ADMIN_KEY")
    if admin_key:
        payload["admin_key"] = admin_key

    score_data = Scores()

    try:
        # Make the actual HTTP POST request
        response = requests.post(validation_endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response JSON
        result = response.json()

        if debug:
            console = Console()
            console.print(f"Payload: {payload}")

        if not result or "status" not in result:
            score_data.status = StatusEnum.FAILED
            return score_data

        # Update status
        status = StatusEnum.from_string(result["status"])
        score_data.status = status

        # Update scores if available
        if "score" in result:
            score_data.from_response(result["score"])

    except Exception as e:
        score_data.status = StatusEnum.FAILED
        bt.logging.error(e)
        bt.logging.error(f"Failed to get score and status for {namespace}/{name}")

    bt.logging.debug(f"Model {namespace}/{name} has score data {score_data}")
    return score_data


# Main function to test _get_model_score
if __name__ == "__main__":
    config = config()

    network_name = config.subtensor["network"] or "finney"
    netuid = config.netuid or 11

    wallet = bt.wallet(config=config)
    subtensor = Subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid, lite=False)

    validator_uid = 0
    if not config.offline:
        validator_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    hotkey = metagraph.hotkeys[validator_uid]

    local_metadata = local_metadata()

    local_metadata = LocalMetadata(
        commit=local_metadata.commit,
        btversion=local_metadata.btversion,
        hotkey=wallet.hotkey.ss58_address,
        coldkey=wallet.coldkeypub.ss58_address,
        uid=validator_uid,
    )

    signed_payload = sign_request(
        wallet.hotkey,
        hotkey,
    )

    score_data = get_model_score(
        namespace="test5050",
        name="test123",
        hash="123344",
        template="",
        hotkey=hotkey,
        config=config,
        local_metadata=local_metadata,
        signatures=signed_payload,
    )

    bt.logging.error(f"API validation tester script completed. {score_data}")
