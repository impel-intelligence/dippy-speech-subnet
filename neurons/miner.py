import argparse
import logging
from typing import Optional, Type

import bittensor as bt
from pydantic import BaseModel, Field, PositiveInt

from common.data import ModelId
from utilities.validation_utils import regenerate_hash

DEFAULT_NETUID = 231

EXAMPLE_CLI_COMMAND = """python neurons/miner.py \
    --repo_namespace ipsilondev \
    --repo_name parler_tts \
    --config_template default \
    --netuid 231 \
    --subtensor.network test \
    --online True \
    --model_hash d1 \
    --wallet.name coldkey1 \
    --wallet.hotkey hotkey1 \
    --logging.debug True \
"""


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_namespace",
        default="DippyAI",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--repo_name",
        default="your-model-here",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    parser.add_argument(
        "--config_template",
        type=str,
        default="default",
        help="The default config template for the model.",
    )

    parser.add_argument(
        "--netuid",
        type=str,
        default=f"{DEFAULT_NETUID}",
        help="The subnet UID.",
    )
    parser.add_argument(
        "--online",
        type=bool,
        default=False,
        help="Toggle to make the commit call to the bittensor network",
    )
    parser.add_argument(
        "--model_hash",
        type=str,
        default="d1",
        help="Model hash of the submission (currently optional for now)",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)
    return config


def register():

    config = get_config()

    bt.logging(config=config)

    wallet = bt.wallet(config=config, path=config.wallet.path)
    subtensor = bt.subtensor(config=config)

    logger = logging.getLogger("bittensor")
    logger.setLevel(logging.INFO)

    hotkey = wallet.hotkey.ss58_address
    namespace = config.repo_namespace
    repo_name = config.repo_name
    config_template = config.config_template
    entry_hash = str(regenerate_hash(namespace, repo_name, config_template, hotkey))
    model_id = ModelId(
        namespace=namespace,
        name=repo_name,
        config_template=config_template,
        competition_id=config.competition_id,
        hotkey=hotkey,
        hash=entry_hash,
    )
    model_commit_str = model_id.to_compressed_str()

    bt.logging.info(f"Registering with the following data")
    bt.logging.info(f"Coldkey: {wallet.coldkey.ss58_address}")
    bt.logging.info(f"Hotkey: {hotkey}")
    bt.logging.info(f"repo_namespace: {namespace}")
    bt.logging.info(f"repo_name: {repo_name}")
    bt.logging.info(f"config_template: {config_template}")
    bt.logging.info(f"entry_hash: {entry_hash}")
    bt.logging.info(f"Full Model Details: {model_id}")
    bt.logging.info(f"Subtensor Network: {subtensor.network}")
    bt.logging.info(f"model_hash: {config.model_hash}")
    bt.logging.info(f"String to be committed: {model_commit_str}")

    try:
        netuid = int(config.netuid)
    except ValueError:
        netuid = DEFAULT_NETUID
    netuid = netuid or DEFAULT_NETUID
    if config.online:
        try:
            subtensor.commit(wallet, netuid, model_commit_str)
            bt.logging.info(f"Succesfully commited {model_commit_str} under {hotkey} on netuid {netuid}")
        except Exception as e:
            print(e)

    # Check if the commit was successfully stored by fetching metadata
    try:
        metadata = bt.core.extrinsics.serving.get_metadata(subtensor, netuid, hotkey)
        if not metadata or "info" not in metadata or "fields" not in metadata["info"]:
            raise RuntimeError(f"No valid metadata found for netuid {netuid}")

        bt.logging.info("Metadata successfully retrieved for validation.")
        print(metadata)

        # Extract and verify the commitment details from the metadata
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment.get(list(commitment.keys())[0], "")[2:]

        if not hex_data:
            raise ValueError("Commitment data is empty or malformed.")

        try:
            chain_str = bytes.fromhex(hex_data).decode()
            model_id = ModelId.from_compressed_str(chain_str)
            bt.logging.info("Model ID successfully reconstructed from commitment.")
        except (ValueError, UnicodeDecodeError) as decode_error:
            bt.logging.error(f"Decoding failed: {decode_error}")
            raise

        # Compare the reconstructed model ID with the original commitment string
        if model_id != ModelId.from_compressed_str(model_commit_str):
            raise RuntimeError("Reconstructed model ID does not match the original commitment.")

        # Construct metadata with the model ID and block number for validation
        metadata = {
            "model_id": model_id,
            "block": metadata["block"],
        }
        bt.logging.info(f"Validation successful. Model ID: {model_id}, Block: {metadata['block']}")
    except Exception as e:
        bt.logging.error(f"Verification failed: {e}")
        raise


if __name__ == "__main__":
    register()
