import gc
import json
import sys
import logging
import traceback
from typing import Optional

import torch
import typer

from scoring.common import EvaluateModelRequest
from scoring.logging_setup import setup_logging

app = typer.Typer()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def write_to_json(data: dict, filepath: str = "/tmp/output.json"):
    logger.info(f"Writing results to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Final Score: {data}")
    typer.echo(f" Final Score {data}")
    typer.echo(f"Results written to {filepath}")


def _run(
    request: EvaluateModelRequest,
    run_type: str,
):
    from scoring.get_eval_score import get_eval_score
    from scoring.get_tts_score import get_tts_score

    logger.info(f"Starting {run_type} evaluation")
    logger.debug(f"Request parameters: {request}")
    typer.echo(f"Evaluating with parameters: {request}")
    
    result = {"completed": False}
    try:
        if run_type == "eval":
            logger.info("Running evaluation score calculation")
            result = get_eval_score(request)
        if run_type == "tts":
            logger.info("Running TTS score calculation")
            result = get_tts_score(request)
        result["completed"] = True
        logger.info("Evaluation completed successfully")
        logger.debug(f"Evaluation result: {result}")
        typer.echo(f"Evaluated with parameters: {result}")
    except Exception as e:
        logger.error("Error during evaluation", exc_info=True)
        (
            exc_type,
            exc_value,
            exc_traceback,
        ) = sys.exc_info()  # Capture exception information
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = f'{"".join(traceback_details)} {str(e)}'
        logger.error(f"Detailed error: {error_msg}")
        result["error"] = error_msg
    write_to_json(result, f"/tmp/{run_type}_output.json")


# python entrypoint.py eval --repo
@app.command("eval")
def evaluate(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    config_template: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        config_template=config_template,
        hash=hash,
    )
    _run(request=request, run_type="eval")


@app.command("tts")
def inference_score(
    repo_name: str = typer.Argument(help="Repository name"),
    repo_namespace: str = typer.Argument(help="Repository namespace"),
    config_template: str = typer.Argument(help="Chat template type"),
    hash: Optional[str] = typer.Argument(help="hash"),
):
    request = EvaluateModelRequest(
        repo_namespace=repo_namespace,
        repo_name=repo_name,
        config_template=config_template,
        hash=hash,
    )
    _run(request=request, run_type="tts")


@app.command("stub")
def stub():
    print("stub")
    result = {"g": True}
    write_to_json(result, "/tmp/output.json")


# example: python entrypoint.py python entrypoint.py eval repo_name repo_namespace chat_template_type hash
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    app()
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
