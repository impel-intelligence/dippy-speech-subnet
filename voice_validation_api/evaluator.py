import copy
import io
import json
import os
import tarfile
import time
from typing import Any, Dict, Optional, Union

import docker
from dotenv import load_dotenv
from pydantic import BaseModel

from common.event_logger import EventLogger
from common.scores import Scores
from scoring.common import EvaluateModelRequest

DEFAULT_IMAGE_NAME = "speech:latest"

DEFAULT_HOME_DIR = os.environ.get("EVALUATOR_HOME_DIR", "/home/new_prod_user/dippy-voice-subnet")
DEFAULT_MODEL_CACHE_DIR = os.environ.get("EVALUATOR_MODEL_CACHE_DIR", "/workdir/model_cache_dir")


class EvaluationScore(BaseModel):
    eval_score: float
    latency_score: float
    eval_model_size_score: float
    creativity_score: float


class RunError(BaseModel):
    error: str


class HumanSimilarityScore(BaseModel):
    human_similarity_score: float


class Evaluator:
    def __init__(
        self,
        image_name: str = DEFAULT_IMAGE_NAME,
        gpu_ids: str = "0",
        logger: EventLogger = EventLogger(),
        trace: bool = False,
    ):
        # self.client = docker.from_env(version="auto", timeout=600)
        self.client = docker.DockerClient(
            base_url='unix://var/run/docker.sock',  # Use the appropriate base URL for your system
            version="auto",  # Automatically detect the server's version
            timeout=600  # Set the API call timeout in seconds
        )
        self.logger = logger

        if trace:
            self.logger = EventLogger(
                filepath="/tmp/valapi_event_logs/trace_{time:UNIX}.log",
                level="DEBUG",
                stderr=True,
            )
        self.image_name = image_name

        prompt_template_path = os.path.join(DEFAULT_HOME_DIR, "scoring/prompt_templates")
        evalsets_template_path = os.path.join(DEFAULT_HOME_DIR, "evalsets")

        prompt_template_path = str(prompt_template_path)

        self.volume_configuration = {
            prompt_template_path: {
                "bind": "/app/prompt_templates",
                "mode": "ro",
            },
            evalsets_template_path: {
                "bind": "/app/evalsets",
                "mode": "rw",
            },
        }
        if trace:
            scoring_path = os.path.join(DEFAULT_HOME_DIR, "scoring")
            self.volume_configuration[str(scoring_path)] = {
                "bind": "/app/scoring",
                "mode": "ro",
            }
            self.volume_configuration[DEFAULT_MODEL_CACHE_DIR] = {
                "bind": "/app/model_cache_dir",
                "mode": "rw",
            }
            self.volume_configuration[DEFAULT_MODEL_CACHE_DIR] = {
                "bind": "/app/model_cache_dir",
                "mode": "rw",
            }

        self.device_requests = [docker.types.DeviceRequest(device_ids=[gpu_ids], capabilities=[["gpu"]])]

        # Load environment variables from the .env file
        load_dotenv()

        self.env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "VLLM_WORKER_MULTIPROC_METHOD": "_",
            "PYTORCH_CUDA_ALLOC_CONF": "_",
            "DATASET_API_KEY": os.environ.get("DATASET_API_KEY"),
        }
        self.trace = trace

    def run_docker_container(
        self,
        job_type: str,
        request: EvaluateModelRequest,
    ) -> dict:
        volumes = self.volume_configuration
        device_requests = self.device_requests

        command = f"{job_type} {request.to_args()}"
        self.logger.info("command", command=command)
        self.logger.info("device_requests", device_requests=device_requests)

        env = copy.copy(self.env)
        if job_type == "tts":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.logger.debug("env", env=env)
        self.logger.info("image", image=self.image_name )

        container = self.client.containers.run(
            image=self.image_name,
            command=command,
            environment={
                **env,
                "PYTHONPATH": "/app"  # Add /app to PYTHONPATH
            },
            working_dir='/app',
            device_requests=device_requests,
            detach=True

        )


        # container = self.client.containers.run(
        #     image='speech:latest',
        #     command=command,
        #     environment={
        #         **env,
        #         "PYTHONPATH": "/app"  # Add /app to PYTHONPATH
        #     },
        #     working_dir='/app',
        #     detach=True,
        #     stdout=True,
        #     stderr=True
        # )





        # Log the output of the container in real-time
        for log in container.logs(stream=True):
            print(log.decode("utf-8"))  # Print to console or use `self.logger.info`

        filepath = f"/tmp/{job_type}_output.json"
        filename = f"{job_type}_output.json"

        print("now waiting for container to complete")
        result = container.wait()
        self.logger.debug(f"container_run_complete, {result}")
        print(f"container_run_complete, {result}")

        # try:
        #     bits, _ = container.get_archive(filepath)

        #     with io.BytesIO() as file_data:

        #         for chunk in bits:

        #             file_data.write(chunk)

        #         file_data.seek(0)

        #         with tarfile.open(fileobj=file_data) as tar:

        #             content = tar.extractfile(filename).read().decode("utf-8")

        #             container_results = json.loads(content)

        #             self.logger.info(
        #                 "container_run_results",
        #                 details={
        #                     "filepath": filepath,
        #                     "content": content,
        #                     "result": result,
        #                     "container_id": container.id,
        #                 },
        #             )

        #             if not self.trace:
        #                 try:
        #                     container.remove()
        #                 except Exception as e:
        #                     self.logger.error("container_remove_error")
        #             return container_results
        # except Exception as e:
        #     self.logger.error("docker_error", error=e)
        #     if not self.trace:
        #         container.remove()
        #     return {"error": str(e)}

    def human_similarity_score(self, request: EvaluateModelRequest) -> Union[HumanSimilarityScore, RunError]:
        try:
            human_similarity_result = self.run_docker_container(
                job_type="tts",
                request=request,
            )
            if "error" in human_similarity_result:
                raise Exception(human_similarity_result["error"])
            if human_similarity_result["completed"] is False:
                raise Exception("completion internal error")
            score = HumanSimilarityScore(
                human_similarity_score=human_similarity_result["final_score"],
            )
            return score
        except Exception as e:
            return RunError(error=str(e))


import math

STEEPNESS = 5
THRESHOLD = 0.2


def calculate_c_score(initial_score, creativity_score, threshold=0.2, steepness=5):
    final_score = initial_score / (1 + math.exp(-steepness * (creativity_score - threshold)))
    return final_score


# Command to manually run evaluation
def entry():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run a single evaluation instance")
    parser.add_argument("--image", type=str, default="grader:latest", help="image to use")
    parser.add_argument("--repo_namespace", type=str, required=True, help="Repository namespace")
    parser.add_argument("--repo_name", type=str, required=True, help="Repository name")
    parser.add_argument("--chat_template_type", type=str, required=True, help="Chat template type")
    parser.add_argument("--hash", type=str, required=True, help="Unique hash value")

    args = parser.parse_args()
    image_name = args.image
    req = EvaluateModelRequest(
        repo_namespace=args.repo_namespace,
        repo_name=args.repo_name,
        chat_template_type=args.chat_template_type,
        hash=args.hash,
    )
    print(f"running {image_name} with {req}")

    try:
        evaler = Evaluator(image_name=image_name, trace=True, gpu_ids="0")

        human_similarity_result = evaler.human_similarity_score(req)
        if isinstance(human_similarity_result, RunError):
            raise Exception(human_similarity_result.error)
        print(f"human_similarity_result : {human_similarity_result}")

        # eval_result = evaler.eval_score(req)
        # print(f"eval_result : {eval_result}")
        # if isinstance(eval_result, RunError):
        #     raise Exception(eval_result.error)

        # scores_data = Scores()
        # scores_data.qualitative_score = eval_result.eval_score
        # scores_data.latency_score = eval_result.latency_score
        # scores_data.creativity_score = eval_result.creativity_score
        # scores_data.llm_size_score = eval_result.eval_model_size_score
        # scores_data.vibe_score = infrence_result.vibe_score
        # scores_data.coherence_score = infrence_result.coherence_score

        # final_eval_score = (
        #     scores_data.adjusted_q_score(
        #         scores_data.qualitative_score,
        #         scores_data.creativity_score,
        #     )
        #     * 0.82
        # )
        # final_model_size_score = scores_data.llm_size_score * 0.06
        # final_latency_score = scores_data.latency_score * 0.06
        # final_vibe_score = scores_data.vibe_score * 0.06

        # total_score = final_eval_score + final_model_size_score + final_latency_score + final_vibe_score
        # print(f"final_model_size_score {final_model_size_score}")
        # print(f"final_latency_score {final_latency_score}")
        # print(f"final_vibe_score {final_vibe_score}")
        # print(f"final_eval_score {final_eval_score}")
        # print(f"coherence score: {scores_data.coherence_score}")
        # print(f"score pre coherence: {total_score}")
        # print(f"total score: {scores_data.calculate_total_score()}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    entry()