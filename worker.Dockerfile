FROM python:3.12


RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io && \
    rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
WORKDIR /app


## Copy the requirements file into the container
COPY requirements.api.txt requirements.txt

RUN uv pip install -r requirements.txt --prerelease=allow

COPY voice_validation_api ./voice_validation_api
COPY scoring ./scoring
COPY utilities ./utilities
COPY common/ common/
COPY constants/ constants/
COPY neurons ./neurons
COPY constants ./constants
# Required for self installing module
COPY .env .
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY voice_validation_api/worker_queue.py .

ENTRYPOINT ["python", "worker_queue.py", "--queue", "8"]
