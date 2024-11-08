FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
WORKDIR /app


## Copy the requirements file into the container
COPY requirements_modelq.txt requirements.txt

RUN uv pip install -r requirements.txt --prerelease=allow

COPY dippy_validation_api ./dippy_validation_api
COPY scoring ./scoring
COPY utilities ./utilities
COPY model ./model
COPY neurons ./neurons
COPY constants ./constants
# Required for self installing module
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY neurons/model_queue.py .

CMD ["python", "model_queue.py"]


