FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
WORKDIR /app

# Accept environment variables as build arguments
ARG SUPABASE_KEY
ARG SUPABASE_URL
ARG ADMIN_KEY
ARG HF_ACCESS_TOKEN
ARG HF_USER
ARG DIPPY_KEY
ARG OPENAI_API_KEY
ARG DATASET_API_KEY
ARG POSTGRES_URL
ARG VALIDATION_API_S58


# Set environment variables from build arguments
ENV SUPABASE_KEY=$SUPABASE_KEY
ENV SUPABASE_URL=$SUPABASE_URL
ENV ADMIN_KEY=$ADMIN_KEY
ENV HF_ACCESS_TOKEN=$HF_ACCESS_TOKEN
ENV HF_USER=$HF_USER
ENV DIPPY_KEY=$DIPPY_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV DATASET_API_KEY=$DATASET_API_KEY
ENV POSTGRES_URL=$POSTGRES_URL
ENV VALIDATION_API_58=$VALIDATION_API_S58

## Copy the requirements file into the container
COPY requirements_modelq.txt requirements.txt

RUN uv pip install -r requirements.txt --prerelease=allow

COPY voice_validation_api ./voice_validation_api
COPY scoring ./scoring
COPY utilities ./utilities
COPY common/ common/
COPY constants/ constants/
COPY neurons ./neurons
COPY constants ./constants
# Required for self installing module
# COPY .env .
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY neurons/model_queue.py .

ENTRYPOINT ["python", "model_queue.py"]


