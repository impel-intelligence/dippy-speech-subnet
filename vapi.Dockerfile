# Use Python 3.12 as the base image
FROM python:3.12

# Copy uv from Astral SH's image
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv

# Enable system Python for uv
ENV UV_SYSTEM_PYTHON=1

# Set the working directory
WORKDIR /app

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy the requirements file into the container
COPY requirements.api.txt requirements.txt

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Copy application source code into the container
COPY voice_validation_api ./voice_validation_api
COPY scoring ./scoring
COPY utilities ./utilities
COPY common ./common
COPY constants ./constants

COPY README.md .
COPY pyproject.toml .
COPY .git .git

# Install package in editable mode
RUN uv pip install -e .

# Copy the main API file
COPY voice_validation_api/validation_api.py .

# Set the entry point for the container
ENTRYPOINT ["python", "validation_api.py"]
