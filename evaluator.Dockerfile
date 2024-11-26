FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel


RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN python -m pip install --upgrade pip

# env variable required by uv
ENV CONDA_PREFIX=/opt/conda
RUN pip install uv

WORKDIR /app

# Create required directories and files
RUN mkdir -p \
    voice_validation_api \
    model_cache_dir \
    scoring/prompt_templates \
    scoring/scoring_logic \
    common \
    constants \
    utilities \
    neurons && \
    touch voice_validation_api/__init__.py


# Ensure /tmp is writable
RUN chmod 777 /tmp

# Copy requirements first to leverage Docker cache
COPY requirements.api.txt requirements.api.txt
RUN uv pip install --system -r requirements.api.txt --no-build-isolation 


# Copy project files maintaining directory structure
COPY scoring/prompt_templates/*.jinja scoring/prompt_templates/
COPY scoring/prompt_templates/*.py scoring/prompt_templates/
COPY scoring/scoring_logic/*.py scoring/scoring_logic/
COPY scoring/*.py scoring/
COPY utilities/ utilities/
COPY common/ common/
COPY constants/ constants/
COPY neurons/ neurons/
COPY voice_validation_api/ voice_validation_api/


# Copy project configuration files
COPY README.md pyproject.toml ./
COPY .git/ ./.git/

# Install the package in editable mode
RUN uv pip install --system -e .

# Copy entrypoint script
COPY scoring/entrypoint.py ./

ENTRYPOINT ["python", "entrypoint.py"]

