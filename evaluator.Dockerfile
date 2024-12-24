FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN python -m pip install --upgrade pip

# env variable required by uv
ENV CONDA_PREFIX=/opt/conda
RUN pip install uv

WORKDIR /app

# Install required dependencies and td-agent
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -L https://packages.treasuredata.com/GPG-KEY-td-agent | apt-key add - && \
    echo "deb https://packages.treasuredata.com/4/ubuntu/jammy jammy contrib" > /etc/apt/sources.list.d/td-agent.list && \
    apt-get update && \
    apt-get install -y td-agent && \
    td-agent-gem install fluent-plugin-logtail -v 0.2.0 && \
    apt-get clean

# Configure Fluentd to direct logs to BetterStack
COPY scoring/fluentd/fluent.conf /etc/td-agent/td-agent.conf

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
COPY requirements_val_api.txt requirements_val_api.txt
RUN uv pip install --system -r requirements_val_api.txt --no-build-isolation 

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

# Expose Fluentd port to forward logs externally 
EXPOSE 24224

# Start both Fluentd (td-agent) and Python application
ENTRYPOINT ["sh", "-c", "service td-agent start && exec python /app/entrypoint.py"]
