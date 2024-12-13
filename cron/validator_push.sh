#!/bin/bash

BUILD_DATE=$(date +%Y%m%d%H%M%S)  # Generate a unique build timestamp

# Define variables
DOCKERFILE="validator_updater/Dockerfile.validator"
IMAGE="ghcr.io/impel-intelligence/validator_script_image:latest"

# Log in to GHCR (you only need this once, but you can include it for safety)
#echo "<your-PAT>" | docker login ghcr.io -u <your-github-username> --password-stdin

# Build the Docker image and pass the BUILD_DATE as a build argument
docker build --build-arg BUILD_DATE=$BUILD_DATE -f $DOCKERFILE -t $IMAGE .

# Push the Docker image
docker push $IMAGE

echo "Docker image pushed at $(date)"
