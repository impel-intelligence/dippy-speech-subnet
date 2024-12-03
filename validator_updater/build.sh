#!/bin/bash

# Define variables
IMAGE_NAME="outer-dind"
HOST_WALLETS_DIR="$HOME/.bittensor/wallets"
CONTAINER_DIR="/host_wallets"

# Build the Docker image
echo "Building the Docker image..."
docker build -t "$IMAGE_NAME" .
docker build -f Dockerfile.outer-dind -t "$IMAGE_NAME" .

# Run the Docker container
echo "Running the Docker container..."
docker run -v "$HOST_WALLETS_DIR":"$CONTAINER_DIR" --privileged -d "$IMAGE_NAME"

# Output success message
echo "Container started successfully with image '$IMAGE_NAME'."
