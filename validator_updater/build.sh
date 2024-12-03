#!/bin/bash

# Set strict mode for safer script execution
set -euo pipefail

# Define variables
IMAGE_NAME="outer-dind"
DOCKERFILE="validator_updater/Dockerfile.outer-dind"
HOST_WALLETS_DIR="${HOME}/.bittensor/wallets"
CONTAINER_DIR="/host_wallets"

# Function to build the Docker image
build_image() {
    echo "Building the Docker image with tag '${IMAGE_NAME}'..."
    docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" .
    echo "Docker image '${IMAGE_NAME}' built successfully."
}

# Function to run the Docker container
run_container() {
    echo "Running the Docker container..."
    docker run -v "$HOST_WALLETS_DIR":"$CONTAINER_DIR" --privileged -d "$IMAGE_NAME"
    echo "Container started successfully with image '${IMAGE_NAME}'."
}

# Main script logic
main() {
    # Check if Dockerfile exists
    if [[ ! -f "$DOCKERFILE" ]]; then
        echo "Error: Dockerfile '$DOCKERFILE' not found."
        exit 1
    fi

    # Check if the host wallets directory exists
    if [[ ! -d "$HOST_WALLETS_DIR" ]]; then
        echo "Error: Host wallets directory '$HOST_WALLETS_DIR' does not exist."
        exit 1
    fi

    build_image
    run_container
}

# Execute the main function
main
