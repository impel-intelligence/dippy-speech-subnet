#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display informational messages
echo_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Function to display error messages
echo_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo_error "Docker is not installed. Please install Docker and try again."
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

echo_info "Docker is installed."

# Pull the Validator Script Image
echo_info "Pulling the Validator Script Image..."
docker pull ghcr.io/impel-intelligence/validator_script_image:latest

# Watchtower configuration
WATCHTOWER_IMAGE="containrrr/watchtower:latest"
WATCHTOWER_CONTAINER_NAME="watchtower"
WATCHTOWER_INTERVAL=300  # Check for updates every 300 seconds (5 minutes)

# Pull the latest Watchtower image
echo_info "Pulling the latest Watchtower image..."
docker pull $WATCHTOWER_IMAGE

# Stop and remove existing Watchtower container if it exists
if docker ps -a --format '{{.Names}}' | grep -Eq "^${WATCHTOWER_CONTAINER_NAME}\$"; then
    echo_info "Stopping and removing existing Watchtower container..."
    docker stop $WATCHTOWER_CONTAINER_NAME
    docker rm $WATCHTOWER_CONTAINER_NAME
fi

# Run Watchtower with label filtering
echo_info "Starting Watchtower container with label filtering..."
docker run -d \
    --name $WATCHTOWER_CONTAINER_NAME \
    --restart unless-stopped \
    -v /var/run/docker.sock:/var/run/docker.sock \
    containrrr/watchtower \
    --interval $WATCHTOWER_INTERVAL \
    --label-enable

echo_info "Watchtower is now running and monitoring labeled containers for updates."

# Validator Script configuration
VALIDATOR_CONTAINER_NAME="validator_script"
WALLET_HOST_DIR="~/.bittensor/wallets/"
WALLET_CONTAINER_DIR="/root/.bittensor/wallets/"


# Run Validator Script Container with Watchtower Label and Volume Mapping
echo_info "Starting Validator Script container with Watchtower label and volume mapping..."
docker run -d \
    --name $VALIDATOR_CONTAINER_NAME \
    --restart unless-stopped \
    --label com.centurylinklabs.watchtower.enable=true \
    -v "$WALLET_HOST_DIR":"$WALLET_CONTAINER_DIR" \
    ghcr.io/impel-intelligence/validator_script_image:latest

echo_info "Validator Script container is now running and monitored by Watchtower."
