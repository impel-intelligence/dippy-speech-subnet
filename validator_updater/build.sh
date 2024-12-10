#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 --wallet.name <coldkey> --wallet.hotkey <hotkey> --org.name <orgname> [--wallet.path <path>] [--offline] [--immediate]"
    echo "Required arguments:"
    echo "  --wallet.name: Name of the wallet (e.g., coldkey4)"
    echo "  --wallet.hotkey: Hotkey for the wallet (e.g., hotkey4)"
    echo "  --org.name: Name of the organization (e.g., MyOrganization)"
    echo "Optional arguments (defaults):"
    echo "  --wallet.path: /root/.bittensor/wallets/"
    echo "  --offline: Include offline mode"
    echo "  --immediate: Execute with immediate mode"
    echo "Default values:"
    echo "  netuid: 58"
    echo "  subtensor.network: finney"
    exit 1
}

# Default values
default_wallet_path="/root/.bittensor/wallets/"
default_netuid=58
default_network="finney"

# Initialize command with default values
command_args="--netuid $default_netuid --subtensor.network $default_network --wallet.path $default_wallet_path"
org_name=""
offline_mode=false
immediate_mode=false

# Flags to check required arguments
wallet_name_set=false
wallet_hotkey_set=false
org_name_set=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --wallet.name)
            command_args+=" $key $2"
            wallet_name_set=true
            shift
            shift
            ;;
        --wallet.hotkey)
            command_args+=" $key $2"
            wallet_hotkey_set=true
            shift
            shift
            ;;
        --org.name)
            org_name="$2"
            org_name_set=true
            shift
            shift
            ;;
        --wallet.path)
            command_args=$(echo "$command_args" | sed "s|--wallet.path $default_wallet_path|--wallet.path $2|")
            shift
            shift
            ;;
        --offline)
            offline_mode=true
            shift
            ;;
        --immediate)
            immediate_mode=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Ensure required arguments are provided
if [ "$wallet_name_set" = false ] || [ "$wallet_hotkey_set" = false ] || [ "$org_name_set" = false ]; then
    echo "Error: --wallet.name, --wallet.hotkey, and --org.name are required."
    usage
fi

# Add optional --offline flag if specified
if [ "$offline_mode" = true ]; then
    command_args+=" --offline"
fi

# Add optional --immediate flag if specified
if [ "$immediate_mode" = true ]; then
    command_args+=" --immediate"
fi

# Add default --debug flag
command_args+=" --debug"

# Export environment variables for Docker Compose
export VALIDATOR_COMMAND="python validator.py$command_args"
export ORG_NAME="$org_name"

# Run Docker Compose with overrides
docker compose up --build -d
