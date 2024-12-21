#!/bin/bash

echo "This script will remove all Docker containers, images, volumes, and networks."
echo "This action is irreversible. Do you want to proceed? (yes/no)"
read -r confirmation

if [[ "$confirmation" != "yes" && "$confirmation" != "y" ]]; then
  echo "Aborting. No changes were made."
  exit 0
fi

# Stop all running containers (if any)
if [[ -n $(docker ps -q) ]]; then
  echo "Stopping all running containers..."
  docker stop $(docker ps -q)
else
  echo "No running containers to stop."
fi

# Remove all containers (if any)
if [[ -n $(docker ps -aq) ]]; then
  echo "Removing all containers..."
  docker rm $(docker ps -aq)
else
  echo "No containers to remove."
fi

# Remove all images (if any)
if [[ -n $(docker images -q) ]]; then
  echo "Removing all images..."
  docker rmi $(docker images -q) -f
else
  echo "No images to remove."
fi

# Remove all volumes (if any)
if [[ -n $(docker volume ls -q) ]]; then
  echo "Removing all volumes..."
  docker volume rm $(docker volume ls -q) -f
else
  echo "No volumes to remove."
fi


# Clean up unused Docker system resources
echo "Cleaning up unused Docker system resources..."
docker system prune -a --volumes -f

echo "Docker cleanup complete!"

