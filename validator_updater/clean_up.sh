#!/bin/bash

echo "This script will remove all Docker containers, images, volumes, and networks."
echo "This action is irreversible. Do you want to proceed? (yes/no)"
read -r confirmation

if [[ "$confirmation" != "yes" && "$confirmation" != "y" ]]; then
  echo "Aborting. No changes were made."
  exit 0
fi

# Stop all running containers
echo "Stopping all running containers..."
docker stop $(docker ps -q) 2>/dev/null || echo "No running containers to stop."

# Remove all containers
echo "Removing all containers..."
docker rm $(docker ps -aq) 2>/dev/null || echo "No containers to remove."

# Remove all images
echo "Removing all images..."
docker rmi $(docker images -q) -f 2>/dev/null || echo "No images to remove."

# Remove all volumes
echo "Removing all volumes..."
docker volume rm $(docker volume ls -q) -f 2>/dev/null || echo "No volumes to remove."

# Remove all networks
echo "Removing all networks..."
docker network rm $(docker network ls -q) 2>/dev/null || echo "No networks to remove."

# Remove unused Docker system resources
echo "Cleaning up unused Docker system resources..."
docker system prune -a --volumes -f

echo "Docker cleanup complete!"
