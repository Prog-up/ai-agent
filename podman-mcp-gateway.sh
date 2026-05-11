#!/bin/bash
set -e
echo "Updating packages and installing Podman..."
sudo apt update && sudo apt install -y podman
sudo loginctl enable-linger enzo

echo "Enabling Podman socket..."
systemctl --user enable --now podman.socket

echo "Downloading Docker MCP Gateway..."
mkdir -p ~/.local/bin
curl -L https://github.com/docker/mcp-gateway/releases/latest/download/docker-mcp-linux-amd64 -o ~/.local/bin/docker-mcp
chmod +x ~/.local/bin/docker-mcp

echo "Configuring environment variables..."
if ! grep -q 'DOCKER_MCP_IN_CONTAINER' ~/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  echo 'export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"' >> ~/.bashrc
  echo 'export DOCKER_MCP_IN_CONTAINER=1' >> ~/.bashrc
fi

echo "VM Setup Complete!"
