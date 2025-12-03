#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Python virtual environment in .venv and installing dependencies..."

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade wheel setuptools

echo "Installing requirements.txt..."
python -m pip install -r requirements.txt

echo "Setup complete. To activate the environment run:"
echo "  source .venv/bin/activate"
