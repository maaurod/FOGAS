#!/bin/bash
# Activation script for FOGAS virtual environment
# Usage: source activate_venv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

echo "✅ FOGAS virtual environment activated"
echo "Python: $(which python3)"
echo "Location: $SCRIPT_DIR/venv"
