#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Running classroom.py..."
python3 classroom.py

echo "Running kitchen.py..."
python3 kitchen.py