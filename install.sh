#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV=${VENV:-venv}
PYTHON=${PYTHON:-python3.12}
CPU=0
DATA=0

for arg in "$@"; do
    case $arg in
        --cpu)  CPU=1 ;;
        --data) DATA=1 ;;
        *) echo "usage: ./install.sh [--cpu] [--data]"; exit 1 ;;
    esac
done

if ! command -v "$PYTHON" > /dev/null; then
    echo "$PYTHON not found: install Python 3.12 or set PYTHON=/path/to/python"
    exit 1
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip

if [ "$CPU" = 1 ]; then
    sed 's/^jax\[cuda12\]/jax/' requirements.txt > "$VENV/requirements_cpu.txt"
    "$VENV/bin/python" -m pip install -r "$VENV/requirements_cpu.txt"
else
    "$VENV/bin/python" -m pip install -r requirements.txt
fi

if [ "$DATA" = 1 ]; then
    ./download_data.sh
fi

JAX_PLATFORMS=cpu "$VENV/bin/python" tests/test_formulas.py

echo "installation complete: source $VENV/bin/activate"
