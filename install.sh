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
    mkdir -p data/mnist data/cifar10 data/diffusion_sorption
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
        [ -f "data/mnist/$f.gz" ] || curl -fL -o "data/mnist/$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
    done
    if [ ! -f data/cifar10/test_batch.bin ]; then
        curl -fL -o data/cifar10.tar.gz https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
        tar -xzf data/cifar10.tar.gz -C data/cifar10 --strip-components=1
        rm data/cifar10.tar.gz
    fi
    if [ ! -f data/diffusion_sorption/1D_diff-sorp_NA_NA.h5 ]; then
        echo "data/diffusion_sorption/1D_diff-sorp_NA_NA.h5 (PDEBench, case 5) must be downloaded manually"
    fi
fi

JAX_PLATFORMS=cpu "$VENV/bin/python" tests/test_formulas.py

echo "installation complete: source $VENV/bin/activate"
