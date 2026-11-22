#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p data/mnist data/cifar10 data/diffusion_sorption
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [ -f "data/mnist/$f.gz" ] || curl -fL -o "data/mnist/$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
done
if [ ! -f data/cifar10/test_batch.bin ] || [ ! -f data/cifar10/data_batch_1.bin ]; then
    curl -fL -o data/cifar10.tar.gz https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz
    tar -xzf data/cifar10.tar.gz -C data/cifar10 --strip-components=1
    rm data/cifar10.tar.gz
fi
if [ ! -f data/diffusion_sorption/1D_diff-sorp_NA_NA.h5 ]; then
    echo "data/diffusion_sorption/1D_diff-sorp_NA_NA.h5 (PDEBench, case 5) must be downloaded manually"
fi
