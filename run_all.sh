#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${GPU_FRACTION:-0.3}
cd "$(dirname "$0")"
mkdir -p logs

if [ -n "$PYTHON" ]; then
    PY=$PYTHON
elif [ -x venv/bin/python ]; then
    PY=$PWD/venv/bin/python
else
    PY=python
fi
if ! "$PY" -c "import jax" 2> /dev/null; then
    echo "$PY cannot import jax: run ./install.sh first, or set PYTHON=/path/to/python"
    exit 1
fi

DIFFSORP=data/diffusion_sorption/1D_diff-sorp_NA_NA.h5
missing=0
for f in data/mnist/train-images-idx3-ubyte.gz data/cifar10/data_batch_1.bin; do
    if [ ! -f "$f" ]; then
        echo "missing dataset: $f"
        missing=1
    fi
done
if [ "$missing" = 1 ]; then
    echo "./install.sh --data downloads MNIST and CIFAR-10"
    exit 1
fi
if [ ! -f "$DIFFSORP" ]; then
    echo "missing dataset: $DIFFSORP (diffusion-sorption steps will be skipped)"
fi

run() {
    name=$1; shift
    case "$name $*" in
        *diffsorp*|*diffusion*)
            if [ ! -f "$DIFFSORP" ]; then
                echo "=== skipped $name (no diffusion-sorption data)" | tee -a logs/queue.log
                return
            fi ;;
    esac
    echo "=== $(date +%H:%M:%S) start $name" | tee -a logs/queue.log
    $PY "$@" > "logs/$name.log" 2>&1
    code=$?
    echo "    $(date +%H:%M:%S) exit code $code ($name)" | tee -a logs/queue.log
}

JAX_PLATFORMS=cpu run test_formulas tests/test_formulas.py

run mnist_32_prelu           case_1_mnist_dense/run.py 32 prelu ablation
run diffusion_sorption       case_5_diffusion_sorption/run.py
run laplace_holomorphic_32   case_2_laplace_holomorphic/run.py 32
run cifar_conv               case_3_cifar_conv/run.py
run radio_complex_conv_crelu case_4_radio_complex_conv/run.py crelu
run cifar_resnet20           case_6_cifar_resnet/run.py

run mnist_64_prelu           case_1_mnist_dense/run.py 64 prelu
run mnist_256_prelu          case_1_mnist_dense/run.py 256 prelu
run mnist_64_tanh            case_1_mnist_dense/run.py 64 tanh
run mnist_32_prelu_unit_pixels case_1_mnist_dense/run.py 32 prelu --unit-pixels

run study_first_order_accuracy study_first_order_accuracy.py
run study_epoch_time           study_epoch_time.py
for case in mnist cifar radio; do
    run study_layer_collapse_$case study_layer_collapse.py $case
done
run study_sensitivity          study_sensitivity.py
run study_single_target        study_single_target.py
run study_regularisation       study_regularisation.py
run study_complementary_moments study_complementary_moments.py
run study_cost_reduction       study_cost_reduction.py
for case in mnist diffsorp laplace cifar radio; do
    run study_ranking_$case    study_ranking.py $case
done
run study_residual_term        study_residual_term.py
run study_residual_term_8_steps study_residual_term.py 8
for case in mnist diffsorp laplace cifar radio resnet; do
    run study_without_labels_$case study_without_labels.py $case
done
for case in mnist cifar laplace; do
    run study_robustness_$case study_robustness.py $case
done
run study_roundoff             study_roundoff.py
for case in mnist diffsorp laplace cifar radio resnet; do
    for phase in train ec ec_progressive magnitude_joint obc; do
        XLA_FLAGS=--xla_gpu_autotune_level=0 run study_memory_${case}_$phase study_memory.py $case $phase
    done
done

echo "QUEUE_DONE $(date +%H:%M:%S)" >> logs/queue.log
