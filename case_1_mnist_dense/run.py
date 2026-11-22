import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datasets
import experiment
from ecp import networks as nets

arguments = experiment.positional_arguments()
width = int(arguments[0]) if len(arguments) > 0 else 32
activation = arguments[1] if len(arguments) > 1 else "prelu"
with_ablation = len(arguments) > 2
unit_pixels = "--unit-pixels" in sys.argv


def make_spec(width_factor):
    hidden = max(1, round(width_factor * width))
    return [
        nets.dense(784, hidden, activation),
        nets.dense(hidden, hidden, activation),
        nets.dense(hidden, 10),
    ]


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist(unit_interval=unit_pixels)
    return dict(
        x_train=x_train.reshape(-1, 784), y_train=y_train,
        x_test=x_test.reshape(-1, 784), y_test=y_test,
    )


case = experiment.Case(
    name=f"mnist_dense_{width}_{activation}" + ("_unit_pixels" if unit_pixels else ""),
    task="classification",
    make_spec=make_spec,
    load_data=load_data,
    seeds=[0, 1, 2, 3, 4],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    epochs_baseline=15,
    epochs_fine=15,
    batch_size=128,
    learning_rate_baseline=lambda epoch: 1e-3 if epoch < 10 else 1e-4,
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=10000,
    factorised=True,
    moments_chunk_size=5000,
    joint_chunk_size=1000,
)

if with_ablation:
    case.methods = case.methods + ("ec_loss_metric", "ec_joint", "ec_progressive_independent")

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
