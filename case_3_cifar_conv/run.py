import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datasets
import experiment
from ecp import networks as nets


def make_spec(width_factor):
    c1 = max(1, round(width_factor * 32))
    c2 = max(1, round(width_factor * 64))
    c3 = max(1, round(width_factor * 64))
    hidden = max(1, round(width_factor * 64))
    return [
        nets.conv(3, c1, (3, 3), "relu"), nets.avg_pool((2, 2)),
        nets.conv(c1, c2, (3, 3), "relu"), nets.avg_pool((2, 2)),
        nets.conv(c2, c3, (3, 3), "relu"), nets.avg_pool((2, 2)),
        nets.flatten(),
        nets.dense(c3 * 4 * 4, hidden, "relu"),
        nets.dense(hidden, 10),
    ]


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.load_cifar10()
    return dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


case = experiment.Case(
    name="cifar_conv",
    task="classification",
    make_spec=make_spec,
    load_data=load_data,
    seeds=[0, 1, 2, 3, 4],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    epochs_baseline=20,
    epochs_fine=10,
    batch_size=128,
    learning_rate_baseline=lambda epoch: 1e-3 if epoch < 15 else 1e-4,
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=5000,
    moments_chunk_size=25,
    joint_chunk_size=50,
)

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
