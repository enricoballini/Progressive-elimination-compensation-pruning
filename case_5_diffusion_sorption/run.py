import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import datasets
import experiment
from ecp import networks as nets

HIDDEN_WIDTHS = (64, 32, 32, 16, 16)


def make_spec(width_factor):
    widths = [68] + [max(1, round(width_factor * n)) for n in HIDDEN_WIDTHS] + [1]
    spec = [nets.dense(n_in, n_out, "prelu") for n_in, n_out in zip(widths[:-2], widths[1:-1])]
    return spec + [nets.dense(widths[-2], widths[-1])]


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.load_diffusion_sorption()
    return dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


case = experiment.Case(
    name="diffusion_sorption",
    task="regression",
    make_spec=make_spec,
    load_data=load_data,
    seeds=[0, 1, 2, 3, 4],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    epochs_baseline=20,
    epochs_fine=10,
    batch_size=512,
    learning_rate_baseline=lambda epoch: 1e-3 if epoch < 14 else 1e-4,
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=20000,
    factorised=True,
    moments_chunk_size=5000,
    joint_chunk_size=2000,
)

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
