import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

import datasets
import experiment
from ecp import networks as nets

arguments = experiment.positional_arguments()
activation = arguments[0] if len(arguments) > 0 else "crelu"
NUM_CLASSES = 5
SIGNAL_LENGTH = 128


def make_spec(width_factor):
    c1 = max(1, round(width_factor * 16))
    c2 = max(1, round(width_factor * 32))
    c3 = max(1, round(width_factor * 32))
    hidden = max(1, round(width_factor * 64))
    return [
        nets.conv(1, c1, (7,), activation), nets.avg_pool((2,)),
        nets.conv(c1, c2, (5,), activation), nets.avg_pool((2,)),
        nets.conv(c2, c3, (3,), activation), nets.avg_pool((2,)),
        nets.flatten(),
        nets.dense(c3 * (SIGNAL_LENGTH // 8), hidden, activation),
        nets.dense(hidden, NUM_CLASSES),
    ]


def make_predict(spec):
    return lambda params, x: jnp.real(nets.forward(spec, params, x))


def load_data():
    x_train, y_train = datasets.make_modulation_dataset(150000, seed=0, length=SIGNAL_LENGTH)
    x_test, y_test = datasets.make_modulation_dataset(10000, seed=1, length=SIGNAL_LENGTH)
    return dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


case = experiment.Case(
    name=f"radio_complex_conv_{activation}",
    task="classification",
    make_spec=make_spec,
    load_data=load_data,
    make_predict=make_predict,
    is_complex=True,
    seeds=[0, 1, 2, 3, 4],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    epochs_baseline=15,
    epochs_fine=8,
    batch_size=128,
    learning_rate_baseline=lambda epoch: 1e-3 if epoch < 10 else 1e-4,
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=2000,
    moments_chunk_size=50,
    joint_chunk_size=100,
)

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
