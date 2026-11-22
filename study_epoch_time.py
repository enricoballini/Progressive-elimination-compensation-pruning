import pathlib
import pickle
import runpy
import sys
import time

import jax
import jax.numpy as jnp

import experiment
from ecp import networks, training

HERE = pathlib.Path(__file__).resolve().parent

CASES = {
    "mnist_dense_32_prelu": ("case_1_mnist_dense/run.py", ["32", "prelu"]),
    "laplace_holomorphic_32": ("case_2_laplace_holomorphic/run.py", ["32"]),
    "cifar_conv": ("case_3_cifar_conv/run.py", []),
    "radio_complex_conv_crelu": ("case_4_radio_complex_conv/run.py", ["crelu"]),
    "diffusion_sorption": ("case_5_diffusion_sorption/run.py", []),
    "cifar_resnet20": ("case_6_cifar_resnet/run.py", []),
}


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def time_of_one_epoch(case):
    data = case.load_data()
    x_train, y_train = jnp.asarray(data["x_train"]), jnp.asarray(data["y_train"])
    spec = case.make_spec(1.0)
    if case.train_dense is not None:
        keys = jax.random.split(jax.random.PRNGKey(0))
        jax.block_until_ready(case.train_dense(spec, *keys, x_train, y_train, epochs=1))
        times = {}
        for epochs in (1, 4):
            start = time.perf_counter()
            jax.block_until_ready(case.train_dense(spec, *keys, x_train, y_train, epochs=epochs))
            times[epochs] = time.perf_counter() - start
        return (times[4] - times[1]) / 3
    model = experiment.make_model(case, spec)
    params = networks.init_params(spec, jax.random.PRNGKey(0), case.is_complex, case.init_gain)
    existing = networks.existing_weights(spec, params)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    opt_state = training.adam_init(params)
    key = jax.random.PRNGKey(0)

    def one_epoch():
        return jax.block_until_ready(train_epoch(params, opt_state, existing, x_train, y_train, key, 1e-3))

    one_epoch()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        one_epoch()
        times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    out_file = HERE / "results_studies" / "epoch_time.pkl"
    names = sys.argv[1:] or list(CASES)
    results = {}
    if sys.argv[1:] and out_file.exists():
        with open(out_file, "rb") as f:
            results = pickle.load(f)
    for name in names:
        script, arguments = CASES[name]
        case = load_case(script, arguments)
        results[name] = dict(
            time_epoch=time_of_one_epoch(case),
            epochs_baseline=case.epochs_baseline,
            epochs_fine=case.epochs_fine,
        )
        print(name, results[name], flush=True)

    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
