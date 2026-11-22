import pathlib
import pickle
import runpy
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import compensation, networks, procedures, training
from ecp import importance as imp

HERE = pathlib.Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
SETTINGS = [
    dict(probes=None, iterations=2, samples=5000),
    dict(probes=None, iterations=1, samples=5000),
    dict(probes=2, iterations=1, samples=5000),
    dict(probes=1, iterations=1, samples=5000),
    dict(probes=1, iterations=1, samples=2000),
]


def main():
    sys.argv = ["run.py"]
    case = runpy.run_path(str(HERE / "case_3_cifar_conv/run.py"))["case"]
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    qoi = model["qoi_loss_metric"]
    num_components = spec[-1]["n_out"]

    functions = []
    for setting in SETTINGS:
        components = num_components if setting["probes"] is None else setting["probes"]
        setting_qoi = qoi if setting["probes"] is None else imp.with_random_probes(qoi)
        larger_chunks = num_components // components
        functions.append(dict(
            moments=imp.make_second_moments(setting_qoi, case.moments_chunk_size * larger_chunks),
            joint=compensation.JointCompensation(setting_qoi, case.joint_chunk_size * larger_chunks),
        ))

    shape = (len(SEEDS), len(SETTINGS), len(case.ratios))
    losses, accuracies = np.zeros(shape), np.zeros(shape)
    times = np.zeros(shape[:2])
    for s, seed in enumerate(SEEDS):
        key_init, key_train, _, key_probes = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, data["x_train"], data["y_train"],
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        for i, (setting, fn) in enumerate(zip(SETTINGS, functions)):
            x_cal = data["x_train"][: setting["samples"]]
            samples = (x_cal, jax.nn.softmax(model["batched_predict"](params, x_cal)))
            if setting["probes"] is not None:
                probes = jax.random.rademacher(
                    key_probes, (setting["samples"], num_components, setting["probes"])
                ).astype(jnp.float32)
                samples = samples + (probes,)
            jax.block_until_ready(samples)

            start = time.perf_counter()
            pruned = procedures.progressive_elimination_compensation(
                params, existing, fn["moments"], samples, case.ratios,
                joint=fn["joint"], joint_samples=samples,
                steps_between_ratios=case.steps_between_ratios,
                gauss_newton_iterations=setting["iterations"],
            )
            jax.block_until_ready(pruned)
            times[s, i] = time.perf_counter() - start

            for j, (_, params_pruned) in enumerate(pruned):
                metrics = experiment.evaluate(model["metrics"], params_pruned, data["x_test"], data["y_test"])
                losses[s, i, j], accuracies[s, i, j] = metrics["loss"], metrics["accuracy"]
        print(f"seed {seed} done", flush=True)

    print("ratios:", case.ratios)
    for i, setting in enumerate(SETTINGS):
        print(setting, f"time {times[:, i].min():6.2f}s  accuracy:",
              " ".join(f"{100 * value:5.1f}" for value in accuracies[:, i].mean(axis=0)))
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "cost_reduction.pkl", "wb") as f:
        pickle.dump(dict(settings=SETTINGS, ratios=case.ratios, seeds=SEEDS, times=times,
                         losses=losses, accuracies=accuracies), f)


if __name__ == "__main__":
    main()
