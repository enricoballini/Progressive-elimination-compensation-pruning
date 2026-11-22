import pathlib
import pickle
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import compensation, networks, procedures, training
from ecp import importance as imp

HERE = pathlib.Path(__file__).resolve().parent
TARGETS = [0.8, 0.9, 0.95, 0.98]
STEPS_BETWEEN_RATIOS = 2
GAUSS_NEWTON_ITERATIONS = 2


def main():
    sys.argv = ["run.py", "32", "prelu"]
    case = runpy.run_path(str(HERE / "case_1_mnist_dense/run.py"))["case"]
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments = imp.make_second_moments_dense(spec, model["output_to_qoi_loss_metric"], chunk_size=5000)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], 1000)

    def prune(params, existing, samples, ratios, steps):
        return procedures.progressive_elimination_compensation(
            params, existing, moments, samples, ratios, joint=joint, joint_samples=samples,
            steps_between_ratios=steps, gauss_newton_iterations=GAUSS_NEWTON_ITERATIONS,
        )

    def metrics(params):
        return experiment.evaluate(model["metrics"], params, data["x_test"], data["y_test"])

    steps_to_target = {r: STEPS_BETWEEN_RATIOS * (case.ratios.index(r) + 1) for r in TARGETS}
    path = {"accuracy": np.zeros((len(case.seeds), len(TARGETS))), "loss": np.zeros((len(case.seeds), len(TARGETS)))}
    single = {"accuracy": np.zeros_like(path["accuracy"]), "loss": np.zeros_like(path["loss"])}

    for s, seed in enumerate(case.seeds):
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, data["x_train"], data["y_train"],
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        x_cal = data["x_train"][: case.calibration_size]
        samples = (x_cal, jax.nn.softmax(model["batched_predict"](params, x_cal)))

        on_path = prune(params, existing, samples, case.ratios, STEPS_BETWEEN_RATIOS)
        for t, target in enumerate(TARGETS):
            result = metrics(on_path[case.ratios.index(target)][1])
            for key in path:
                path[key][s, t] = result[key]
            (_, direct), = prune(params, existing, samples, [target], steps_to_target[target])
            result = metrics(direct)
            for key in single:
                single[key][s, t] = result[key]
        print(f"seed {seed}: path {np.round(100 * path['accuracy'][s], 1)}, "
              f"single target {np.round(100 * single['accuracy'][s], 1)}", flush=True)

    difference = 100 * (single["accuracy"] - path["accuracy"])
    print("targets:", TARGETS, " steps:", [steps_to_target[r] for r in TARGETS])
    print("path          ", " ".join(f"{100 * m:.1f}+-{100 * d:.1f}" for m, d in zip(path["accuracy"].mean(0), path["accuracy"].std(0))))
    print("single target ", " ".join(f"{100 * m:.1f}+-{100 * d:.1f}" for m, d in zip(single["accuracy"].mean(0), single["accuracy"].std(0))))
    print("difference    ", " ".join(f"{m:+.1f}+-{d:.1f}" for m, d in zip(difference.mean(0), difference.std(0))))

    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "single_target.pkl", "wb") as f:
        pickle.dump(dict(targets=TARGETS, steps=steps_to_target, path=path, single=single), f)


if __name__ == "__main__":
    main()
