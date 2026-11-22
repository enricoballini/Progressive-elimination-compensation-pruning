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
REFERENCE = dict(steps=2, iterations=2, samples=10000)
VARIATIONS = [
    ("steps", [1, 2, 4, 8]),
    ("iterations", [0, 1, 2, 5]),
    ("samples", [500, 2000, 10000, 60000]),
]


def main():
    sys.argv = ["run.py", "32", "prelu"]
    case = runpy.run_path(str(HERE / "case_1_mnist_dense/run.py"))["case"]
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments = imp.make_second_moments_dense(spec, model["output_to_qoi_loss_metric"], chunk_size=5000)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], 1000)

    settings = [dict(REFERENCE, **{name: value}) for name, values in VARIATIONS for value in values]
    losses = np.zeros((len(case.seeds), len(settings), len(case.ratios)))

    for s, seed in enumerate(case.seeds):
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, data["x_train"], data["y_train"],
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        for i, setting in enumerate(settings):
            x_cal = data["x_train"][: setting["samples"]]
            samples = (x_cal, jax.nn.softmax(model["batched_predict"](params, x_cal)))
            pruned = procedures.progressive_elimination_compensation(
                params, existing, moments, samples, case.ratios,
                joint=joint if setting["iterations"] > 0 else None, joint_samples=samples,
                steps_between_ratios=setting["steps"], gauss_newton_iterations=setting["iterations"],
            )
            losses[s, i] = [
                experiment.evaluate(model["metrics"], p, data["x_test"], data["y_test"])["loss"]
                for _, p in pruned
            ]
        print(f"seed {seed} done", flush=True)

    mean = losses.mean(axis=0)
    print("ratios:", case.ratios)
    for setting, row in zip(settings, mean):
        print(setting, " ".join(f"{value:.3f}" for value in row))
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sensitivity.pkl", "wb") as f:
        pickle.dump(dict(settings=settings, losses=losses, ratios=case.ratios, variations=VARIATIONS), f)


if __name__ == "__main__":
    main()
