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
REGULARISATIONS = [1e-6, 1e-4, 1e-3, 1e-2]


def main():
    sys.argv = ["run.py", "32"]
    case = runpy.run_path(str(HERE / "case_2_laplace_holomorphic/run.py"))["case"]
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments = imp.make_second_moments(model["qoi_output"], case.moments_chunk_size)
    joint = compensation.JointCompensation(model["qoi_output"], case.joint_chunk_size)
    x_cal = data["x_train"][: case.calibration_size]

    def test_losses(pruned):
        return [
            experiment.evaluate(model["metrics"], p, data["x_test"], data["y_test"])["loss"]
            for _, p in pruned
        ]

    shape = (len(case.seeds), len(REGULARISATIONS), len(case.ratios))
    losses = {"ec": np.zeros(shape), "ec_progressive": np.zeros(shape)}
    for s, seed in enumerate(case.seeds):
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init, case.is_complex, case.init_gain)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, data["x_train"], data["y_train"],
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        for i, regularisation in enumerate(REGULARISATIONS):
            losses["ec"][s, i] = test_losses(procedures.elimination_compensation(
                params, existing, moments, x_cal, case.ratios, regularisation=regularisation,
            ))
            losses["ec_progressive"][s, i] = test_losses(procedures.progressive_elimination_compensation(
                params, existing, moments, x_cal, case.ratios, joint=joint, joint_samples=x_cal,
                steps_between_ratios=case.steps_between_ratios,
                gauss_newton_iterations=case.gauss_newton_iterations, regularisation=regularisation,
            ))
        print(f"seed {seed} done", flush=True)

    print("ratios:", case.ratios)
    for method, values in losses.items():
        for regularisation, row in zip(REGULARISATIONS, values.mean(axis=0)):
            print(f"{method:16s} {regularisation:.0e}  " + " ".join(f"{value:.3e}" for value in row))
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "regularisation.pkl", "wb") as f:
        pickle.dump(dict(regularisations=REGULARISATIONS, ratios=case.ratios, losses=losses), f)


if __name__ == "__main__":
    main()
