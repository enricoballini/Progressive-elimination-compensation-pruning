import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import compensation, networks, procedures, training
from study_ranking import moments_of_progressive_ec
from study_without_labels import CASES, load_case

HERE = pathlib.Path(__file__).resolve().parent


def main(steps):
    case = load_case(*CASES["mnist"])
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments = moments_of_progressive_ec(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)

    results = {"ratios": case.ratios, "steps_between_ratios": steps, "without": [], "with": []}
    for seed in case.seeds:
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = experiment.train_dense_network(
            case, spec, train_epoch, data["x_train"], data["y_train"], key_init, key_train)
        existing = networks.existing_weights(spec, params)
        x_cal = data["x_train"][: case.calibration_size]
        samples = (x_cal, jax.nn.softmax(model["batched_predict"](params, x_cal)))
        for variant, flag in (("without", False), ("with", True)):
            pruned = procedures.progressive_elimination_compensation(
                params, existing, moments, samples, case.ratios, joint=joint, joint_samples=samples,
                steps_between_ratios=steps,
                gauss_newton_iterations=case.gauss_newton_iterations, with_residual_term=flag,
            )
            metrics = [experiment.evaluate(model["metrics"], p, data["x_test"], data["y_test"]) for _, p in pruned]
            results[variant].append(metrics)
        a = np.array([[m["accuracy"] for m in results[v][-1]] for v in ("without", "with")])
        print(f"seed {seed}: without {np.round(100 * a[0], 1)}\n        with    {np.round(100 * a[1], 1)}", flush=True)

    a = {v: 100 * np.array([[m["accuracy"] for m in run] for run in results[v]]) for v in ("without", "with")}
    print("ratios", case.ratios)
    print("difference (with - without), mean:", np.round((a["with"] - a["without"]).mean(0), 2))
    print("                              std:", np.round((a["with"] - a["without"]).std(0), 2))
    name = "residual_term.pkl" if steps == case.steps_between_ratios else f"residual_term_{steps}_steps.pkl"
    with open(HERE / "results_studies" / name, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
