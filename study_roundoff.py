import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import compensation, networks, procedures, training
from study_ranking import moments_of_progressive_ec
from study_without_labels import CASES, load_case

HERE = pathlib.Path(__file__).resolve().parent
NAME = "mnist"
RELATIVE_PERTURBATION = 1e-7
NUM_PERTURBATIONS = 3


def study():
    script, arguments = CASES[NAME]
    case = load_case(script, arguments)
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal = x_train[: case.calibration_size]

    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size, case.augment)
    second_moments = moments_of_progressive_ec(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)

    def accuracy_after_pruning(params):
        existing = networks.existing_weights(spec, params)
        reference = jax.nn.softmax(model["batched_predict"](params, x_cal))
        samples = (x_cal, reference)
        pruned = procedures.progressive_elimination_compensation(
            params, existing, second_moments, samples, case.ratios,
            joint=joint, joint_samples=samples,
            steps_between_ratios=case.steps_between_ratios,
            gauss_newton_iterations=case.gauss_newton_iterations,
        )
        return [
            100 * float(experiment.evaluate(model["metrics"], params_pruned, x_test, y_test)["accuracy"])
            for _, params_pruned in pruned
        ]

    rng = np.random.default_rng(0)
    results = dict(name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), unperturbed=[], perturbed=[])
    for seed in case.seeds:
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        stored = HERE / "results_studies" / "trained" / f"{NAME}_seed_{seed}.pkl"
        if stored.exists():
            params = experiment.load_params(stored)
        else:
            params = experiment.train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train)
        results["unperturbed"].append(accuracy_after_pruning(params))
        perturbed = []
        for _ in range(NUM_PERTURBATIONS):
            params_perturbed = [
                {"W": p["W"] * (1 + RELATIVE_PERTURBATION * jnp.asarray(rng.standard_normal(p["W"].shape), p["W"].dtype)),
                 "b": p["b"]}
                for p in params
            ]
            perturbed.append(accuracy_after_pruning(params_perturbed))
        results["perturbed"].append(perturbed)
        print(f"[{NAME}] seed {seed}: unperturbed " + " ".join(f"{a:6.2f}" for a in results["unperturbed"][-1]), flush=True)
        for accuracies in perturbed:
            print(" " * 16 + "perturbed   " + " ".join(f"{a:6.2f}" for a in accuracies), flush=True)

    unperturbed = np.array(results["unperturbed"])
    perturbed = np.array(results["perturbed"])
    difference = perturbed - unperturbed[:, None]
    print("ratios                            " + " ".join(f"{r:6.2f}" for r in case.ratios))
    print("largest change in one seed        " + " ".join(f"{a:6.2f}" for a in np.abs(difference).max(axis=(0, 1))))
    print("largest change of the mean        " + " ".join(f"{a:6.2f}" for a in np.abs(difference.mean(axis=0)).max(axis=0)))
    return results


if __name__ == "__main__":
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    result = study()
    with open(out_dir / "roundoff.pkl", "wb") as f:
        pickle.dump(result, f)
