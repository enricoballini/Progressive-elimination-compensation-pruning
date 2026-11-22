import pathlib
import pickle
import runpy
import sys

import jax
import jax.numpy as jnp

import experiment
from ecp import compensation, networks, procedures, training
from ecp import importance as imp

HERE = pathlib.Path(__file__).resolve().parent

CASES = {
    "laplace": ("case_2_laplace_holomorphic/run.py", ["32"]),
    "radio": ("case_4_radio_complex_conv/run.py", ["crelu"]),
}
METHODS = ("ec", "ec_progressive")


def ignoring_complementary_moments(second_moments):
    return lambda params, samples: imp.without_complementary_moments(second_moments(params, samples))


def variant_name(method, complementary):
    return method if complementary else f"{method}_without_complementary"


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def study(name):
    script, arguments = CASES[name]
    case = load_case(script, arguments)
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal = x_train[: case.calibration_size]

    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments_output = imp.make_second_moments(model["qoi_output"], case.moments_chunk_size)
    moments_loss_metric = moments_output
    if case.task == "classification":
        moments_loss_metric = imp.make_second_moments(model["qoi_loss_metric"], case.moments_chunk_size)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)

    def prune(method, params, existing, samples_loss_metric, complementary):
        if method == "ec":
            moments = moments_output
            if not complementary:
                moments = ignoring_complementary_moments(moments)
            return procedures.elimination_compensation(params, existing, moments, x_cal, case.ratios)
        moments = moments_loss_metric
        if not complementary:
            moments = ignoring_complementary_moments(moments)
        return procedures.progressive_elimination_compensation(
            params, existing, moments, samples_loss_metric, case.ratios,
            joint=joint, joint_samples=samples_loss_metric,
            steps_between_ratios=case.steps_between_ratios,
            gauss_newton_iterations=case.gauss_newton_iterations,
        )

    variants = [(method, complementary) for method in METHODS for complementary in (True, False)]
    results = dict(
        name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), baseline=[],
        methods={variant_name(*v): dict(pruned=[], finetuned=[]) for v in variants},
    )
    for seed in case.seeds:
        key_init, key_train, key_fine, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init, case.is_complex, case.init_gain)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, x_train, y_train,
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        results["baseline"].append(experiment.evaluate(model["metrics"], params, x_test, y_test))
        print(f"\n[{name}] seed {seed}: baseline {results['baseline'][-1]}", flush=True)

        samples_loss_metric = x_cal
        if case.task == "classification":
            reference_probabilities = jax.nn.softmax(model["batched_predict"](params, x_cal))
            samples_loss_metric = (x_cal, reference_probabilities)

        for method, complementary in variants:
            after_pruning, after_finetuning = [], []
            for masks, params_pruned in prune(method, params, existing, samples_loss_metric, complementary):
                after_pruning.append(experiment.evaluate(model["metrics"], params_pruned, x_test, y_test))
                params_fine = training.train(
                    params_pruned, masks, train_epoch, x_train, y_train,
                    case.epochs_fine, case.learning_rate_fine, key_fine,
                )
                after_finetuning.append(experiment.evaluate(model["metrics"], params_fine, x_test, y_test))
            stored = results["methods"][variant_name(method, complementary)]
            stored["pruned"].append(after_pruning)
            stored["finetuned"].append(after_finetuning)
            metric = "accuracy" if case.task == "classification" else "loss"
            print(
                f"  {variant_name(method, complementary):28s} after pruning: "
                + " ".join(f"{m[metric]:.3e}" for m in after_pruning)
                + "\n" + " " * 30 + "after fine-tuning: "
                + " ".join(f"{m[metric]:.3e}" for m in after_finetuning),
                flush=True,
            )
    return results


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CASES)
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    for name in names:
        result = study(name)
        with open(out_dir / f"complementary_moments_{name}.pkl", "wb") as f:
            pickle.dump(result, f)
