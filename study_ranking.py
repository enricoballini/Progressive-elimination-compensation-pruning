import pathlib
import pickle
import runpy
import sys

import jax
import jax.numpy as jnp

import experiment
from ecp import baselines, compensation, networks, procedures, training
from ecp import importance as imp

HERE = pathlib.Path(__file__).resolve().parent

CASES = {
    "mnist": ("case_1_mnist_dense/run.py", ["32", "prelu"]),
    "diffsorp": ("case_5_diffusion_sorption/run.py", []),
    "laplace": ("case_2_laplace_holomorphic/run.py", ["32"]),
    "cifar": ("case_3_cifar_conv/run.py", []),
    "radio": ("case_4_radio_complex_conv/run.py", ["crelu"]),
}

RANKINGS = {
    "ec_progressive": None,
    "obd_progressive": lambda params, moments: imp.importance_without_compensation(params, moments),
    "magnitude_progressive": lambda params, moments: baselines.magnitude(params),
}


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def moments_of_progressive_ec(case, spec, model):
    if not case.factorised:
        return imp.make_second_moments(model["qoi_loss_metric"], case.moments_chunk_size)
    if case.task == "classification":
        return imp.make_second_moments_dense(
            spec, model["output_to_qoi_loss_metric"], chunk_size=case.moments_chunk_size
        )
    return imp.make_second_moments_dense(spec, chunk_size=case.moments_chunk_size)


def study(name):
    script, arguments = CASES[name]
    case = load_case(script, arguments)
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal = x_train[: case.calibration_size]

    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    second_moments = moments_of_progressive_ec(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)

    results = dict(
        name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), baseline=[],
        methods={variant: dict(pruned=[], finetuned=[]) for variant in RANKINGS},
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

        samples = x_cal
        if case.task == "classification":
            reference_probabilities = jax.nn.softmax(model["batched_predict"](params, x_cal))
            samples = (x_cal, reference_probabilities)

        for variant, ranking in RANKINGS.items():
            pruned = procedures.progressive_elimination_compensation(
                params, existing, second_moments, samples, case.ratios,
                joint=joint, joint_samples=samples,
                steps_between_ratios=case.steps_between_ratios,
                gauss_newton_iterations=case.gauss_newton_iterations,
                ranking=ranking,
            )
            after_pruning, after_finetuning = [], []
            for masks, params_pruned in pruned:
                after_pruning.append(experiment.evaluate(model["metrics"], params_pruned, x_test, y_test))
                params_fine = training.train(
                    params_pruned, masks, train_epoch, x_train, y_train,
                    case.epochs_fine, case.learning_rate_fine, key_fine,
                )
                after_finetuning.append(experiment.evaluate(model["metrics"], params_fine, x_test, y_test))
            results["methods"][variant]["pruned"].append(after_pruning)
            results["methods"][variant]["finetuned"].append(after_finetuning)
            metric = "accuracy" if case.task == "classification" else "loss"
            print(
                f"  {variant:22s} after pruning: "
                + " ".join(f"{m[metric]:.3e}" for m in after_pruning)
                + "\n" + " " * 24 + "after fine-tuning: "
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
        with open(out_dir / f"ranking_{name}.pkl", "wb") as f:
            pickle.dump(result, f)
