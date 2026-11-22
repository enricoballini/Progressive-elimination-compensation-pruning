import pathlib
import pickle
import runpy
import sys
import time

import jax
import jax.numpy as jnp

import experiment
from ecp import baselines, compensation, networks, procedures, pruning, reconstruction, training
from study_ranking import CASES as RANKING_CASES
from study_ranking import moments_of_progressive_ec

HERE = pathlib.Path(__file__).resolve().parent

CASES = dict(RANKING_CASES, resnet=("case_6_cifar_resnet/run.py", []))
VARIANTS = (
    "ec_progressive", "obc_uniform", "obc_ec_allocation",
    "obc_ec_allocation_joint", "obc_ec_allocation_distillation",
    "magnitude", "magnitude_mean_correction", "magnitude_joint",
    "magnitude_distillation", "ec_progressive_distillation",
)
WITH_FINE_TUNING = ("ec_progressive", "obc_uniform", "obc_ec_allocation")
JOINT_ITERATIONS = 5


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def study(name):
    script, arguments = CASES[name]
    case = load_case(script, arguments)
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal = x_train[: case.calibration_size]
    network_inputs = x_cal if case.network_inputs is None else case.network_inputs(x_cal)

    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size, case.augment)
    second_moments = moments_of_progressive_ec(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)
    obc = reconstruction.OptimalBrainCompression(spec, chunk_size=case.joint_chunk_size)
    mean_correction = compensation.LayerwiseMeanCorrection(spec, chunk_size=case.joint_chunk_size)
    distill = experiment.make_distillation(case, model)
    trained_dir = pathlib.Path(script).parent / "results" / case.name

    results = dict(
        name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), baseline=[],
        methods={v: dict(pruned=[], finetuned=[], time=[]) for v in VARIANTS},
        layer_ratios_ec=[],
    )
    for seed in case.seeds:
        key_init, key_train, key_fine, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        stored = HERE / trained_dir / f"trained_seed_{seed}.pkl"
        own = HERE / "results_studies" / "trained" / f"{name}_seed_{seed}.pkl"
        if stored.exists():
            params = experiment.load_params(stored)
        elif own.exists():
            params = experiment.load_params(own)
        else:
            params = experiment.train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train)
            own.parent.mkdir(parents=True, exist_ok=True)
            experiment.save_params(own, params)
        existing = networks.existing_weights(spec, params)
        results["baseline"].append(experiment.evaluate(model["metrics"], params, x_test, y_test))
        print(f"\n[{name}] seed {seed}: baseline {results['baseline'][-1]}", flush=True)

        if case.task == "classification":
            reference = jax.nn.softmax(model["batched_predict"](params, x_cal))
            samples = (x_cal, reference)
        else:
            reference = model["batched_predict"](params, x_cal)
            samples = x_cal

        pruned, elapsed = {}, {}
        t0 = time.time()
        pruned["ec_progressive"] = procedures.progressive_elimination_compensation(
            params, existing, second_moments, samples, case.ratios,
            joint=joint, joint_samples=samples,
            steps_between_ratios=case.steps_between_ratios,
            gauss_newton_iterations=case.gauss_newton_iterations,
        )
        elapsed["ec_progressive"] = time.time() - t0
        layer_ratios_ec = [reconstruction.ratios_of_masks(masks, existing) for masks, _ in pruned["ec_progressive"]]
        results["layer_ratios_ec"].append(layer_ratios_ec)

        t0 = time.time()
        pruned["obc_uniform"] = [
            obc.prune(params, existing, network_inputs, reconstruction.uniform_ratios(spec, ratio))
            for ratio in case.ratios
        ]
        elapsed["obc_uniform"] = time.time() - t0
        t0 = time.time()
        pruned["obc_ec_allocation"] = [
            obc.prune(params, existing, network_inputs, layer_ratios) for layer_ratios in layer_ratios_ec
        ]
        elapsed["obc_ec_allocation"] = time.time() - t0
        t0 = time.time()
        pruned["obc_ec_allocation_joint"] = [
            (masks, joint.gauss_newton(p, params, samples, num_iterations=JOINT_ITERATIONS))
            for masks, p in pruned["obc_ec_allocation"]
        ]
        elapsed["obc_ec_allocation_joint"] = time.time() - t0 + elapsed["obc_ec_allocation"]
        t0 = time.time()
        pruned["obc_ec_allocation_distillation"] = [
            (masks, distill(p, masks, x_cal, reference, key_fine, x_train.shape[0]))
            for masks, p in pruned["obc_ec_allocation"]
        ]
        elapsed["obc_ec_allocation_distillation"] = time.time() - t0 + elapsed["obc_ec_allocation"]

        t0 = time.time()
        pruned["magnitude"] = procedures.prune_with_scores(params, existing, baselines.magnitude(params), case.ratios)
        elapsed["magnitude"] = time.time() - t0
        t0 = time.time()
        pruned["magnitude_mean_correction"] = [
            (masks, mean_correction.correct(p, params, network_inputs)) for masks, p in pruned["magnitude"]
        ]
        elapsed["magnitude_mean_correction"] = time.time() - t0 + elapsed["magnitude"]
        t0 = time.time()
        pruned["magnitude_joint"] = [
            (masks, joint.gauss_newton(p, params, samples, num_iterations=JOINT_ITERATIONS))
            for masks, p in pruned["magnitude"]
        ]
        elapsed["magnitude_joint"] = time.time() - t0 + elapsed["magnitude"]

        t0 = time.time()
        pruned["magnitude_distillation"] = [
            (masks, distill(p, masks, x_cal, reference, key_fine, x_train.shape[0])) for masks, p in pruned["magnitude"]
        ]
        elapsed["magnitude_distillation"] = time.time() - t0 + elapsed["magnitude"]
        t0 = time.time()
        pruned["ec_progressive_distillation"] = [
            (masks, distill(p, masks, x_cal, reference, key_fine, x_train.shape[0]))
            for masks, p in pruned["ec_progressive"]
        ]
        elapsed["ec_progressive_distillation"] = time.time() - t0 + elapsed["ec_progressive"]

        for variant in VARIANTS:
            after_pruning, after_finetuning = [], []
            for (masks, params_pruned), ratio in zip(pruned[variant], case.ratios):
                assert abs(pruning.pruning_ratio(masks, existing) - ratio) < 1e-3
                after_pruning.append(experiment.evaluate(model["metrics"], params_pruned, x_test, y_test))
                if variant in WITH_FINE_TUNING:
                    params_fine = training.train(
                        params_pruned, masks, train_epoch, x_train, y_train,
                        case.epochs_fine, case.learning_rate_fine, key_fine,
                    )
                    after_finetuning.append(experiment.evaluate(model["metrics"], params_fine, x_test, y_test))
            results["methods"][variant]["pruned"].append(after_pruning)
            results["methods"][variant]["finetuned"].append(after_finetuning)
            results["methods"][variant]["time"].append(elapsed[variant])
            metric = "accuracy" if case.task == "classification" else "loss"
            line = f"  {variant:28s} {elapsed[variant]:6.1f}s  pruned: " + " ".join(f"{m[metric]:.3e}" for m in after_pruning)
            if after_finetuning:
                line += "\n" + " " * 38 + "fine-tuned: " + " ".join(f"{m[metric]:.3e}" for m in after_finetuning)
            print(line, flush=True)
    return results


if __name__ == "__main__":
    name = sys.argv[1]
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    result = study(name)
    with open(out_dir / f"without_labels_{name}.pkl", "wb") as f:
        pickle.dump(result, f)
