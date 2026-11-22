import dataclasses
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp

import experiment
from ecp import baselines, compensation, networks, procedures, reconstruction, training
from study_ranking import moments_of_progressive_ec
from study_without_labels import CASES, load_case

HERE = pathlib.Path(__file__).resolve().parent


def study(name):
    case = load_case(*CASES[name])
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal = x_train[: case.calibration_size]
    network_inputs = x_cal if case.network_inputs is None else case.network_inputs(x_cal)

    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size, case.augment)
    second_moments = moments_of_progressive_ec(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)
    obc = {
        damping: reconstruction.OptimalBrainCompression(spec, case.joint_chunk_size, relative_damping=damping)
        for damping in (0.01, 0.1, 0.001)
    }
    distill = {
        "": experiment.make_distillation(case, model),
        "_lr_1e-3": experiment.make_distillation(
            dataclasses.replace(case, learning_rate_fine=lambda epoch: 1e-3), model),
    }

    results = dict(name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), baseline=[], methods={})
    for seed in case.seeds:
        key_init, key_train, key_fine, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = experiment.train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train)
        existing = networks.existing_weights(spec, params)
        results["baseline"].append(experiment.evaluate(model["metrics"], params, x_test, y_test))
        print(f"\n[{name}] seed {seed}: baseline {results['baseline'][-1]}", flush=True)
        if case.task == "classification":
            reference = jax.nn.softmax(model["batched_predict"](params, x_cal))
            samples = (x_cal, reference)
        else:
            reference = model["batched_predict"](params, x_cal)
            samples = x_cal

        def progressive(steps):
            return procedures.progressive_elimination_compensation(
                params, existing, second_moments, samples, case.ratios, joint=joint, joint_samples=samples,
                steps_between_ratios=steps, gauss_newton_iterations=case.gauss_newton_iterations,
            )

        pruned = {"ec_progressive": progressive(case.steps_between_ratios), "ec_progressive_8_steps": progressive(8)}
        ec_ratios = [reconstruction.ratios_of_masks(m, existing) for m, _ in pruned["ec_progressive"]]
        magnitude = procedures.prune_with_scores(params, existing, baselines.magnitude(params), case.ratios)
        magnitude_ratios = [reconstruction.ratios_of_masks(m, existing) for m, _ in magnitude]
        pruned["obc_ec_allocation"] = [obc[0.01].prune(params, existing, network_inputs, r) for r in ec_ratios]
        pruned["obc_ec_allocation_damping_0.1"] = [obc[0.1].prune(params, existing, network_inputs, r) for r in ec_ratios]
        pruned["obc_ec_allocation_damping_0.001"] = [obc[0.001].prune(params, existing, network_inputs, r) for r in ec_ratios]
        pruned["obc_magnitude_allocation"] = [obc[0.01].prune(params, existing, network_inputs, r) for r in magnitude_ratios]
        for suffix, function in distill.items():
            for source, start in (("magnitude", magnitude), ("ec_progressive", pruned["ec_progressive"])):
                pruned[f"{source}_distillation{suffix}"] = [
                    (masks, function(p, masks, x_cal, reference, key_fine, x_train.shape[0])) for masks, p in start
                ]

        metric = "accuracy" if case.task == "classification" else "loss"
        for variant, networks_of_ratios in pruned.items():
            metrics = [experiment.evaluate(model["metrics"], p, x_test, y_test) for _, p in networks_of_ratios]
            stored = results["methods"].setdefault(variant, dict(pruned=[], finetuned=[]))
            stored["pruned"].append(metrics)
            print(f"  {variant:36s} " + " ".join(f"{m[metric]:.3e}" for m in metrics), flush=True)
    return results


if __name__ == "__main__":
    name = sys.argv[1]
    result = study(name)
    with open(HERE / "results_studies" / f"robustness_{name}.pkl", "wb") as f:
        pickle.dump(result, f)
