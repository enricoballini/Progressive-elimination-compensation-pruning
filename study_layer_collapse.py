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

CASES = {
    "mnist": ("case_1_mnist_dense/run.py", ["32", "prelu"]),
    "cifar": ("case_3_cifar_conv/run.py", []),
    "radio": ("case_4_radio_complex_conv/run.py", ["crelu"]),
}


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def moment_functions(case, spec, model):
    if case.factorised:
        output = imp.make_second_moments_dense(spec, chunk_size=case.moments_chunk_size)
        loss_metric = imp.make_second_moments_dense(
            spec, model["output_to_qoi_loss_metric"], chunk_size=case.moments_chunk_size
        )
    else:
        output = imp.make_second_moments(model["qoi_output"], case.moments_chunk_size)
        loss_metric = imp.make_second_moments(model["qoi_loss_metric"], case.moments_chunk_size)
    return output, loss_metric


def study(name):
    case = load_case(*CASES[name])
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    moments_output, moments_loss_metric = moment_functions(case, spec, model)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)
    x_cal = data["x_train"][: case.calibration_size]
    ratios = case.ratios

    kept = {"ec": [], "ec_progressive": [], "obd": []}
    for seed in case.seeds:
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(seed), 4)
        params = networks.init_params(spec, key_init, case.is_complex, case.init_gain)
        existing = networks.existing_weights(spec, params)
        params = training.train(
            params, existing, train_epoch, data["x_train"], data["y_train"],
            case.epochs_baseline, case.learning_rate_baseline, key_train,
        )
        samples = (x_cal, jax.nn.softmax(model["batched_predict"](params, x_cal)))
        obd_scores = imp.importance_without_compensation(params, moments_output(params, x_cal))
        pruned = {
            "ec": procedures.elimination_compensation(params, existing, moments_output, x_cal, ratios),
            "ec_progressive": procedures.progressive_elimination_compensation(
                params, existing, moments_loss_metric, samples, ratios, joint=joint, joint_samples=samples,
                steps_between_ratios=case.steps_between_ratios,
                gauss_newton_iterations=case.gauss_newton_iterations,
            ),
            "obd": procedures.prune_with_scores(params, existing, obd_scores, ratios),
        }
        for method, result in pruned.items():
            kept[method].append([[int(mask.sum()) for mask in masks if mask is not None] for masks, _ in result])
            accuracy = [
                experiment.evaluate(model["metrics"], p, data["x_test"], data["y_test"])["accuracy"]
                for _, p in result
            ]
            print(f"seed {seed} {method:16s} accuracy after pruning: {np.round(100 * np.array(accuracy), 1)}", flush=True)

    total = [int(e.sum()) for e in existing if e is not None]
    kinds = [layer["kind"] for layer, e in zip(spec, existing) if e is not None]
    print("weights per layer:", total, kinds)
    is_kernel = np.array([kind == "conv" for kind in kinds])
    for method, values in kept.items():
        values = np.array(values)
        for j, ratio in enumerate(ratios):
            mean = values[:, j].mean(axis=0)
            line = f"{method:16s} r = {ratio}: kept per layer, mean {mean.round(1)}, min {values[:, j].min(axis=0)}"
            if is_kernel.any():
                removed_kernel = 1 - mean[is_kernel].sum() / np.sum(np.array(total)[is_kernel])
                removed_dense = 1 - mean[~is_kernel].sum() / np.sum(np.array(total)[~is_kernel])
                line += f"; removed: kernel entries {100 * removed_kernel:.1f}%, dense weights {100 * removed_dense:.1f}%"
            print(line, flush=True)

    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    out_name = "layer_collapse.pkl" if name == "mnist" else f"layer_collapse_{name}.pkl"
    with open(out_dir / out_name, "wb") as f:
        pickle.dump(dict(kept=kept, ratios=ratios, total=total, kinds=kinds), f)


if __name__ == "__main__":
    study(sys.argv[1] if len(sys.argv) > 1 else "mnist")
