import pathlib
import pickle
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

import experiment
from ecp import baselines, networks, training
from ecp import importance as imp

HERE = pathlib.Path(__file__).resolve().parent

CASES = {
    "mnist": ("case_1_mnist_dense/run.py", ["32", "prelu"], 300, 2000, 50),
    "laplace": ("case_2_laplace_holomorphic/run.py", ["32"], 150, 600, 25),
    "cifar": ("case_3_cifar_conv/run.py", [], 150, 250, 10),
    "radio": ("case_4_radio_complex_conv/run.py", ["crelu"], 150, 500, 15),
    "diffsorp": ("case_5_diffusion_sorption/run.py", [], 150, 2000, 50),
}


def load_case(script, arguments):
    sys.argv = [script] + arguments
    return runpy.run_path(str(HERE / script))["case"]


def study(name):
    script, arguments, weights_per_layer, num_samples, chunk = CASES[name]
    case = load_case(script, arguments)
    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)

    key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(0), 4)
    params = networks.init_params(spec, key_init, case.is_complex, case.init_gain)
    existing = networks.existing_weights(spec, params)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size)
    params = training.train(
        params, existing, train_epoch, data["x_train"], data["y_train"],
        case.epochs_baseline, case.learning_rate_baseline, key_train,
    )

    qoi = model["qoi_output"]
    x = data["x_train"][:num_samples]
    moments = imp.make_second_moments(qoi, chunk_size=min(case.moments_chunk_size, 100))(params, x)
    importances, compensations = imp.ec_importance(params, moments)
    without = imp.importance_without_compensation(params, moments)
    nonlinear = baselines.make_nonlinear_importance(qoi, num_iterations=8)
    is_complex = case.is_complex
    if is_complex:
        importances_0, compensations_0 = imp.ec_importance(params, imp.without_complementary_moments(moments))

    rng = np.random.default_rng(0)
    keys = ["first_order", "first_order_no_comp", "optimal", "guess", "no_comp", "layer"]
    if is_complex:
        keys.append("guess_without_complementary")
    collected = {key: [] for key in keys}
    for layer_index, p in enumerate(params):
        if not p:
            continue
        candidates = np.argwhere(np.asarray(existing[layer_index]) > 0)
        chosen = candidates[rng.choice(len(candidates), size=min(weights_per_layer, len(candidates)), replace=False)]

        @jax.jit
        def evaluate(params, x, indices, shifts):
            return jax.vmap(
                lambda index, shift: nonlinear(params, layer_index, tuple(index), shift, x)
            )(indices, shifts)

        index_tuple = tuple(chosen.T)
        shifts = compensations[layer_index][index_tuple]
        for start in range(0, len(chosen), chunk):
            out = evaluate(params, x, jnp.asarray(chosen[start : start + chunk]), shifts[start : start + chunk])
            for key in ("optimal", "guess"):
                collected[key].append(np.asarray(out[key]))
            collected["no_comp"].append(np.asarray(out["no_compensation"]))
        if is_complex:
            shifts_0 = compensations_0[layer_index][index_tuple]
            for start in range(0, len(chosen), chunk):
                out = evaluate(params, x, jnp.asarray(chosen[start : start + chunk]), shifts_0[start : start + chunk])
                collected["guess_without_complementary"].append(np.asarray(out["guess"]))
        collected["first_order"].append(np.asarray(importances[layer_index][index_tuple]))
        collected["first_order_no_comp"].append(np.asarray(without[layer_index][index_tuple]))
        collected["layer"].append(np.full(len(chosen), layer_index))

    collected = {key: np.concatenate(value) for key, value in collected.items()}
    result = dict(
        name=name,
        num_weights=len(collected["layer"]),
        spearman_optimal=spearmanr(collected["first_order"], collected["optimal"])[0],
        spearman_guess=spearmanr(collected["first_order"], collected["guess"])[0],
        spearman_no_comp=spearmanr(collected["first_order_no_comp"], collected["no_comp"])[0],
        median_gain_guess=np.nanmedian(collected["no_comp"] / collected["guess"]),
        median_gain_optimal=np.nanmedian(collected["no_comp"] / collected["optimal"]),
        per_layer={
            int(layer): spearmanr(
                collected["first_order"][collected["layer"] == layer],
                collected["optimal"][collected["layer"] == layer],
            )[0]
            for layer in np.unique(collected["layer"])
        },
        raw=collected,
    )
    if is_complex:
        def all_weights(values):
            return np.concatenate([np.asarray(v)[np.asarray(e) > 0] for v, e in zip(values, existing) if v is not None])

        result["spearman_with_without_complementary"] = spearmanr(all_weights(importances), all_weights(importances_0))[0]
        result["median_gain_without_complementary"] = np.nanmedian(
            collected["no_comp"] / collected["guess_without_complementary"]
        )
        print(
            f"{name:12s} without the complementary moments: median reduction of the change by the shift "
            f"{result['median_gain_without_complementary']:.2f}x; Spearman between the importances with and "
            f"without them (all weights) {result['spearman_with_without_complementary']:.4f}",
            flush=True,
        )
    print(
        f"{name:12s} weights {result['num_weights']:5d}  "
        f"Spearman: first-order vs non-linear (optimal shift) {result['spearman_optimal']:.4f}, "
        f"(first-order shift) {result['spearman_guess']:.4f}, without compensation {result['spearman_no_comp']:.4f}; "
        f"per layer {[round(v, 3) for v in result['per_layer'].values()]}; "
        f"median reduction of the change by the shift: first-order {result['median_gain_guess']:.2f}x, "
        f"optimal {result['median_gain_optimal']:.2f}x",
        flush=True,
    )
    return result


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CASES)
    out_dir = HERE / "results_studies"
    out_dir.mkdir(exist_ok=True)
    for name in names:
        result = study(name)
        with open(out_dir / f"first_order_accuracy_{name}.pkl", "wb") as f:
            pickle.dump(result, f)
