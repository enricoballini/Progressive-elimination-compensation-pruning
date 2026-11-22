import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ecp import baselines
from ecp import networks as nets
from ecp import importance as imp
from ecp import pruning, compensation
from ecp import reconstruction as rec

NO_REGULARISATION = 1e-10


def semi_holomorphic_spec(widths, m_fraction=0.5):
    spec = []
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        last = i == len(widths) - 2
        m_in = max(1, int(n_in * m_fraction))
        m_out = n_out if last else max(1, int(n_out * m_fraction))
        structure = np.ones((n_out, n_in), dtype=np.float32)
        structure[m_out:, :m_in] = 0.0
        activation = "identity" if last else (m_out, "exp", "cprelu")
        spec.append(nets.dense(n_in, n_out, activation, structure=structure))
    return spec


def build_cases():
    return {
        "real dense, prelu": (
            [nets.dense(5, 7, "prelu"), nets.dense(7, 6, "prelu"), nets.dense(6, 3)],
            False, (5,), 2.0,
        ),
        "real dense, tanh": (
            [nets.dense(4, 8, "tanh"), nets.dense(8, 8, "tanh"), nets.dense(8, 2)],
            False, (4,), 1.0,
        ),
        "complex dense, holomorphic (exp)": (
            [nets.dense(3, 6, "exp"), nets.dense(6, 5, "exp"), nets.dense(5, 2)],
            True, (3,), 0.3,
        ),
        "complex dense, split prelu": (
            [nets.dense(4, 7, "cprelu"), nets.dense(7, 6, "cprelu"), nets.dense(6, 2)],
            True, (4,), 2.0,
        ),
        "complex dense, cardioid": (
            [nets.dense(4, 7, "cardioid"), nets.dense(7, 6, "cardioid"), nets.dense(6, 2)],
            True, (4,), 2.0,
        ),
        "complex dense, semi-holomorphic": (semi_holomorphic_spec([4, 6, 6, 1]), True, (4,), 0.5),
        "real conv 2d, relu": (
            [
                nets.conv(2, 4, (3, 3), "relu"), nets.avg_pool((2, 2)),
                nets.conv(4, 5, (3, 3), "relu", padding="VALID"), nets.flatten(),
                nets.dense(5 * 2 * 2, 3),
            ],
            False, (2, 8, 8), 2.0,
        ),
        "complex conv 1d, split relu": (
            [
                nets.conv(2, 4, (5,), "crelu", padding="CIRCULAR"), nets.avg_pool((2,)),
                nets.conv(4, 4, (3,), "cardioid"), nets.flatten(),
                nets.dense(4 * 8, 3),
            ],
            True, (2, 16), 2.0,
        ),
    }


def random_inputs(key, num_samples, shape, is_complex):
    key_real, key_imag = jax.random.split(key)
    x = jax.random.normal(key_real, (num_samples,) + shape)
    if is_complex:
        x = x + 1j * jax.random.normal(key_imag, (num_samples,) + shape)
    return x


def random_network(name, seed=0, num_samples=150):
    spec, is_complex, input_shape, gain = build_cases()[name]
    key = jax.random.PRNGKey(seed)
    key_params, key_bias, key_bias_imag, key_x = jax.random.split(key, 4)
    params = nets.init_params(spec, key_params, is_complex=is_complex, gain=gain)
    new_params = []
    for i, p in enumerate(params):
        if not p:
            new_params.append({})
            continue
        b = 0.3 * jax.random.normal(jax.random.fold_in(key_bias, i), p["b"].shape)
        if is_complex:
            b = b + 0.3j * jax.random.normal(jax.random.fold_in(key_bias_imag, i), p["b"].shape)
        new_params.append({"W": p["W"], "b": b.astype(p["b"].dtype)})
    x = random_inputs(key_x, num_samples, input_shape, is_complex)
    return spec, new_params, x


def reference_importance(qoi_fn, params, x, layer_index, weight_index, step=1e-6):
    batched_qoi = jax.jit(jax.vmap(qoi_fn, in_axes=(None, 0)))
    q_original = np.asarray(batched_qoi(params, x))
    p = params[layer_index]
    w = p["W"][weight_index]
    bias_index = weight_index[0]

    def directional_derivative(delta_w, delta_b):
        new_layer = {
            "W": p["W"].at[weight_index].add(step * delta_w),
            "b": p["b"].at[bias_index].add(step * delta_b),
        }
        new_params = params[:layer_index] + [new_layer] + params[layer_index + 1 :]
        return (np.asarray(batched_qoi(new_params, x)) - q_original) / step

    effect_of_weight = directional_derivative(w, 0.0)
    bias_columns = [directional_derivative(0.0, 1.0)]
    if jnp.iscomplexobj(p["b"]):
        bias_columns.append(directional_derivative(0.0, 1.0j))

    A = np.stack([c.reshape(-1) for c in bias_columns], axis=1)
    target = effect_of_weight.reshape(-1)
    d, *_ = np.linalg.lstsq(A, target, rcond=None)
    importance = np.sum((target - A @ d) ** 2) / x.shape[0]
    importance_no_compensation = np.sum(target**2) / x.shape[0]
    shift = d[0] + 1j * d[1] if len(d) == 2 else d[0]
    return importance, shift, importance_no_compensation


def random_weight_indices(params, existing, how_many, seed=0):
    rng = np.random.default_rng(seed)
    indices = []
    layers_with_weights = [i for i, p in enumerate(params) if p]
    for _ in range(how_many):
        layer_index = int(rng.choice(layers_with_weights))
        candidates = np.argwhere(np.asarray(existing[layer_index]) > 0)
        indices.append((layer_index, tuple(int(i) for i in candidates[rng.integers(len(candidates))])))
    return indices


def test_closed_form_against_finite_differences():
    print("\n1) closed-form importance / compensation  vs  finite differences + least squares")
    print(f"{'network':38s} {'max rel. err. importance':>26s} {'max rel. err. shift':>22s} {'no-comp.':>10s}")
    for name in build_cases():
        spec, params, x = random_network(name)
        qoi_fn = imp.output_qoi(spec)
        existing = nets.existing_weights(spec, params)

        moments = imp.make_second_moments(qoi_fn, chunk_size=50)(params, x)
        importances, compensations = imp.ec_importance(params, moments, NO_REGULARISATION)
        without = imp.importance_without_compensation(params, moments)

        errors_importance, errors_shift, errors_without = [], [], []
        for layer_index, weight_index in random_weight_indices(params, existing, how_many=12):
            ref_importance, ref_shift, ref_without = reference_importance(
                qoi_fn, params, x, layer_index, weight_index
            )
            scale = ref_without + 1e-300
            errors_importance.append(abs(float(importances[layer_index][weight_index]) - ref_importance) / scale)
            errors_without.append(abs(float(without[layer_index][weight_index]) - ref_without) / scale)
            shift = complex(compensations[layer_index][weight_index])
            errors_shift.append(abs(shift - ref_shift) / (abs(ref_shift) + 1e-12))
        print(f"{name:38s} {max(errors_importance):26.2e} {max(errors_shift):22.2e} {max(errors_without):10.1e}")
        assert max(errors_importance) < 1e-4, name
        assert max(errors_shift) < 1e-3, name
        assert max(errors_without) < 1e-4, name


def test_factorised_dense_equals_generic():
    print("\n2) factorised evaluation for dense layers  vs  generic evaluation")
    for name, (spec, *_rest) in build_cases().items():
        if not all(layer["kind"] == "dense" for layer in spec):
            continue
        spec, params, x = random_network(name)
        generic = imp.make_second_moments(imp.output_qoi(spec), chunk_size=64)(params, x)
        factorised = imp.make_second_moments_dense(spec, chunk_size=64)(params, x)
        worst = 0.0
        for layer, m_generic, m_factorised in zip(spec, generic, factorised):
            structure = 1.0 if layer["structure"] is None else layer["structure"]
            for key in m_generic:
                a, b = m_generic[key], m_factorised[key]
                if a.ndim == 2:
                    a, b = a * structure, b * structure
                scale = jnp.abs(m_generic["S" + key[1:]]).max()
                worst = max(worst, float(jnp.abs(a - b).max() / scale))
        print(f"{name:38s} max rel. difference of the moments = {worst:.2e}")
        assert worst < 1e-10, name


def test_first_order_is_the_small_perturbation_limit():
    print("\n3) non-linear change for a weight scaled by (1 - t)  ->  t^2 * importance  as t -> 0")
    for name in build_cases():
        spec, params, x = random_network(name)
        qoi_fn = imp.output_qoi(spec)
        existing = nets.existing_weights(spec, params)
        moments = imp.make_second_moments(qoi_fn, chunk_size=50)(params, x)
        importances, compensations = imp.ec_importance(params, moments, NO_REGULARISATION)
        batched_qoi = jax.jit(jax.vmap(qoi_fn, in_axes=(None, 0)))
        q_original = batched_qoi(params, x)

        worst = 0.0
        t = 1e-4
        for layer_index, weight_index in random_weight_indices(params, existing, how_many=6, seed=1):
            p = params[layer_index]
            new_layer = {
                "W": p["W"].at[weight_index].mul(1.0 - t),
                "b": p["b"].at[weight_index[0]].add(t * compensations[layer_index][weight_index]),
            }
            new_params = params[:layer_index] + [new_layer] + params[layer_index + 1 :]
            nonlinear = float(jnp.sum((batched_qoi(new_params, x) - q_original) ** 2) / x.shape[0])
            first_order = t**2 * float(importances[layer_index][weight_index])
            worst = max(worst, abs(nonlinear - first_order) / (first_order + 1e-300))
        print(f"{name:38s} max rel. difference = {worst:.2e}")
        assert worst < 2e-2, name


def test_joint_compensation():
    print("\n4) discrepancy E||q_pruned - q||^2 after removing 30% of the weights")
    print(f"{'network':38s} {'no comp.':>10s} {'independent':>12s} {'joint linear':>13s} {'Gauss-Newton':>13s}")
    for name in build_cases():
        spec, params, x = random_network(name, num_samples=400)
        qoi_fn = imp.output_qoi(spec)
        existing = nets.existing_weights(spec, params)
        moments = imp.make_second_moments(qoi_fn, chunk_size=50)(params, x)
        importances, compensations = imp.ec_importance(params, moments)
        masks = pruning.global_masks(importances, existing, ratio=0.3)
        joint = compensation.JointCompensation(qoi_fn, chunk_size=100)

        pruned_plain = pruning.apply_masks(params, masks)
        pruned_independent = pruning.apply_masks_and_compensation(params, masks, existing, compensations)
        removed = pruning.removed_weights(params, masks, existing)
        shifts = joint.linear(params, removed, x)
        pruned_joint = compensation.add_bias_shifts(pruned_plain, shifts)
        pruned_gn = joint.gauss_newton(pruned_joint, params, x, num_iterations=5)

        values = [joint.discrepancy(p, params, x) for p in (pruned_plain, pruned_independent, pruned_joint, pruned_gn)]
        print(f"{name:38s} {values[0]:10.3e} {values[1]:12.3e} {values[2]:13.3e} {values[3]:13.3e}")
        assert values[3] <= values[2] * (1 + 1e-9), name


def test_singular_bias_direction():
    print("\n5) complex bias with a direction that does not influence the QoI: effect of round-off")
    print(f"{'':38s} {'rel. change of the shift':>26s} {'|shift| / |reference|':>23s}")
    spec, params, x = random_network("complex dense, split prelu")
    qoi_fn = lambda p, sample: jnp.real(nets.forward(spec, p, imp.network_input(sample)))
    moments = imp.make_second_moments(qoi_fn, chunk_size=50)(params, x)

    last = moments[-1]
    reference = jnp.real(params[-1]["W"] * last["S_bw"]) / last["S_bb"][:, None]

    rng = np.random.default_rng(0)
    perturbed = [
        {key: value * (1.0 + 1e-7 * rng.standard_normal(value.shape)) for key, value in m.items()}
        for m in moments
    ]
    for label, regularisation in (("penalty of the code", imp.COMPLEX_REGULARISATION), ("without penalty", NO_REGULARISATION)):
        _, shifts = imp.ec_importance(params, moments, regularisation)
        _, shifts_perturbed = imp.ec_importance(params, perturbed, regularisation)
        change = float(jnp.abs(shifts_perturbed[-1] - shifts[-1]).max() / jnp.abs(reference).max())
        size = float(jnp.abs(shifts_perturbed[-1]).max() / jnp.abs(reference).max())
        print(f"{label:38s} {change:26.2e} {size:23.2e}")
        if regularisation == imp.COMPLEX_REGULARISATION:
            assert change < 1e-2
            assert float(jnp.abs(shifts[-1] - reference).max() / jnp.abs(reference).max()) < 1e-2


def test_penalised_importance():
    print("\n6) importance and shift used in the experiments = minimum of the penalised problem")
    rng = np.random.default_rng(1)
    num_samples, num_qoi, n_out, n_in = 40, 3, 4, 5
    regularisation = imp.COMPLEX_REGULARISATION

    def complex_normal(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    real_part = rng.standard_normal((num_samples, num_qoi, n_out))
    derivatives_of_bias = {
        "well conditioned": complex_normal(num_samples, num_qoi, n_out),
        "nearly singular, |T_bb| ~ S_bb": real_part + 1e-4j * rng.standard_normal(real_part.shape),
        "bias without effect, S_bb = 0": np.zeros((num_samples, num_qoi, n_out), dtype=complex),
    }
    W = complex_normal(n_out, n_in)
    D = complex_normal(num_samples, num_qoi, n_out, n_in)

    def expectation(values):
        return values.mean(axis=0).sum(axis=0)

    for label, B in derivatives_of_bias.items():
        moments = {
            "S_ww": expectation(np.abs(D) ** 2),
            "S_bw": expectation(np.conj(B)[..., None] * D),
            "S_bb": expectation(np.abs(B) ** 2),
            "T_ww": expectation(D**2),
            "T_bw": expectation(B[..., None] * D),
            "T_bb": expectation(B**2),
        }
        importance, shift = imp.layer_importance(
            jnp.asarray(W), {key: jnp.asarray(value) for key, value in moments.items()}, regularisation
        )
        largest_difference = 0.0
        for i in range(n_out):
            for j in range(n_in):
                target = np.real(D[:, :, i, j] * W[i, j]).ravel() / np.sqrt(num_samples)
                matrix = np.stack([np.real(B[:, :, i]), -np.imag(B[:, :, i])], axis=-1).reshape(-1, 2)
                matrix = matrix / np.sqrt(num_samples)
                penalty = np.sqrt(regularisation / 2 * moments["S_bb"][i]) * np.eye(2)
                d = np.linalg.lstsq(np.vstack([matrix, penalty]), np.concatenate([target, np.zeros(2)]), rcond=None)[0]
                minimum = np.sum((target - matrix @ d) ** 2) + regularisation / 2 * moments["S_bb"][i] * np.sum(d**2)
                without_shift = np.sum(target**2)
                largest_difference = max(
                    largest_difference,
                    abs(float(importance[i, j]) - minimum) / without_shift,
                    abs(complex(shift[i, j]) - complex(d[0], d[1])) / max(1.0, abs(complex(d[0], d[1]))),
                )
        print(f"   {label:34s} largest relative difference: {largest_difference:.1e}")
        assert largest_difference < 1e-8


def test_optimal_brain_compression():
    print("\n7) OBC: cumulated losses of the greedy removals against least squares")
    rng = np.random.default_rng(0)
    for is_complex in (False, True):
        rows, columns, num_samples = 3, 7, 40
        X = rng.normal(size=(columns, num_samples)) + (1j * rng.normal(size=(columns, num_samples)) if is_complex else 0)
        X[-1] = 1.0
        H = X @ np.conj(X).T + 0.01 * np.eye(columns)
        W = rng.normal(size=(rows, columns)) + (1j * rng.normal(size=(rows, columns)) if is_complex else 0)
        exists = np.ones((rows, columns), dtype=bool)
        exists[0, 2], W[0, 2] = False, 0.0
        removable = exists.copy()
        removable[:, -1] = False
        H_inverse = jnp.asarray(rec.restricted_inverses(H, exists))
        order, losses = rec.removal_traces(jnp.asarray(W), H_inverse, jnp.asarray(removable))
        order, losses = np.asarray(order), np.asarray(losses)
        worst = 0.0
        for i in range(rows):
            for k in range(1, removable[i].sum() + 1):
                kept = exists[i].copy()
                kept[order[i, :k]] = False
                w = np.zeros(columns, dtype=W.dtype)
                w[kept] = np.linalg.solve(H[np.ix_(kept, kept)].T, (W[i] @ H)[kept])
                objective = np.real((W[i] - w) @ H @ np.conj(W[i] - w))
                worst = max(worst, abs(objective - losses[i, :k].sum()) / objective)
        kept = exists & ~np.eye(rows, columns, dtype=bool)
        w_kept = np.asarray(rec.least_squares_rows(jnp.asarray(W), jnp.asarray(H), jnp.asarray(kept)))
        print(f"{'complex' if is_complex else 'real':8s} largest relative error {worst:.1e}; "
              f"weights removed or not existing stay zero: {np.all(w_kept[~kept] == 0)}")
        assert worst < 1e-8 and np.all(w_kept[~kept] == 0)

    for is_complex in (False, True):
        for padding, stride, kernel, shape in (("SAME", 2, (3, 3), (4, 8, 8)), ("CIRCULAR", 1, (5,), (3, 16))):
            layer = nets.conv(shape[0], 5, kernel, padding=padding, stride=stride)
            p = nets.init_params([layer], jax.random.PRNGKey(1), is_complex)[0]
            z = random_inputs(jax.random.PRNGKey(2), 1, shape, is_complex)[0]
            a = nets.affine_map(layer, p, z) - p["b"].reshape((-1,) + (1,) * len(kernel))
            a_patches = (p["W"].reshape(5, -1) @ rec.patches(layer, z)).reshape(a.shape)
            error = float(jnp.max(jnp.abs(a - a_patches)))
            print(f"patches of a {'complex' if is_complex else 'real'} convolution, {padding}, stride {stride}: error {error:.1e}")
            assert error < 1e-10


def test_layerwise_mean_correction():
    print("\n8) layer-wise mean correction after removing 50% of the weights by magnitude")
    for name in build_cases():
        spec, params, x = random_network(name, num_samples=200)
        existing = nets.existing_weights(spec, params)
        masks = pruning.global_masks(baselines.magnitude(params), existing, ratio=0.5)
        correction = compensation.LayerwiseMeanCorrection(spec, chunk_size=64)
        corrected = correction.correct(pruning.apply_masks(params, masks), params, x)
        worst = 0.0
        for i in correction.weighted_layers:
            target = correction.mean_pre_activation(params, x, i)
            difference = correction.mean_pre_activation(corrected, x, i) - target
            worst = max(worst, float(jnp.max(jnp.abs(difference)) / jnp.max(jnp.abs(target))))
        print(f"{name:38s} largest relative difference of the mean pre-activations {worst:.1e}")
        assert worst < 1e-10, name


if __name__ == "__main__":
    test_closed_form_against_finite_differences()
    test_factorised_dense_equals_generic()
    test_first_order_is_the_small_perturbation_limit()
    test_joint_compensation()
    test_singular_bias_direction()
    test_penalised_importance()
    test_optimal_brain_compression()
    test_layerwise_mean_correction()
    print("\nAll checks passed.")
