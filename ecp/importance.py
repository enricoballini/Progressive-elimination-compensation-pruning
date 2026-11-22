import jax
import jax.numpy as jnp

from . import networks


def network_input(sample):
    return sample[0] if isinstance(sample, tuple) else sample


def as_real_vector(y):
    y = jnp.reshape(y, (-1,))
    if jnp.iscomplexobj(y):
        return jnp.concatenate([y.real, y.imag])
    return y


def output_qoi(spec):
    return lambda params, x: as_real_vector(networks.forward(spec, params, x))


def whitened_logits(logits, reference_probabilities):
    p = reference_probabilities
    return jnp.sqrt(p) * (logits - jnp.sum(p * logits))


def with_random_probes(qoi_fn):

    def probed_qoi(params, sample):
        *original, probes = sample
        original = original[0] if len(original) == 1 else tuple(original)
        return probes.T @ qoi_fn(params, original) / jnp.sqrt(probes.shape[1])

    return probed_qoi


def bias_like_weight(B, W):
    return B.reshape(B.shape + (1,) * (W.ndim - 1))


def sum_over_samples_and_qoi(values):
    return values.sum(axis=(0, 1))


def chunk_moments_generic(qoi_fn, params, x_chunk):
    jacobians = jax.vmap(jax.jacrev(qoi_fn), in_axes=(None, 0))(params, x_chunk)

    moments = []
    for p, jac in zip(params, jacobians):
        if not p:
            moments.append({})
            continue
        D = jac["W"]
        B = jac["b"]
        B_w = bias_like_weight(B, p["W"])
        layer_moments = {
            "S_ww": sum_over_samples_and_qoi(jnp.abs(D) ** 2),
            "S_bw": sum_over_samples_and_qoi(jnp.conj(B_w) * D),
            "S_bb": sum_over_samples_and_qoi(jnp.abs(B) ** 2),
        }
        if jnp.iscomplexobj(D):
            layer_moments["T_ww"] = sum_over_samples_and_qoi(D**2)
            layer_moments["T_bw"] = sum_over_samples_and_qoi(B_w * D)
            layer_moments["T_bb"] = sum_over_samples_and_qoi(B**2)
        moments.append(layer_moments)
    return moments


def make_second_moments(qoi_fn, chunk_size=64):
    chunk_fn = jax.jit(lambda params, x_chunk: chunk_moments_generic(qoi_fn, params, x_chunk))

    def second_moments(params, x):
        return average_over_chunks(lambda x_chunk: chunk_fn(params, x_chunk), x, chunk_size)

    return second_moments


def chunk_moments_dense(spec, params, qoi_of_output, x_chunk):
    zero_shifts = [jnp.zeros_like(p["b"]) for p in params]

    def qoi_of_shifts(shifts, sample):
        y = networks.forward_with_bias_shifts(spec, params, shifts, network_input(sample))
        return qoi_of_output(y, sample)

    def layer_inputs(sample):
        return networks.forward_with_layer_inputs(spec, params, network_input(sample))[1]

    g = jax.vmap(jax.jacrev(qoi_of_shifts), in_axes=(None, 0))(zero_shifts, x_chunk)
    z = jax.vmap(layer_inputs)(x_chunk)

    moments = []
    for g_l, z_l in zip(g, z):
        alpha = (jnp.abs(g_l) ** 2).sum(axis=1)
        layer_moments = {
            "S_ww": alpha.T @ (jnp.abs(z_l) ** 2),
            "S_bw": alpha.T @ z_l,
            "S_bb": alpha.sum(axis=0),
        }
        if jnp.iscomplexobj(g_l):
            gamma = (g_l**2).sum(axis=1)
            layer_moments["T_ww"] = gamma.T @ (z_l**2)
            layer_moments["T_bw"] = gamma.T @ z_l
            layer_moments["T_bb"] = gamma.sum(axis=0)
        moments.append(layer_moments)
    return moments


def make_second_moments_dense(spec, qoi_of_output=None, chunk_size=1024):
    assert all(layer["kind"] == "dense" for layer in spec), "dense layers only"
    if qoi_of_output is None:
        qoi_of_output = lambda y, sample: as_real_vector(y)
    chunk_fn = jax.jit(
        lambda params, x_chunk: chunk_moments_dense(spec, params, qoi_of_output, x_chunk)
    )

    def second_moments(params, x):
        return average_over_chunks(lambda x_chunk: chunk_fn(params, x_chunk), x, chunk_size)

    return second_moments


def num_samples_of(samples):
    return jax.tree_util.tree_leaves(samples)[0].shape[0]


def take_chunk(samples, start, chunk_size):
    return jax.tree_util.tree_map(lambda leaf: leaf[start : start + chunk_size], samples)


def sum_over_chunks(chunk_fn, samples, chunk_size):
    total = None
    for start in range(0, num_samples_of(samples), chunk_size):
        part = chunk_fn(take_chunk(samples, start, chunk_size))
        total = part if total is None else jax.tree_util.tree_map(jnp.add, total, part)
    return total


def average_over_chunks(chunk_fn, samples, chunk_size):
    total = sum_over_chunks(chunk_fn, samples, chunk_size)
    return jax.tree_util.tree_map(lambda value: value / num_samples_of(samples), total)


def safe_divide(numerator, denominator):
    is_zero = denominator == 0
    return jnp.where(is_zero, 0.0, numerator / jnp.where(is_zero, 1.0, denominator))


COMPLEX_REGULARISATION = 1e-3


def layer_importance(W, m, regularisation=COMPLEX_REGULARISATION):
    S_bb = bias_like_weight(m["S_bb"], W)

    if not jnp.iscomplexobj(W):
        compensation = W * safe_divide(m["S_bw"], S_bb)
        importance = W**2 * (m["S_ww"] - safe_divide(m["S_bw"] ** 2, S_bb))
        return jnp.maximum(importance, 0.0), compensation

    T_bb = bias_like_weight(m["T_bb"], W)
    r = W * m["S_bw"] + jnp.conj(W * m["T_bw"])
    S_bb_reg = S_bb * (1.0 + regularisation)
    determinant = S_bb_reg**2 - jnp.abs(T_bb) ** 2
    compensation = safe_divide(S_bb_reg * r - jnp.conj(T_bb) * jnp.conj(r), determinant)

    no_compensation = 0.5 * (jnp.abs(W) ** 2 * m["S_ww"] + jnp.real(W**2 * m["T_ww"]))
    importance = no_compensation - 0.5 * jnp.real(jnp.conj(compensation) * r)
    return jnp.maximum(importance, 0.0), compensation


def layer_importance_without_compensation(W, m):
    if not jnp.iscomplexobj(W):
        return W**2 * m["S_ww"]
    return 0.5 * (jnp.abs(W) ** 2 * m["S_ww"] + jnp.real(W**2 * m["T_ww"]))


def without_complementary_moments(moments):
    return [
        {name: jnp.zeros_like(value) if name.startswith("T_") else value for name, value in m.items()}
        for m in moments
    ]


def ec_importance(params, moments, regularisation=COMPLEX_REGULARISATION):
    importances, compensations = [], []
    for p, m in zip(params, moments):
        if not p:
            importances.append(None)
            compensations.append(None)
            continue
        importance, compensation = layer_importance(p["W"], m, regularisation)
        importances.append(importance)
        compensations.append(compensation)
    return importances, compensations


def importance_without_compensation(params, moments):
    return [
        layer_importance_without_compensation(p["W"], m) if p else None
        for p, m in zip(params, moments)
    ]
