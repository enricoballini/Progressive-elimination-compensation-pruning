import numpy as np
import jax
import jax.numpy as jnp

from . import networks


def magnitude(params):
    return [jnp.abs(p["W"]) if p else None for p in params]


def gradient_magnitude(loss_fn, params, x, y, chunk_size=2000):
    grad_fn = jax.jit(jax.grad(loss_fn))
    grads = None
    for start in range(0, x.shape[0], chunk_size):
        x_chunk, y_chunk = x[start : start + chunk_size], y[start : start + chunk_size]
        weight = x_chunk.shape[0] / x.shape[0]
        part = jax.tree_util.tree_map(lambda g: weight * g, grad_fn(params, x_chunk, y_chunk))
        grads = part if grads is None else jax.tree_util.tree_map(jnp.add, grads, part)
    return [jnp.abs(p["W"]) * jnp.abs(g["W"]) if p else None for p, g in zip(params, grads)]


def random_scores(params, seed):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.random(p["W"].shape)) if p else None for p in params]


def wanda(spec, params, existing, x, chunk_size=500):

    @jax.jit
    def squared_input_norms(params, x_chunk):
        layer_inputs = jax.vmap(lambda xs: networks.forward_with_layer_inputs(spec, params, xs)[1])(x_chunk)
        norms = []
        for layer, z in zip(spec, layer_inputs):
            if layer["kind"] == "dense":
                norms.append((jnp.abs(z) ** 2).sum(axis=0))
            elif layer["kind"] == "conv":
                other_axes = (0,) + tuple(range(2, z.ndim))
                norms.append((jnp.abs(z) ** 2).sum(axis=other_axes))
            else:
                norms.append(jnp.zeros(()))
        return norms

    squared_norms = None
    for start in range(0, x.shape[0], chunk_size):
        part = squared_input_norms(params, x[start : start + chunk_size])
        squared_norms = part if squared_norms is None else [a + b for a, b in zip(squared_norms, part)]

    scores = []
    for p, ex, squared_norm in zip(params, existing, squared_norms):
        if not p:
            scores.append(None)
            continue
        input_norm = jnp.sqrt(squared_norm).reshape((1, -1) + (1,) * (p["W"].ndim - 2))
        raw = np.asarray(jnp.abs(p["W"]) * input_norm)
        scores.append(jnp.asarray(fractional_rank_per_row(raw, np.asarray(ex) > 0)))
    return scores


def fractional_rank_per_row(values, existing):
    rows = values.reshape(values.shape[0], -1)
    existing_rows = existing.reshape(values.shape[0], -1)
    ranks = np.zeros_like(rows, dtype=np.float32)
    for i in range(rows.shape[0]):
        columns = np.flatnonzero(existing_rows[i])
        order = np.argsort(rows[i, columns], kind="stable")
        ranks[i, columns[order]] = (np.arange(columns.size) + 0.5) / columns.size
    return ranks.reshape(values.shape)


def make_nonlinear_importance(qoi_fn, num_iterations=8):

    def to_real(shift):
        if jnp.iscomplexobj(shift):
            return jnp.stack([shift.real, shift.imag])
        return jnp.reshape(shift, (1,))

    def to_shift(v, like):
        if jnp.iscomplexobj(like):
            return (v[0] + 1j * v[1]).astype(like.dtype)
        return v[0].astype(like.dtype)

    def nonlinear_importance(params, layer_index, weight_index, shift_guess, x):
        q_original = jax.vmap(lambda xs: qoi_fn(params, xs))(x)
        p = params[layer_index]
        bias_index = weight_index[0]

        def residuals(v):
            new_layer = {
                "W": p["W"].at[weight_index].set(0.0),
                "b": p["b"].at[bias_index].add(to_shift(v, p["b"])),
            }
            new_params = params[:layer_index] + [new_layer] + params[layer_index + 1 :]
            return jax.vmap(lambda xs: qoi_fn(new_params, xs))(x) - q_original

        def discrepancy(v):
            return jnp.sum(residuals(v) ** 2) / x.shape[0]

        def gauss_newton_step(v, _):
            jac = jax.jacfwd(residuals)(v)
            res = residuals(v)
            gram = jnp.einsum("smi,smj->ij", jac, jac)
            rhs = -jnp.einsum("smi,sm->i", jac, res)
            gram = gram + 1e-6 * jnp.trace(gram) * jnp.eye(v.size) + 1e-30 * jnp.eye(v.size)
            v_new = v + jnp.linalg.solve(gram, rhs)
            v_new = jnp.where(discrepancy(v_new) < discrepancy(v), v_new, v)
            return v_new, None

        v_guess = to_real(shift_guess)
        v_optimal, _ = jax.lax.scan(gauss_newton_step, v_guess, None, length=num_iterations)
        return {
            "no_compensation": discrepancy(jnp.zeros_like(v_guess)),
            "guess": discrepancy(v_guess),
            "optimal": discrepancy(v_optimal),
            "optimal_shift": to_shift(v_optimal, p["b"]),
        }

    return nonlinear_importance
