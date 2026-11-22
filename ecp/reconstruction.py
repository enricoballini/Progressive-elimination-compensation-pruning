import numpy as np
import jax
import jax.numpy as jnp

from . import networks

RELATIVE_DAMPING = 0.01


def patches(layer, z):
    kernel = layer["kernel"]
    padding = layer["padding"]
    if padding == "CIRCULAR":
        pad = [(0, 0)] + [((k - 1) // 2, k // 2) for k in kernel]
        z = jnp.pad(z, pad, mode="wrap")
        padding = "VALID"
    strides = (layer["stride"],) * len(kernel)

    def real_patches(u):
        return jax.lax.conv_general_dilated_patches(u[None], kernel, strides, padding)[0]

    if jnp.iscomplexobj(z):
        result = real_patches(z.real) + 1j * real_patches(z.imag)
    else:
        result = real_patches(z)
    return result.reshape(result.shape[0], -1)


def input_matrix(layer, z):
    if layer["kind"] == "dense":
        columns = z[:, None]
    else:
        columns = patches(layer, z)
    ones = jnp.ones((1, columns.shape[1]), dtype=columns.dtype)
    return jnp.concatenate([columns, ones], axis=0)


def weight_matrix(p):
    W = p["W"].reshape(p["W"].shape[0], -1)
    return jnp.concatenate([W, p["b"][:, None]], axis=1)


def split_weight_matrix(matrix, p):
    return {"W": matrix[:, :-1].reshape(p["W"].shape), "b": matrix[:, -1]}


def make_hessian(spec, layer_index, chunk_size):
    layer = spec[layer_index]

    def chunk_hessian(params, x_chunk):
        def inputs_of_sample(x_sample):
            z = networks.forward_with_layer_inputs(spec, params, x_sample)[1][layer_index]
            return input_matrix(layer, z)

        X = jax.vmap(inputs_of_sample)(x_chunk)
        return jnp.einsum("sdp,sep->de", X, jnp.conj(X))

    chunk_hessian = jax.jit(chunk_hessian)

    def hessian(params, x):
        total = None
        for start in range(0, x.shape[0], chunk_size):
            part = chunk_hessian(params, x[start : start + chunk_size])
            total = part if total is None else total + part
        return total

    return hessian


def damped(H, relative_damping=RELATIVE_DAMPING):
    damping = relative_damping * jnp.mean(jnp.real(jnp.diagonal(H)))
    return H + damping * jnp.eye(H.shape[0], dtype=H.dtype)


def restricted_inverses(H, exists):
    H = np.asarray(H, dtype=np.complex128 if np.iscomplexobj(H) else np.float64)
    patterns, row_pattern = np.unique(exists, axis=0, return_inverse=True)
    inverses = []
    for pattern in patterns:
        keep = pattern.astype(H.dtype)
        A = keep[:, None] * H * keep[None, :] + np.diag(1.0 - keep)
        inverses.append(np.linalg.inv(A))
    return np.stack(inverses)[np.ravel(row_pattern)]


@jax.jit
def removal_traces(W, H_inverse, removable):
    rows, columns = W.shape
    row_index = jnp.arange(rows)

    def step(carry, _):
        w, h_inv, removed = carry
        diagonal = jnp.maximum(jnp.real(jnp.diagonal(h_inv, axis1=1, axis2=2)), 1e-30)
        candidate = removable & ~removed
        loss = jnp.where(candidate, jnp.abs(w) ** 2 / diagonal, jnp.inf)
        q = jnp.argmin(loss, axis=1)
        loss_q = loss[row_index, q]
        active = jnp.isfinite(loss_q)

        h_row = h_inv[row_index, q, :]
        h_column = h_inv[row_index, :, q]
        h_qq = diagonal[row_index, q]
        factor = jnp.where(active, w[row_index, q] / h_qq, 0.0)
        w = w - factor[:, None] * h_row
        w = w.at[row_index, q].set(jnp.where(active, 0.0, w[row_index, q]))
        downdate = h_column[:, :, None] * h_row[:, None, :] / h_qq[:, None, None]
        h_inv = h_inv - jnp.where(active[:, None, None], downdate, 0.0)
        removed = removed.at[row_index, q].set(removed[row_index, q] | active)
        return (w, h_inv, removed), (q, loss_q)

    initial = (W, H_inverse, jnp.zeros_like(removable))
    _, (order, losses) = jax.lax.scan(step, initial, None, length=columns)
    return order.T, losses.T


def choose_removals(order, losses, num_removed):
    order, losses = np.asarray(order), np.asarray(losses)
    monotone = np.maximum.accumulate(losses, axis=1)
    chosen = np.argsort(monotone, axis=None, kind="stable")[:num_removed]
    count = np.bincount(np.unravel_index(chosen, losses.shape)[0], minlength=losses.shape[0])
    removed = np.zeros(losses.shape, dtype=bool)
    for i, k in enumerate(count):
        removed[i, order[i, :k]] = True
    return removed


@jax.jit
def least_squares_rows(W, H, kept):
    keep = kept.astype(H.dtype)
    eye = jnp.eye(H.shape[0], dtype=H.dtype)
    A = keep[:, :, None] * H[None] * keep[:, None, :] + (1 - keep)[:, :, None] * eye[None]
    rhs = keep * (W @ H)
    return jnp.linalg.solve(jnp.conj(A), rhs[:, :, None])[:, :, 0]


class OptimalBrainCompression:

    def __init__(self, spec, chunk_size=256, relative_damping=RELATIVE_DAMPING):
        self.spec = spec
        self.relative_damping = relative_damping
        self.weighted_layers = [i for i, layer in enumerate(spec) if networks.has_weights(layer)]
        self.hessians = {i: make_hessian(spec, i, chunk_size) for i in self.weighted_layers}

    def prune(self, params, existing, x, layer_ratios):
        current = list(params)
        masks = [None] * len(params)
        for i in self.weighted_layers:
            p = current[i]
            H = damped(self.hessians[i](current, x), self.relative_damping)
            W = weight_matrix(p)
            exists = np.asarray(existing[i]).reshape(W.shape[0], -1) > 0
            bias_column = np.ones((W.shape[0], 1), dtype=bool)
            exists_with_bias = np.concatenate([exists, bias_column], axis=1)
            removable = np.concatenate([exists, ~bias_column], axis=1)
            num_removed = int(round(layer_ratios[i] * exists.sum()))

            H_inverse = jnp.asarray(restricted_inverses(H, exists_with_bias), dtype=H.dtype)
            order, losses = removal_traces(W, H_inverse, jnp.asarray(removable))
            removed = choose_removals(order, losses, num_removed)
            kept = ~removed & exists_with_bias
            W_new = least_squares_rows(W, H, jnp.asarray(kept))

            current[i] = split_weight_matrix(W_new, p)
            masks[i] = jnp.asarray(kept[:, :-1].reshape(p["W"].shape), dtype=jnp.float32)
        return masks, current


def uniform_ratios(spec, ratio):
    return [ratio if networks.has_weights(layer) else None for layer in spec]


def ratios_of_masks(masks, existing):
    return [
        None if mask is None else 1.0 - float(mask.sum()) / float(ex.sum())
        for mask, ex in zip(masks, existing)
    ]
