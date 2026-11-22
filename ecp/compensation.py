import numpy as np
import jax
import jax.numpy as jnp

from . import networks
from .importance import num_samples_of, sum_over_chunks


def real_vector_to_bias_shifts(vector, params):
    shifts, start = [], 0
    for p in params:
        if not p:
            shifts.append(0.0)
            continue
        n = p["b"].size
        real = vector[start : start + n]
        start += n
        if jnp.iscomplexobj(p["b"]):
            imag = vector[start : start + n]
            start += n
            shifts.append((real + 1j * imag).astype(p["b"].dtype))
        else:
            shifts.append(real.astype(p["b"].dtype))
    return shifts


def add_bias_shifts(params, shifts):
    return [{"W": p["W"], "b": p["b"] + s} if p else {} for p, s in zip(params, shifts)]


def bias_jacobian(qoi_fn, params, x_chunk):

    def qoi_of_shifts(shifts, x):
        return qoi_fn(add_bias_shifts(params, shifts), x)

    zero_shifts = [jnp.zeros_like(p["b"]) if p else 0.0 for p in params]
    jac = jax.vmap(jax.jacrev(qoi_of_shifts), in_axes=(None, 0))(zero_shifts, x_chunk)

    columns = []
    for p, j in zip(params, jac):
        if not p:
            continue
        columns.append(j.real)
        if jnp.iscomplexobj(j):
            columns.append(-j.imag)
    return jnp.concatenate(columns, axis=-1)


def weight_effect(qoi_fn, params, direction, x_sample):

    def qoi_of_weights(weights):
        new_params = [{"W": w, "b": p["b"]} if p else {} for p, w in zip(params, weights)]
        return qoi_fn(new_params, x_sample)

    weights = [p["W"] if p else jnp.zeros(()) for p in params]
    return jax.jvp(qoi_of_weights, (weights,), (direction,))[1]


def solve_damped(gram, rhs, relative_damping):
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    damping = relative_damping * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram + damping * np.eye(gram.shape[0]), rhs)


class JointCompensation:

    def __init__(self, qoi_fn, chunk_size=256):
        self.chunk_size = chunk_size

        def linear_normal_equations(params, direction, x_chunk):
            J_b = bias_jacobian(qoi_fn, params, x_chunk)
            t = jax.vmap(lambda xs: weight_effect(qoi_fn, params, direction, xs))(x_chunk)
            return jnp.einsum("smi,smj->ij", J_b, J_b), jnp.einsum("smi,sm->i", J_b, t)

        def residual(params_pruned, params_original, x_chunk):
            return jax.vmap(lambda xs: qoi_fn(params_pruned, xs) - qoi_fn(params_original, xs))(x_chunk)

        def gauss_newton_normal_equations(params_pruned, params_original, x_chunk):
            J_b = bias_jacobian(qoi_fn, params_pruned, x_chunk)
            res = residual(params_pruned, params_original, x_chunk)
            return jnp.einsum("smi,smj->ij", J_b, J_b), -jnp.einsum("smi,sm->i", J_b, res)

        def squared_residual(params_pruned, params_original, x_chunk):
            return jnp.sum(residual(params_pruned, params_original, x_chunk) ** 2)

        self._linear_normal_equations = jax.jit(linear_normal_equations)
        self._gauss_newton_normal_equations = jax.jit(gauss_newton_normal_equations)
        self._squared_residual = jax.jit(squared_residual)
        self._squared_residual_gradient = jax.jit(jax.grad(squared_residual))

    def _sum_over_chunks(self, chunk_fn, x):
        return sum_over_chunks(chunk_fn, x, self.chunk_size)

    def discrepancy_gradient(self, params_pruned, params_original, x):
        total = self._sum_over_chunks(
            lambda xc: self._squared_residual_gradient(params_pruned, params_original, xc), x
        )
        return jax.tree_util.tree_map(lambda g: g / num_samples_of(x), total)

    def discrepancy(self, params_pruned, params_original, x):
        total = self._sum_over_chunks(
            lambda xc: self._squared_residual(params_pruned, params_original, xc), x
        )
        return float(total) / num_samples_of(x)

    def linear(self, params, removed, x, relative_damping=1e-6):
        direction = [r if r is not None else jnp.zeros(()) for r in removed]
        gram, rhs = self._sum_over_chunks(
            lambda xc: self._linear_normal_equations(params, direction, xc), x
        )
        u = solve_damped(gram, rhs, relative_damping)
        return real_vector_to_bias_shifts(jnp.asarray(u), params)

    def gauss_newton(self, params_pruned, params_original, x, num_iterations=3, relative_damping=1e-3):
        params_current = params_pruned
        discrepancy = self.discrepancy(params_current, params_original, x)

        for _ in range(num_iterations):
            gram, rhs = self._sum_over_chunks(
                lambda xc: self._gauss_newton_normal_equations(params_current, params_original, xc), x
            )
            u = solve_damped(gram, rhs, relative_damping)

            step_length, improved = 1.0, False
            while step_length > 1e-2 and not improved:
                shifts = real_vector_to_bias_shifts(jnp.asarray(step_length * u), params_current)
                params_trial = add_bias_shifts(params_current, shifts)
                discrepancy_trial = self.discrepancy(params_trial, params_original, x)
                improved = discrepancy_trial < discrepancy
                step_length *= 0.5

            if not improved:
                break
            params_current, discrepancy = params_trial, discrepancy_trial

        return params_current


class LayerwiseMeanCorrection:

    def __init__(self, spec, chunk_size=256):
        self.spec = spec
        self.chunk_size = chunk_size
        self.weighted_layers = [i for i, layer in enumerate(spec) if networks.has_weights(layer)]
        self._sum_of_pre_activations = {i: jax.jit(self._make_sum(i)) for i in self.weighted_layers}

    def _make_sum(self, index):
        spec = self.spec

        def mean_over_positions(params, x):
            z, kept = x, None
            for layer, p in zip(spec[:index], params[:index]):
                z, kept = networks.apply_layer(layer, p, z, kept)
            a = networks.affine_map(spec[index], params[index], z)
            return a.reshape(a.shape[0], -1).mean(axis=1)

        def sum_over_samples(params, x_chunk):
            return jnp.sum(jax.vmap(lambda xs: mean_over_positions(params, xs))(x_chunk), axis=0)

        return sum_over_samples

    def mean_pre_activation(self, params, x, index):
        total = sum_over_chunks(lambda xc: self._sum_of_pre_activations[index](params, xc), x, self.chunk_size)
        return total / num_samples_of(x)

    def correct(self, params_pruned, params_original, x):
        current = list(params_pruned)
        for i in self.weighted_layers:
            shift = self.mean_pre_activation(params_original, x, i) - self.mean_pre_activation(current, x, i)
            current[i] = {"W": current[i]["W"], "b": current[i]["b"] + shift.astype(current[i]["b"].dtype)}
        return current
