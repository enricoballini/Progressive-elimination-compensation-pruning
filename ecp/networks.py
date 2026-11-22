import jax
import jax.numpy as jnp
import numpy as np


def prelu(a, slope=0.1):
    return jnp.where(a >= 0, a, slope * a)


def split(real_function):
    return lambda a: real_function(a.real) + 1j * real_function(a.imag)


def cardioid(a):
    modulus = jnp.sqrt(a.real**2 + a.imag**2 + 1e-12)
    return 0.5 * (1.0 + a.real / modulus) * a


def modrelu(a, offset=-0.1):
    modulus = jnp.sqrt(a.real**2 + a.imag**2 + 1e-12)
    return jnp.maximum(modulus + offset, 0.0) * a / modulus


ACTIVATIONS = {
    "identity": lambda a: a,
    "relu": lambda a: jnp.maximum(a, 0.0),
    "prelu": prelu,
    "tanh": jnp.tanh,
    "exp": jnp.exp,
    "sin": jnp.sin,
    "cprelu": split(prelu),
    "crelu": split(lambda a: jnp.maximum(a, 0.0)),
    "ctanh": split(jnp.tanh),
    "cardioid": cardioid,
    "modrelu": modrelu,
}


def apply_activation(activation, a):
    if isinstance(activation, str):
        return ACTIVATIONS[activation](a)
    m, name_1, name_2 = activation
    return jnp.concatenate([ACTIVATIONS[name_1](a[:m]), ACTIVATIONS[name_2](a[m:])])


def dense(n_in, n_out, activation="identity", structure=None):
    return dict(kind="dense", n_in=n_in, n_out=n_out, activation=activation, structure=structure)


def conv(c_in, c_out, kernel, activation="identity", stride=1, padding="SAME"):
    return dict(
        kind="conv", c_in=c_in, c_out=c_out, kernel=tuple(kernel),
        activation=activation, stride=stride, padding=padding,
    )


def avg_pool(window):
    return dict(kind="avg_pool", window=tuple(window))


def flatten():
    return dict(kind="flatten")


def global_avg_pool():
    return dict(kind="global_avg_pool")


def save_input():
    return dict(kind="save_input")


def add_shortcut(activation="identity"):
    return dict(kind="add_shortcut", activation=activation)


def has_weights(layer):
    return layer["kind"] in ("dense", "conv")


def convolution(z, W, stride, padding):
    num_spatial = W.ndim - 2
    if padding == "CIRCULAR":
        pad = [(0, 0)] + [((k - 1) // 2, k // 2) for k in W.shape[2:]]
        z = jnp.pad(z, pad, mode="wrap")
        padding = "VALID"
    out = jax.lax.conv_general_dilated(z[None], W, (stride,) * num_spatial, padding)
    return out[0]


def affine_map(layer, p, z):
    W = p["W"]
    if layer["kind"] == "dense":
        if layer["structure"] is not None:
            W = W * layer["structure"]
        return W @ z + p["b"]
    bias = p["b"].reshape((-1,) + (1,) * (W.ndim - 2))
    return convolution(z, W, layer["stride"], layer["padding"]) + bias


def shortcut(kept, shape):
    if kept.shape == shape:
        return kept
    step = kept.shape[1] // shape[1]
    kept = kept[(slice(None),) + (slice(None, None, step),) * (kept.ndim - 1)]
    extra = shape[0] - kept.shape[0]
    return jnp.pad(kept, [(extra // 2, extra - extra // 2)] + [(0, 0)] * (kept.ndim - 1))


def apply_layer(layer, p, z, kept=None):
    if has_weights(layer):
        return apply_activation(layer["activation"], affine_map(layer, p, z)), kept
    if layer["kind"] == "avg_pool":
        window = (1,) + layer["window"]
        summed = jax.lax.reduce_window(z, 0.0, jax.lax.add, window, window, "VALID")
        return summed / np.prod(layer["window"]), kept
    if layer["kind"] == "flatten":
        return z.reshape(-1), kept
    if layer["kind"] == "global_avg_pool":
        return z.reshape(z.shape[0], -1).mean(axis=1), kept
    if layer["kind"] == "save_input":
        return z, z
    if layer["kind"] == "add_shortcut":
        return apply_activation(layer["activation"], z + shortcut(kept, z.shape)), None
    raise ValueError(f"unknown layer kind {layer['kind']}")


def forward(spec, params, x):
    z, kept = x, None
    for layer, p in zip(spec, params):
        z, kept = apply_layer(layer, p, z, kept)
    return z


def forward_with_layer_inputs(spec, params, x):
    z, kept = x, None
    layer_inputs = []
    for layer, p in zip(spec, params):
        layer_inputs.append(z)
        z, kept = apply_layer(layer, p, z, kept)
    return z, layer_inputs


def forward_with_bias_shifts(spec, params, bias_shifts, x):
    z, kept = x, None
    for layer, p, shift in zip(spec, params, bias_shifts):
        if has_weights(layer):
            p = {"W": p["W"], "b": p["b"] + shift}
        z, kept = apply_layer(layer, p, z, kept)
    return z


def init_params(spec, key, is_complex=False, gain=2.0):
    params = []
    for layer in spec:
        if not has_weights(layer):
            params.append({})
            continue
        key, key_real, key_imag = jax.random.split(key, 3)
        if layer["kind"] == "dense":
            shape, fan_in = (layer["n_out"], layer["n_in"]), layer["n_in"]
        else:
            shape = (layer["c_out"], layer["c_in"]) + layer["kernel"]
            fan_in = layer["c_in"] * int(np.prod(layer["kernel"]))
        std = np.sqrt(gain / fan_in)
        if is_complex:
            W = (jax.random.normal(key_real, shape) + 1j * jax.random.normal(key_imag, shape))
            W = W * std / np.sqrt(2.0)
            b = jnp.zeros(shape[0], dtype=W.dtype)
        else:
            W = jax.random.normal(key_real, shape) * std
            b = jnp.zeros(shape[0])
        if layer["kind"] == "dense" and layer["structure"] is not None:
            W = W * layer["structure"]
        params.append({"W": W, "b": b})
    return params


def existing_weights(spec, params):
    existing = []
    for layer, p in zip(spec, params):
        if not has_weights(layer):
            existing.append(None)
        elif layer["kind"] == "dense" and layer["structure"] is not None:
            existing.append(jnp.asarray(layer["structure"], dtype=jnp.float32))
        else:
            existing.append(jnp.ones(p["W"].shape, dtype=jnp.float32))
    return existing


def num_weights(spec, params):
    return int(sum(e.sum() for e in existing_weights(spec, params) if e is not None))
