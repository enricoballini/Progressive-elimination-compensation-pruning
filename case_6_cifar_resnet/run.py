import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

import datasets
import experiment
from ecp import networks as nets
from ecp import training

BLOCKS_PER_STAGE = 3
EPSILON = 1e-5
EPOCHS = 80
LEARNING_RATE = lambda epoch: 1e-3 if epoch < 60 else 1e-4


def make_spec(width_factor):
    widths = [max(1, round(width_factor * c)) for c in (16, 32, 64)]
    spec = [nets.conv(3, widths[0], (3, 3), "relu")]
    c_in = widths[0]
    for stage, c in enumerate(widths):
        for block in range(BLOCKS_PER_STAGE):
            stride = 2 if stage > 0 and block == 0 else 1
            spec += [
                nets.save_input(),
                nets.conv(c_in, c, (3, 3), "relu", stride=stride),
                nets.conv(c, c, (3, 3)),
                nets.add_shortcut("relu"),
            ]
            c_in = c
    return spec + [nets.global_avg_pool(), nets.dense(c_in, 10)]


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.load_cifar10()
    return dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def augment(key, x):
    key_flip, key_crop = jax.random.split(key)
    flip = jax.random.bernoulli(key_flip, 0.5, (x.shape[0],))
    x = jnp.where(flip[:, None, None, None], x[..., ::-1], x)
    padded = jnp.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)))
    corners = jax.random.randint(key_crop, (x.shape[0], 2), 0, 9)
    crop = lambda image, corner: jax.lax.dynamic_slice(image, (0, corner[0], corner[1]), x.shape[1:])
    return jax.vmap(crop)(padded, corners)


def forward_batch_norm(spec, params, norm, x, statistics=None):
    z, kept = x, None
    used = {}
    for index, (layer, p) in enumerate(zip(spec, params)):
        if layer["kind"] != "conv":
            z, kept = jax.vmap(lambda zs, ks: nets.apply_layer(layer, p, zs, ks))(z, kept)
            continue
        a = jax.vmap(lambda zs: nets.affine_map(layer, p, zs))(z)
        axes = (0, 2, 3)
        mean, variance = (a.mean(axes), a.var(axes)) if statistics is None else statistics[index]
        used[index] = (mean, variance)
        shape = (1, -1, 1, 1)
        a = (a - mean.reshape(shape)) / jnp.sqrt(variance.reshape(shape) + EPSILON)
        a = a * norm[index]["scale"].reshape(shape) + norm[index]["shift"].reshape(shape)
        z = nets.apply_activation(layer["activation"], a)
    return z, used


def fold(spec, params, norm, statistics):
    folded = []
    for index, (layer, p) in enumerate(zip(spec, params)):
        if layer["kind"] != "conv":
            folded.append(p)
            continue
        mean, variance = statistics[index]
        factor = norm[index]["scale"] / jnp.sqrt(variance + EPSILON)
        folded.append({
            "W": p["W"] * factor[:, None, None, None],
            "b": (p["b"] - mean) * factor + norm[index]["shift"],
        })
    return folded


def train_with_batch_norm(spec, key_init, key_train, x_train, y_train, batch_size=128, epochs=EPOCHS):
    params = nets.init_params(spec, key_init)
    norm = {
        index: {"scale": jnp.ones(layer["c_out"]), "shift": jnp.zeros(layer["c_out"])}
        for index, layer in enumerate(spec) if layer["kind"] == "conv"
    }
    trainable = {"params": params, "norm": norm}

    def loss(trainable, x, y):
        logits, _ = forward_batch_norm(spec, trainable["params"], trainable["norm"], x)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])

    @jax.jit
    def train_epoch(trainable, opt_state, x, y, key, lr):
        num_batches = x.shape[0] // batch_size
        key_permutation, key_augment = jax.random.split(key)
        batches = jax.random.permutation(key_permutation, x.shape[0])[: num_batches * batch_size]
        batches = batches.reshape(num_batches, batch_size)
        batch_keys = jax.random.split(key_augment, num_batches)

        def step(carry, batch):
            trainable, opt_state = carry
            indices, batch_key = batch
            grads = jax.grad(loss)(trainable, augment(batch_key, x[indices]), y[indices])
            return training.adam_update(trainable, grads, opt_state, lr), None

        (trainable, opt_state), _ = jax.lax.scan(step, (trainable, opt_state), (batches, batch_keys))
        return trainable, opt_state

    opt_state = training.adam_init(trainable)
    for epoch in range(epochs):
        key_train, key_epoch = jax.random.split(key_train)
        trainable, opt_state = train_epoch(trainable, opt_state, x_train, y_train, key_epoch, LEARNING_RATE(epoch))

    batch_statistics = jax.jit(lambda t, x: forward_batch_norm(spec, t["params"], t["norm"], x)[1])
    sums = None
    chunk = 1000
    for start in range(0, x_train.shape[0], chunk):
        used = batch_statistics(trainable, x_train[start : start + chunk])
        moments = {i: (m, v + m**2) for i, (m, v) in used.items()}
        sums = moments if sums is None else {i: (sums[i][0] + m[0], sums[i][1] + m[1]) for i, m in moments.items()}
    num_chunks = -(-x_train.shape[0] // chunk)
    statistics = {i: (s / num_chunks, q / num_chunks - (s / num_chunks) ** 2) for i, (s, q) in sums.items()}
    return fold(spec, trainable["params"], trainable["norm"], statistics)


case = experiment.Case(
    name="cifar_resnet20",
    task="classification",
    make_spec=make_spec,
    load_data=load_data,
    seeds=[0, 1, 2],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    epochs_baseline=EPOCHS,
    epochs_fine=5,
    batch_size=128,
    learning_rate_baseline=LEARNING_RATE,
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=5000,
    moments_chunk_size=10,
    joint_chunk_size=50,
    gradient_chunk_size=250,
    augment=augment,
    train_dense=train_with_batch_norm,
    with_smaller_networks=False,
)

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
