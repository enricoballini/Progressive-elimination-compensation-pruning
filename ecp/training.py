import jax
import jax.numpy as jnp

from .pruning import apply_masks

tree_map = jax.tree_util.tree_map


def adam_init(params):
    first_moment = tree_map(jnp.zeros_like, params)
    second_moment = tree_map(lambda p: jnp.zeros(p.shape, dtype=p.real.dtype), params)
    step = jnp.zeros((), dtype=jnp.int32)
    return first_moment, second_moment, step


def adam_update(params, grads, state, lr, beta_1=0.9, beta_2=0.999, eps=1e-8):
    first_moment, second_moment, step = state
    step = step + 1
    grads = tree_map(jnp.conj, grads)

    first_moment = tree_map(lambda m, g: beta_1 * m + (1 - beta_1) * g, first_moment, grads)
    second_moment = tree_map(
        lambda v, g: beta_2 * v + (1 - beta_2) * jnp.abs(g) ** 2, second_moment, grads
    )
    correction_1 = 1.0 - beta_1**step
    correction_2 = 1.0 - beta_2**step

    params = tree_map(
        lambda p, m, v: p - lr * (m / correction_1) / (jnp.sqrt(v / correction_2) + eps),
        params, first_moment, second_moment,
    )
    return params, (first_moment, second_moment, step)


def make_train_epoch(loss_fn, batch_size, augment=None):

    @jax.jit
    def train_epoch(params, opt_state, masks, x, y, key, lr):
        num_batches = x.shape[0] // batch_size
        permutation = jax.random.permutation(key, x.shape[0])
        batches = permutation[: num_batches * batch_size].reshape(num_batches, batch_size)
        batch_keys = jax.random.split(jax.random.fold_in(key, 1), num_batches)

        def step(carry, batch):
            params, opt_state = carry
            batch_indices, batch_key = batch
            x_batch = x[batch_indices]
            if augment is not None:
                x_batch = augment(batch_key, x_batch)
            grads = jax.grad(loss_fn)(params, x_batch, y[batch_indices])
            params, opt_state = adam_update(params, grads, opt_state, lr)
            params = apply_masks(params, masks)
            return (params, opt_state), None

        (params, opt_state), _ = jax.lax.scan(step, (params, opt_state), (batches, batch_keys))
        return params, opt_state

    return train_epoch


def train(params, masks, train_epoch, x, y, num_epochs, learning_rate, key):
    opt_state = adam_init(params)
    for epoch in range(num_epochs):
        key, epoch_key = jax.random.split(key)
        params, opt_state = train_epoch(
            params, opt_state, masks, x, y, epoch_key, learning_rate(epoch)
        )
    return params


def evaluate_in_chunks(metric_fn, params, x, y, chunk_size=1000):
    total = 0.0
    for start in range(0, x.shape[0], chunk_size):
        x_chunk, y_chunk = x[start : start + chunk_size], y[start : start + chunk_size]
        total += float(metric_fn(params, x_chunk, y_chunk)) * x_chunk.shape[0]
    return total / x.shape[0]
