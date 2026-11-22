import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import networks as nets

arguments = experiment.positional_arguments()
width = int(arguments[0]) if len(arguments) > 0 else 32
NUM_HIDDEN_LAYERS = 4
NUM_QUADRATURE_NODES = 32
HALF_SIDE = 0.25


def make_spec(width_factor):
    hidden = max(2, 2 * round(width_factor * width / 2))
    widths = [2] + [hidden] * NUM_HIDDEN_LAYERS + [1]
    spec = []
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        is_last = i == len(widths) - 2
        m_in = n_in // 2
        m_out = n_out if is_last else n_out // 2
        structure = np.ones((n_out, n_in), dtype=np.float32)
        structure[m_out:, :m_in] = 0.0
        activation = "identity" if is_last else (m_out, "exp", "cprelu")
        spec.append(nets.dense(n_in, n_out, activation, structure=structure))
    return spec


theta_nodes = (jnp.arange(NUM_QUADRATURE_NODES) + 0.5) * 2.0 * jnp.pi / NUM_QUADRATURE_NODES
quadrature_weight = 2.0 * jnp.pi / NUM_QUADRATURE_NODES


def network_inputs_at(point):
    x, y, z = point
    zeta = jnp.sin(theta_nodes) * y + jnp.cos(theta_nodes) * z + 1j * x
    return jnp.stack([zeta, (theta_nodes / jnp.pi - 1.0) + 0j], axis=1)


def make_predict(spec):
    def predict(params, point):
        outputs = jax.vmap(lambda inputs: nets.forward(spec, params, inputs)[0])(network_inputs_at(point))
        return jnp.reshape(quadrature_weight * jnp.sum(jnp.real(outputs)), (1,))

    return predict


def reference_solution(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.exp(np.sqrt(13.0) * x) * np.cos(2.0 * y) * np.cos(3.0 * z)


def points_on_cube_boundary(num_points, rng):
    points = rng.uniform(-HALF_SIDE, HALF_SIDE, size=(num_points, 3))
    axis = rng.integers(0, 3, size=num_points)
    side = rng.choice([-HALF_SIDE, HALF_SIDE], size=num_points)
    points[np.arange(num_points), axis] = side
    return points.astype(np.float32)


def load_data():
    rng = np.random.default_rng(0)
    x_train = points_on_cube_boundary(2400, rng)
    x_test = points_on_cube_boundary(600, rng)
    return dict(
        x_train=x_train, y_train=reference_solution(x_train)[:, None].astype(np.float32),
        x_test=x_test, y_test=reference_solution(x_test)[:, None].astype(np.float32),
    )


case = experiment.Case(
    name=f"laplace_holomorphic_{width}",
    task="regression",
    make_spec=make_spec,
    load_data=load_data,
    make_predict=make_predict,
    network_inputs=lambda points: jax.vmap(network_inputs_at)(points).reshape(-1, 2),
    is_complex=True,
    init_gain=0.5,
    seeds=[0, 1, 2, 3, 4],
    ratios=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
    epochs_baseline=3000,
    epochs_fine=1500,
    batch_size=240,
    learning_rate_baseline=lambda epoch: 1e-3 if epoch < 1500 else (3e-4 if epoch < 2500 else 1e-4),
    learning_rate_fine=lambda epoch: 1e-4,
    calibration_size=2400,
    moments_chunk_size=400,
    joint_chunk_size=400,
)

case.methods = case.methods + ("ec_joint", "ec_progressive_independent")

if __name__ == "__main__":
    experiment.main(case, pathlib.Path(__file__).resolve().parent / "results")
