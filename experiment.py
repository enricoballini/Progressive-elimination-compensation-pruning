import dataclasses
import pathlib
import pickle
import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ecp import baselines, compensation, networks, procedures, training
from ecp import importance as imp


@dataclasses.dataclass
class Case:
    name: str
    task: str
    make_spec: Callable
    load_data: Callable
    seeds: list
    ratios: list
    epochs_baseline: int
    epochs_fine: int
    batch_size: int
    learning_rate_baseline: Callable
    learning_rate_fine: Callable
    is_complex: bool = False
    init_gain: float = 2.0
    calibration_size: int = 10000
    make_predict: Optional[Callable] = None
    network_inputs: Optional[Callable] = None
    factorised: bool = False
    moments_chunk_size: int = 64
    joint_chunk_size: int = 256
    gradient_chunk_size: int = 2000
    methods: tuple = (
        "magnitude", "magnitude_joint", "gradient_magnitude", "random", "wanda", "obd",
        "ec", "ec_progressive",
    )
    steps_between_ratios: int = 2
    gauss_newton_iterations: int = 2
    augment: Optional[Callable] = None
    train_dense: Optional[Callable] = None
    with_smaller_networks: bool = True


def make_model(case, spec):
    if case.make_predict is None:
        predict = lambda params, x: imp.as_real_vector(networks.forward(spec, params, x))
    else:
        predict = case.make_predict(spec)
    batched_predict = jax.vmap(predict, in_axes=(None, 0))

    if case.task == "classification":

        def loss(params, x, y):
            log_probabilities = jax.nn.log_softmax(batched_predict(params, x))
            return -jnp.mean(log_probabilities[jnp.arange(y.shape[0]), y])

        def accuracy(params, x, y):
            return jnp.mean(jnp.argmax(batched_predict(params, x), axis=1) == y)

        metrics = {"loss": jax.jit(loss), "accuracy": jax.jit(accuracy)}
        qoi_loss_metric = lambda params, sample: imp.whitened_logits(predict(params, sample[0]), sample[1])
        output_to_qoi_loss_metric = lambda y, sample: imp.whitened_logits(imp.as_real_vector(y), sample[1])
    else:

        def loss(params, x, y):
            return jnp.mean(jnp.sum((batched_predict(params, x) - y) ** 2, axis=1))

        metrics = {"loss": jax.jit(loss)}
        qoi_loss_metric = predict
        output_to_qoi_loss_metric = None

    return dict(
        predict=predict, batched_predict=jax.jit(batched_predict), loss=loss, metrics=metrics,
        qoi_output=predict, qoi_loss_metric=qoi_loss_metric,
        output_to_qoi_loss_metric=output_to_qoi_loss_metric,
    )


def evaluate(metrics, params, x, y):
    return {name: training.evaluate_in_chunks(fn, params, x, y) for name, fn in metrics.items()}


def make_distillation(case, model):
    batched_predict = jax.vmap(model["predict"], in_axes=(None, 0))
    if case.task == "classification":

        def loss(params, x, reference_probabilities):
            log_probabilities = jax.nn.log_softmax(batched_predict(params, x))
            return -jnp.mean(jnp.sum(reference_probabilities * log_probabilities, axis=1))
    else:

        def loss(params, x, reference_outputs):
            return jnp.mean(jnp.sum((batched_predict(params, x) - reference_outputs) ** 2, axis=1))

    train_epoch = training.make_train_epoch(loss, case.batch_size, case.augment)

    def distill(params, masks, x, reference, key, num_train_samples):
        steps = case.epochs_fine * (num_train_samples // case.batch_size)
        epochs = max(1, round(steps / (x.shape[0] // case.batch_size)))
        return training.train(params, masks, train_epoch, x, reference, epochs, case.learning_rate_fine, key)

    return distill


def width_factor_for_ratio(case, ratio, num_weights_full):
    target = (1.0 - ratio) * num_weights_full
    low, high = 1e-3, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        spec = case.make_spec(middle)
        params = networks.init_params(spec, jax.random.PRNGKey(0), case.is_complex)
        if networks.num_weights(spec, params) > target:
            high = middle
        else:
            low = middle
    return high


def run(case, out_dir, add_methods=None):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = case.methods if add_methods is None else tuple(add_methods)

    data = {key: jnp.asarray(value) for key, value in case.load_data().items()}
    x_train, y_train, x_test, y_test = (data[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    x_cal, y_cal = x_train[: case.calibration_size], y_train[: case.calibration_size]

    spec = case.make_spec(1.0)
    model = make_model(case, spec)
    train_epoch = training.make_train_epoch(model["loss"], case.batch_size, case.augment)

    if case.factorised:
        moments_output = imp.make_second_moments_dense(spec, chunk_size=case.moments_chunk_size)
        moments_loss_metric = moments_output
        if case.task == "classification":
            moments_loss_metric = imp.make_second_moments_dense(
                spec, model["output_to_qoi_loss_metric"], chunk_size=case.moments_chunk_size
            )
    else:
        moments_output = imp.make_second_moments(model["qoi_output"], case.moments_chunk_size)
        moments_loss_metric = moments_output
        if case.task == "classification":
            moments_loss_metric = imp.make_second_moments(model["qoi_loss_metric"], case.moments_chunk_size)
    joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)

    params_example = networks.init_params(spec, jax.random.PRNGKey(0), case.is_complex, case.init_gain)
    existing = networks.existing_weights(spec, params_example)
    num_weights = networks.num_weights(spec, params_example)
    print(f"[{case.name}] |W| = {num_weights}, train samples = {x_train.shape[0]}", flush=True)

    results_file = out_dir / "results.pkl"
    num_seeds_done = 0
    if add_methods is None:
        small = smaller_dense_networks(case, num_weights) if case.with_smaller_networks else []
        results = dict(
            name=case.name, ratios=list(case.ratios), seeds=list(case.seeds), num_weights=num_weights,
            baseline=[], methods={method: dict(pruned=[], finetuned=[], time=[]) for method in methods},
            dense_small=dict(ratios=[s["ratio"] for s in small], metrics=[]),
        )
        if results_file.exists():
            with open(results_file, "rb") as f:
                stored = pickle.load(f)
            same_experiment = (
                stored["ratios"] == results["ratios"] and stored["seeds"] == results["seeds"]
                and set(stored["methods"]) == set(methods)
            )
            if same_experiment:
                results = stored
                num_seeds_done = len(stored["baseline"])
                print(f"resuming: {num_seeds_done} seed(s) already done", flush=True)
    else:
        with open(results_file, "rb") as f:
            results = pickle.load(f)
        assert results["ratios"] == list(case.ratios) and results["seeds"] == list(case.seeds)
        for method in methods:
            results["methods"][method] = dict(pruned=[], finetuned=[], time=[])

    for seed_index, seed in enumerate(case.seeds):
        if seed_index < num_seeds_done:
            continue
        t_seed = time.time()
        key_init, key_train, key_fine, _ = jax.random.split(jax.random.PRNGKey(seed), 4)

        trained_file = out_dir / f"trained_seed_{seed}.pkl"
        if trained_file.exists() and add_methods is None:
            params = load_params(trained_file)
        else:
            params = train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train)
            save_params(trained_file, params)
        baseline = evaluate(model["metrics"], params, x_test, y_test)
        print(f"\nseed {seed}: baseline {baseline}", flush=True)
        if add_methods is None:
            results["baseline"].append(baseline)
        else:
            print(f"         stored baseline {results['baseline'][seed_index]}", flush=True)

        samples_output = x_cal
        samples_loss_metric = x_cal
        if case.task == "classification":
            reference_probabilities = jax.nn.softmax(model["batched_predict"](params, x_cal))
            samples_loss_metric = (x_cal, reference_probabilities)

        def prune(method):
            if method == "ec":
                return procedures.elimination_compensation(
                    params, existing, moments_output, samples_output, case.ratios
                )
            if method == "ec_progressive":
                return procedures.progressive_elimination_compensation(
                    params, existing, moments_loss_metric, samples_loss_metric, case.ratios,
                    joint=joint, joint_samples=samples_loss_metric,
                    steps_between_ratios=case.steps_between_ratios,
                    gauss_newton_iterations=case.gauss_newton_iterations,
                )
            if method == "ec_loss_metric":
                return procedures.elimination_compensation(
                    params, existing, moments_loss_metric, samples_loss_metric, case.ratios
                )
            if method == "ec_joint":
                return procedures.elimination_compensation(
                    params, existing, moments_loss_metric, samples_loss_metric, case.ratios,
                    joint=joint, joint_samples=samples_loss_metric,
                )
            if method == "ec_progressive_independent":
                return procedures.progressive_elimination_compensation(
                    params, existing, moments_loss_metric, samples_loss_metric, case.ratios,
                    steps_between_ratios=case.steps_between_ratios,
                )
            if method == "magnitude_joint":
                return procedures.prune_with_scores(
                    params, existing, baselines.magnitude(params), case.ratios,
                    joint=joint, joint_samples=samples_loss_metric,
                )

            if method == "magnitude":
                scores = baselines.magnitude(params)
            elif method == "gradient_magnitude":
                scores = baselines.gradient_magnitude(model["loss"], params, x_cal, y_cal, case.gradient_chunk_size)
            elif method == "random":
                scores = baselines.random_scores(params, seed)
            elif method == "wanda":
                inputs = x_cal if case.network_inputs is None else case.network_inputs(x_cal)
                scores = baselines.wanda(spec, params, existing, inputs)
            elif method == "obd":
                scores = imp.importance_without_compensation(params, moments_output(params, samples_output))
            else:
                raise ValueError(method)
            return procedures.prune_with_scores(params, existing, scores, case.ratios)

        for method in methods:
            t0 = time.time()
            pruned = prune(method)
            elapsed = time.time() - t0

            after_pruning, after_finetuning = [], []
            for masks, params_pruned in pruned:
                after_pruning.append(evaluate(model["metrics"], params_pruned, x_test, y_test))
                params_fine = training.train(
                    params_pruned, masks, train_epoch, x_train, y_train,
                    case.epochs_fine, case.learning_rate_fine, key_fine,
                )
                after_finetuning.append(evaluate(model["metrics"], params_fine, x_test, y_test))

            results["methods"][method]["pruned"].append(after_pruning)
            results["methods"][method]["finetuned"].append(after_finetuning)
            results["methods"][method]["time"].append(elapsed)
            print(
                f"  {method:20s} {elapsed:6.1f}s  loss after pruning: "
                + " ".join(f"{m['loss']:.3e}" for m in after_pruning)
                + "\n" + " " * 30 + "after fine-tuning: "
                + " ".join(f"{m['loss']:.3e}" for m in after_finetuning),
                flush=True,
            )

        if add_methods is None and case.with_smaller_networks:
            small_metrics = []
            for s in small:
                params_small = networks.init_params(s["spec"], key_init, case.is_complex, case.init_gain)
                ones = networks.existing_weights(s["spec"], params_small)
                params_small = training.train(
                    params_small, ones, s["train_epoch"], x_train, y_train,
                    case.epochs_baseline, case.learning_rate_baseline, key_train,
                )
                params_small = training.train(
                    params_small, ones, s["train_epoch"], x_train, y_train,
                    case.epochs_fine, case.learning_rate_fine, key_fine,
                )
                small_metrics.append(evaluate(s["metrics"], params_small, x_test, y_test))
            results["dense_small"]["metrics"].append(small_metrics)
            print(f"  {'smaller dense':20s}          loss: " + " ".join(f"{m['loss']:.3e}" for m in small_metrics))
        print(f"  seed done in {time.time() - t_seed:.0f}s", flush=True)

        if add_methods is None:
            with open(results_file, "wb") as f:
                pickle.dump(results, f)

    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    return results


def train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train):
    if case.train_dense is not None:
        return case.train_dense(spec, key_init, key_train, x_train, y_train)
    params = networks.init_params(spec, key_init, case.is_complex, case.init_gain)
    existing = networks.existing_weights(spec, params)
    return training.train(
        params, existing, train_epoch, x_train, y_train,
        case.epochs_baseline, case.learning_rate_baseline, key_train,
    )


def save_params(path, params):
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)


def load_params(path):
    with open(path, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def smaller_dense_networks(case, num_weights):
    small = []
    for ratio in case.ratios:
        spec = case.make_spec(width_factor_for_ratio(case, ratio, num_weights))
        model = make_model(case, spec)
        params = networks.init_params(spec, jax.random.PRNGKey(0), case.is_complex)
        small.append(dict(
            spec=spec, metrics=model["metrics"],
            train_epoch=training.make_train_epoch(model["loss"], case.batch_size),
            ratio=1.0 - networks.num_weights(spec, params) / num_weights,
        ))
    return small


def positional_arguments():
    import sys

    arguments = sys.argv[1:]
    if "--add" in arguments:
        index = arguments.index("--add")
        arguments = arguments[:index] + arguments[index + 2 :]
    return [a for a in arguments if not a.startswith("--")]


def main(case, results_dir):
    import sys

    out_dir = pathlib.Path(results_dir) / case.name
    if "--add" in sys.argv:
        add_methods = sys.argv[sys.argv.index("--add") + 1].split(",")
        run(case, out_dir, add_methods=add_methods)
    else:
        run(case, out_dir)
