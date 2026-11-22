import pathlib
import pickle
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import experiment
from ecp import baselines, compensation, networks, procedures, reconstruction, training
from ecp import importance as imp
from study_ranking import moments_of_progressive_ec
from study_without_labels import CASES, load_case

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "results_studies" / "memory"
SEED = 0


def bytes_in_use():
    return jax.local_devices()[0].memory_stats()["bytes_in_use"]


def peak_bytes():
    return jax.local_devices()[0].memory_stats()["peak_bytes_in_use"]


def main(name, phase):
    script, arguments = CASES[name]
    case = load_case(script, arguments)
    raw = case.load_data()
    spec = case.make_spec(1.0)
    model = experiment.make_model(case, spec)
    trained_file = OUT_DIR / f"{name}_trained.pkl"
    reference_file = OUT_DIR / f"{name}_reference.npy"

    if phase == "train":
        x_train, y_train = jnp.asarray(raw["x_train"]), jnp.asarray(raw["y_train"])
        jax.block_until_ready((x_train, y_train))
        before = bytes_in_use()
        key_init, key_train, _, _ = jax.random.split(jax.random.PRNGKey(SEED), 4)
        train_epoch = training.make_train_epoch(model["loss"], case.batch_size, case.augment)
        t0 = time.time()
        params = experiment.train_dense_network(case, spec, train_epoch, x_train, y_train, key_init, key_train)
        jax.block_until_ready(params)
        elapsed = time.time() - t0
        working = peak_bytes() - before
        experiment.save_params(trained_file, params)
        if case.task == "classification":
            x_cal = x_train[: case.calibration_size]
            probabilities = [jax.nn.softmax(model["batched_predict"](params, x_cal[i : i + 250]))
                             for i in range(0, x_cal.shape[0], 250)]
            np.save(reference_file, np.asarray(jnp.concatenate(probabilities)))
    else:
        x_cal = jnp.asarray(raw["x_train"][: case.calibration_size])
        params = experiment.load_params(trained_file)
        existing = networks.existing_weights(spec, params)
        if case.task == "classification":
            samples = (x_cal, jnp.asarray(np.load(reference_file)))
        else:
            samples = x_cal
        jax.block_until_ready((samples, params))
        before = bytes_in_use()
        t0 = time.time()
        if phase == "ec":
            if case.factorised:
                moments = imp.make_second_moments_dense(spec, chunk_size=case.moments_chunk_size)
            else:
                moments = imp.make_second_moments(model["qoi_output"], case.moments_chunk_size)
            result = procedures.elimination_compensation(params, existing, moments, x_cal, case.ratios)
        elif phase == "ec_progressive":
            joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)
            result = procedures.progressive_elimination_compensation(
                params, existing, moments_of_progressive_ec(case, spec, model), samples, case.ratios,
                joint=joint, joint_samples=samples, steps_between_ratios=case.steps_between_ratios,
                gauss_newton_iterations=case.gauss_newton_iterations,
            )
        elif phase == "magnitude_joint":
            joint = compensation.JointCompensation(model["qoi_loss_metric"], case.joint_chunk_size)
            result = procedures.prune_with_scores(
                params, existing, baselines.magnitude(params), case.ratios, joint=joint, joint_samples=samples)
        elif phase == "obc":
            inputs = x_cal if case.network_inputs is None else case.network_inputs(x_cal)
            obc = reconstruction.OptimalBrainCompression(spec, chunk_size=case.joint_chunk_size)
            result = [obc.prune(params, existing, inputs, reconstruction.uniform_ratios(spec, r)) for r in case.ratios]
        else:
            raise ValueError(phase)
        jax.block_until_ready(result)
        elapsed = time.time() - t0
        working = peak_bytes() - before

    record = dict(
        case=name, phase=phase, working_bytes=working, data_bytes=before,
        host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, time=elapsed,
    )
    print(record, flush=True)
    with open(OUT_DIR / f"{name}_{phase}.pkl", "wb") as f:
        pickle.dump(record, f)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main(sys.argv[1], sys.argv[2])
