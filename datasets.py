import gzip
import pathlib

import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_items = int.from_bytes(f.read(4), "big")
        if magic == 2049:
            return np.frombuffer(f.read(), dtype=np.uint8)
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)


def load_mnist(unit_interval=False):
    folder = DATA_DIR / "mnist"
    if unit_interval:
        scale = lambda images: images.astype(np.float32) / np.float32(255.0)
    else:
        scale = lambda images: images.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    x_train = scale(read_idx(folder / "train-images-idx3-ubyte.gz"))
    y_train = read_idx(folder / "train-labels-idx1-ubyte.gz")
    x_test = scale(read_idx(folder / "t10k-images-idx3-ubyte.gz"))
    y_test = read_idx(folder / "t10k-labels-idx1-ubyte.gz")
    return (x_train, y_train.astype(np.int32)), (x_test, y_test.astype(np.int32))


def load_cifar10():
    folder = DATA_DIR / "cifar10"

    def read_batch(name):
        raw = np.fromfile(folder / name, dtype=np.uint8).reshape(-1, 3073)
        images = raw[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        return images, raw[:, 0].astype(np.int32)

    batches = [read_batch(f"data_batch_{i}.bin") for i in range(1, 6)]
    x_train = np.concatenate([b[0] for b in batches])
    y_train = np.concatenate([b[1] for b in batches])
    x_test, y_test = read_batch("test_batch.bin")
    return (x_train, y_train), (x_test, y_test)


def constellations():
    qam_levels = lambda n: np.arange(-(n - 1), n, 2)

    def qam(n):
        levels = qam_levels(n)
        return (levels[:, None] + 1j * levels[None, :]).reshape(-1)

    alphabets = {
        "BPSK": np.array([-1.0, 1.0]) + 0j,
        "QPSK": np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))),
        "8PSK": np.exp(1j * np.pi / 4 * np.arange(8)),
        "16QAM": qam(4),
        "4PAM": qam_levels(4) + 0j,
    }
    return {name: a / np.sqrt(np.mean(np.abs(a) ** 2)) for name, a in alphabets.items()}


def root_raised_cosine(samples_per_symbol, span_symbols=8, rolloff=0.35):
    t = np.arange(-span_symbols * samples_per_symbol // 2, span_symbols * samples_per_symbol // 2 + 1)
    t = t / samples_per_symbol
    with np.errstate(divide="ignore", invalid="ignore"):
        taps = (
            np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(np.pi * t * (1 + rolloff))
        ) / (np.pi * t * (1 - (4 * rolloff * t) ** 2))
    taps[t == 0] = 1 - rolloff + 4 * rolloff / np.pi
    singular = np.isclose(np.abs(t), 1 / (4 * rolloff))
    taps[singular] = rolloff / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
    )
    return taps / np.sqrt(np.sum(taps**2))


def make_modulation_dataset(num_samples, seed, length=128, samples_per_symbol=4):
    rng = np.random.default_rng(seed)
    alphabets = list(constellations().values())
    taps = root_raised_cosine(samples_per_symbol)
    num_symbols = length // samples_per_symbol + 2 * 8 + 2

    x = np.zeros((num_samples, 1, length), dtype=np.complex64)
    y = rng.integers(0, len(alphabets), size=num_samples).astype(np.int32)
    time = np.arange(length)
    for i in range(num_samples):
        symbols = rng.choice(alphabets[y[i]], size=num_symbols)
        upsampled = np.zeros(num_symbols * samples_per_symbol, dtype=complex)
        upsampled[::samples_per_symbol] = symbols
        shaped = np.convolve(upsampled, taps, mode="same")
        start = 8 * samples_per_symbol + rng.integers(0, samples_per_symbol)
        signal = shaped[start : start + length]

        phase = rng.uniform(0, 2 * np.pi)
        frequency = rng.uniform(-0.005, 0.005)
        signal = signal * np.exp(1j * (2 * np.pi * frequency * time + phase))
        signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))

        snr_db = rng.uniform(0.0, 20.0)
        noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
        signal = signal + noise_std * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
        x[i, 0] = signal / np.sqrt(np.mean(np.abs(signal) ** 2))
    return x, y


def load_diffusion_sorption(num_trajectories=100, time_step=2, cell_step=16):
    import h5py

    time_indices = np.arange(0, 101, time_step)
    cell_indices = np.unique(np.append(np.arange(0, 1024, cell_step), 1023))
    with h5py.File(DATA_DIR / "diffusion_sorption" / "1D_diff-sorp_NA_NA.h5", "r") as f:
        data = np.stack([
            np.asarray(f[f"{i:04d}/data"], dtype=np.float32)[:, :, 0][time_indices][:, cell_indices]
            for i in range(num_trajectories)
        ])

    times = np.linspace(0.0, 1.0, len(time_indices), dtype=np.float32)[1:]
    cells = np.linspace(0.0, 1.0, len(cell_indices), dtype=np.float32)

    def inputs_and_outputs(trajectories):
        label = trajectories[:, 0, 0]
        grid = np.stack(np.meshgrid(label, times, cells, indexing="ij"), axis=-1).reshape(-1, 3)
        first_snapshot = np.repeat(trajectories[:, 1], len(times) * len(cells), axis=0)
        x = np.hstack([grid, first_snapshot]).astype(np.float32)
        y = trajectories[:, 1:].reshape(-1, 1).astype(np.float32)
        return x, y

    num_train = int(0.8 * num_trajectories)
    x_train, y_train = inputs_and_outputs(data[:num_train])
    x_test, y_test = inputs_and_outputs(data[num_train:])
    permutation = np.random.default_rng(0).permutation(x_train.shape[0])
    return (x_train[permutation], y_train[permutation]), (x_test, y_test)
