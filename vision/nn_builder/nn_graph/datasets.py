"""Dataset generators: 2D toy datasets matching TF Playground's algorithms,
an MNIST loader for CNN demos, and a synthetic sequence task for RNN/Transformer demos.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

TOY_2D_KINDS = {"circle", "xor", "gaussian", "spiral"}
SEQUENCE_KINDS = {"sequence_copy", "sequence_classify"}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")


def _circle(num_samples, noise, rng):
    radius = 5.0
    n = num_samples // 2
    X, y = [], []

    def label_for(x, y_):
        return 1 if (x * x + y_ * y_) < (radius * 0.5) ** 2 else 0

    for _ in range(n):
        r = rng.uniform(0, radius * 0.5)
        angle = rng.uniform(0, 2 * np.pi)
        x_ = r * np.sin(angle)
        y_ = r * np.cos(angle)
        nx = rng.uniform(-radius, radius) * noise / 3
        ny = rng.uniform(-radius, radius) * noise / 3
        X.append([x_, y_])
        y.append(label_for(x_ + nx, y_ + ny))
    for _ in range(num_samples - n):
        r = rng.uniform(radius * 0.7, radius)
        angle = rng.uniform(0, 2 * np.pi)
        x_ = r * np.sin(angle)
        y_ = r * np.cos(angle)
        nx = rng.uniform(-radius, radius) * noise / 3
        ny = rng.uniform(-radius, radius) * noise / 3
        X.append([x_, y_])
        y.append(label_for(x_ + nx, y_ + ny))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _xor(num_samples, noise, rng):
    X, y = [], []
    padding = 0.3
    for _ in range(num_samples):
        x_ = rng.uniform(-5, 5)
        x_ += padding if x_ > 0 else -padding
        y_ = rng.uniform(-5, 5)
        y_ += padding if y_ > 0 else -padding
        nx = rng.uniform(-5, 5) * noise
        ny = rng.uniform(-5, 5) * noise
        label = 1 if (x_ + nx) * (y_ + ny) >= 0 else 0
        X.append([x_, y_])
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _gaussian(num_samples, noise, rng):
    variance = np.interp(noise, [0, 0.5], [0.5, 4.0])
    n = num_samples // 2
    X, y = [], []
    for cx, cy, label in ((2, 2, 1), (-2, -2, 0)):
        pts = rng.normal(loc=[cx, cy], scale=variance, size=(n, 2))
        X.append(pts)
        y.append(np.full(n, label, dtype=np.int64))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def _spiral(num_samples, noise, rng):
    n = num_samples // 2
    X, y = [], []
    for delta_t, label in ((0.0, 1), (np.pi, 0)):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x_ = r * np.sin(t) + rng.uniform(-1, 1) * noise
            y_ = r * np.cos(t) + rng.uniform(-1, 1) * noise
            X.append([x_, y_])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


_TOY_2D_FNS = {"circle": _circle, "xor": _xor, "gaussian": _gaussian, "spiral": _spiral}


def make_2d_dataset(kind, num_samples, noise, seed=0):
    rng = np.random.RandomState(seed)
    return _TOY_2D_FNS[kind](num_samples, noise, rng)


def make_sequence_dataset(kind, num_samples, seq_len, vocab_size, seed=0):
    rng = np.random.RandomState(seed)
    vocab_size = max(2, vocab_size)
    X = rng.randint(0, vocab_size, size=(num_samples, seq_len)).astype(np.int64)
    if kind == "sequence_copy":
        y = X.copy()
    else:  # sequence_classify: majority-parity label
        odd_count = (X % 2).sum(axis=1)
        y = (odd_count > (seq_len / 2)).astype(np.int64)
    return X, y


class DatasetBundle:
    def __init__(self, train_loader, val_loader, num_classes, task, x_range=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.task = task  # "2d" | "image" | "sequence_copy" | "sequence_classify"
        self.x_range = x_range  # (min, max) for 2D decision-boundary sampling


def build_dataset(params):
    kind = params.get("kind", "circle")
    batch_size = int(params.get("batch_size", 32))
    train_ratio = float(params.get("train_ratio", 0.5))
    seed = int(params.get("seed", 0))

    if kind in TOY_2D_KINDS:
        num_samples = int(params.get("num_samples", 500))
        noise = float(params.get("noise", 0.0))
        X, y = make_2d_dataset(kind, num_samples, noise, seed)
        return _split_tensor_dataset(X, y, batch_size, train_ratio, num_classes=2, task="2d", x_range=(-6, 6), seed=seed)

    if kind == "mnist":
        from torchvision import datasets, transforms

        os.makedirs(CACHE_DIR, exist_ok=True)
        full = datasets.MNIST(root=CACHE_DIR, train=True, download=True, transform=transforms.ToTensor())
        num_samples = min(int(params.get("num_samples", 2000)), len(full))
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(full), size=num_samples, replace=False)
        n_train = int(num_samples * train_ratio)
        train_ds = Subset(full, idx[:n_train].tolist())
        val_ds = Subset(full, idx[n_train:].tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None
        return DatasetBundle(train_loader, val_loader, num_classes=10, task="image")

    if kind in SEQUENCE_KINDS:
        num_samples = int(params.get("num_samples", 500))
        seq_len = int(params.get("seq_len", 16))
        vocab_size = int(params.get("vocab_size", 10))
        X, y = make_sequence_dataset(kind, num_samples, seq_len, vocab_size, seed)
        X_t = torch.from_numpy(X).long()
        y_t = torch.from_numpy(y).long()
        n_train = int(num_samples * train_ratio)
        train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
        val_ds = TensorDataset(X_t[n_train:], y_t[n_train:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None
        num_classes = vocab_size if kind == "sequence_copy" else 2
        return DatasetBundle(train_loader, val_loader, num_classes=num_classes, task=kind)

    raise ValueError(f"Unknown dataset kind '{kind}'")


def _split_tensor_dataset(X, y, batch_size, train_ratio, num_classes, task, x_range=None, seed=0):
    # X/y are generated class-contiguous (e.g. spiral/gaussian: all class 1 first,
    # then all class 0) — shuffle before splitting or the held-out set can end up
    # single-class, which breaks accuracy/confusion-matrix/ROC-AUC on it.
    n = len(X)
    perm = np.random.RandomState(seed).permutation(n)
    X, y = X[perm], y[perm]
    n_train = int(n * train_ratio)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:], y_t[n_train:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None
    return DatasetBundle(train_loader, val_loader, num_classes, task, x_range)
