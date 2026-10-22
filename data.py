"""
Dataset loading for Synapse.

Starts with a toy 2D dataset (easy to visualize, trains instantly) so you
can get the engine working first. Swapping to real MNIST later is a
one-line change — see load_mnist() below.
"""

import numpy as np


def make_toy_dataset(n_samples=300, n_classes=3, seed=42):
    """
    Generates a simple 2D 'blobs' dataset — points clustered around
    n_classes centers. Good for sanity-checking the network and for
    plotting decision boundaries later.
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []

    centers = rng.uniform(-5, 5, size=(n_classes, 2))
    for class_idx, center in enumerate(centers):
        points = center + rng.normal(scale=1.0, size=(n_samples // n_classes, 2))
        X.append(points)
        y.extend([class_idx] * (n_samples // n_classes))

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def one_hot(y, n_classes):
    encoded = np.zeros((y.size, n_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(cache_path="mnist_cache.npz", url=MNIST_URL):
    """
    Loads real MNIST (70,000 handwritten digits, 28x28 pixels flattened to 784).
    Standard split: 60,000 train / 10,000 test.

    First run downloads one ~11MB file and caches it locally, so every later
    run loads instantly. Falls back to scikit-learn's (slower) downloader if
    the direct download fails.
    """
    import os
    if os.path.exists(cache_path):
        print("Loading MNIST from local cache...", flush=True)
        d = np.load(cache_path)
        return d["X_train"], d["X_test"], d["y_train"], d["y_test"]

    raw_path = "mnist_raw.npz"
    try:
        import urllib.request
        print("Downloading MNIST (~11MB)...", flush=True)
        urllib.request.urlretrieve(url, raw_path)
        raw = np.load(raw_path)
        X_train = raw["x_train"].reshape(-1, 784) / 255.0   # 28x28 -> 784, scale 0-1
        X_test = raw["x_test"].reshape(-1, 784) / 255.0
        y_train = raw["y_train"].astype(int)
        y_test = raw["y_test"].astype(int)
    except Exception as e:
        print(f"Direct download failed ({e}). Trying scikit-learn "
              f"(can take several minutes)...", flush=True)
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data / 255.0
        y = mnist.target.astype(int)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    np.savez_compressed(cache_path, X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)
    print("MNIST downloaded and cached.", flush=True)
    return X_train, X_test, y_train, y_test


def load_digits_small(seed=0):
    """
    Tiny 8x8 digits dataset that ships inside scikit-learn (no download).
    Same task as MNIST, just smaller -> useful for a quick smoke test.
    """
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data / 16.0, d.target
    idx = np.random.default_rng(seed).permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int(0.8 * len(y))
    return X[:cut], X[cut:], y[:cut], y[cut:]