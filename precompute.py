""
Train every (activation, initialization) network ONCE and save the results,
so the Synapse app can show them instantly instead of training in the browser.

    python precompute.py            # real MNIST  -> precomputed/mnist.pkl
    python precompute.py --small    # tiny digits -> precomputed/small.pkl (seconds)

The settings are exactly the ones used in the report (64 hidden neurons,
20 epochs, learning rate 0.1, batch 128, seed 0), so the numbers match it.

Safe to interrupt (Ctrl+C): every finished network is saved, and running the
script again skips the ones that are already done.
"""

import os
import pickle
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from neural_network import NeuralNetwork
from data import load_digits_small, load_mnist, one_hot

ACTIVATIONS = ["sigmoid", "tanh", "relu"]
INITS = ["zeros", "random", "xavier", "he"]

CONFIG = {"hidden": 64, "epochs": 20, "lr": 0.1, "batch": 128, "seed": 0}


def arg_value(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def run_order():
    """Report experiments first, so an interrupted run still has the key results."""
    first = [(a, "xavier") for a in ACTIVATIONS] + \
            [("relu", i) for i in ("zeros", "random", "he")]
    rest = [(a, i) for a in ACTIVATIONS for i in INITS if (a, i) not in first]
    return first + rest


def train_and_analyze(act, init, data, cfg):
    X_train, X_test, y_train, y_test = data
    net = NeuralNetwork(X_train.shape[1], cfg["hidden"], 10,
                        activation=act, init_method=init, seed=cfg["seed"])
    net.train(X_train, one_hot(y_train, 10), epochs=cfg["epochs"],
              learning_rate=cfg["lr"], batch_size=cfg["batch"],
              X_val=X_test, y_val=y_test)

    ablation = net.neuron_importance(X_test, y_test)
    magnitude = net.neuron_activation_magnitude(X_test)
    if np.ptp(ablation) == 0 or np.ptp(magnitude) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(ablation, magnitude)[0])

    return {
        "activation": act,
        "init": init,
        "loss_history": [float(v) for v in net.loss_history],
        "val_acc_history": [float(v) for v in net.val_acc_history],
        "train_acc": float(net.accuracy(X_train, y_train)),
        "test_acc": float(net.accuracy(X_test, y_test)),
        "ablation": np.asarray(ablation, dtype=np.float64),
        "magnitude": np.asarray(magnitude, dtype=np.float64),
        "rho": rho,
        "W1": net.W1.astype(np.float32),
    }


def save_atomic(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    small = "--small" in sys.argv
    cfg = dict(CONFIG)
    cfg["epochs"] = int(arg_value("--epochs", cfg["epochs"]))  # only for quick tests
    out_dir = arg_value("--out", "precomputed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "small.pkl" if small else "mnist.pkl")

    print("Loading data...", flush=True)
    data = load_digits_small() if small else load_mnist()
    print(f"Train {data[0].shape} | Test {data[1].shape}", flush=True)

    payload = {"config": cfg, "results": {}}
    if os.path.exists(path):
        with open(path, "rb") as f:
            old = pickle.load(f)
        if old.get("config") == cfg:
            payload = old
            print(f"Resuming: {len(payload['results'])} networks already saved.", flush=True)
        else:
            print("Existing file used different settings, starting fresh.", flush=True)

    order = run_order()
    todo = [(a, i) for a, i in order if f"{a}|{i}" not in payload["results"]]
    if not todo:
        print(f"Everything is already computed: {path}")
        sys.exit(0)

    print(f"Training {len(todo)} networks ({cfg['epochs']} epochs each). "
          f"You can stop with Ctrl+C and resume later.\n", flush=True)
    start = time.time()
    for n, (act, init) in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{n}/{len(todo)}] {act} + {init} ... ", end="", flush=True)
        payload["results"][f"{act}|{init}"] = train_and_analyze(act, init, data, cfg)
        save_atomic(payload, path)
        done = time.time() - start
        eta = done / n * (len(todo) - n)
        res = payload["results"][f"{act}|{init}"]
        print(f"done in {time.time() - t0:.0f}s, test acc {res['test_acc'] * 100:.2f}%"
              f"  (about {eta / 60:.1f} min left)", flush=True)

    print(f"\nAll done. Saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB).")
    print("Now run:  streamlit run app.py")
