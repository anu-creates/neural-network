"""
Synapse - interactive explorer for the from-scratch neural network.

Run from the project folder (the one containing neural_network.py):
    streamlit run app.py

By default the app shows saved results: networks trained once on ALL 60,000
MNIST training images (by precompute.py) with the exact settings of the report,
so everything appears instantly. The "Advanced" box lets you train your own
networks live, on fewer images so it finishes in seconds. Either way, every
network is trained by neural_network.py (pure NumPy).
"""

import os
import pickle
import time
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr

from data import load_digits_small, load_mnist, one_hot
from neural_network import ACTIVATIONS, NeuralNetwork

INITS = ["zeros", "random", "xavier", "he"]
DATASETS = ["MNIST (60k images)", "Small digits (fast)"]
CACHE_DIR = "precomputed"
CACHE_FILES = {DATASETS[0]: "mnist.pkl", DATASETS[1]: "small.pkl"}

st.set_page_config(page_title="Synapse", layout="wide")


# =====================================================================
# Cached helpers (Streamlit re-runs this whole script on every click,
# so anything expensive must be cached or it would retrain every time)
# =====================================================================
@st.cache_resource(show_spinner="Loading dataset...")
def get_data(name):
    return load_mnist() if name.startswith("MNIST") else load_digits_small()


@st.cache_resource(show_spinner=False)
def train_model(dataset, activation, init, hidden, epochs, lr, batch, n_train=None):
    """Train one network with the given settings and return its history."""
    X_train, X_test, y_train, y_test = get_data(dataset)
    if n_train:
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    net = NeuralNetwork(X_train.shape[1], hidden, 10,
                        activation=activation, init_method=init, seed=0)
    t0 = time.time()
    net.train(X_train, one_hot(y_train, 10), epochs=epochs, learning_rate=lr,
              batch_size=batch, X_val=X_test, y_val=y_test)
    return {
        "activation": activation,
        "init": init,
        "loss_history": list(net.loss_history),
        "val_acc_history": list(net.val_acc_history),
        "train_acc": net.accuracy(X_train, y_train),
        "test_acc": net.accuracy(X_test, y_test),
        "seconds": time.time() - t0,
        "net": net,
    }


APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _ci_child(parent, name):
    """Entry of `parent` called `name`, ignoring upper/lower case (or None).
    Hosting servers are case-sensitive, Windows is not."""
    try:
        for entry in os.listdir(parent):
            if entry.lower() == name.lower():
                return os.path.join(parent, entry)
    except OSError:
        pass
    return None


def find_cache_file(filename):
    """Find precomputed/<filename> next to app.py or in the working directory.
    Also accepts the file sitting directly next to app.py."""
    for base in dict.fromkeys([APP_DIR, os.getcwd()]):
        folder = _ci_child(base, CACHE_DIR)
        for parent in (folder, base):
            if parent:
                found = _ci_child(parent, filename)
                if found and os.path.isfile(found):
                    return found
    return None


@st.cache_resource(show_spinner=False)
def _load_precomputed(path, mtime):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # unreadable or incompatible file: report, do not crash
        return {"__error__": f"{type(e).__name__}: {e}"}


def precomputed_for(dataset):
    """Results saved by precompute.py for this dataset, or None if unavailable."""
    path = find_cache_file(CACHE_FILES[dataset])
    if path is None:
        return None
    data = _load_precomputed(path, os.path.getmtime(path))
    if "__error__" in data or "results" not in data:
        return None
    return data


def cache_diagnosis(dataset):
    """Plain-text explanation of what the app can see, for debugging hosting."""
    filename = CACHE_FILES[dataset]
    lines = [f"Looking for: {CACHE_DIR}/{filename}",
             f"App folder: {APP_DIR}",
             f"Working directory: {os.getcwd()}"]
    for base in dict.fromkeys([APP_DIR, os.getcwd()]):
        try:
            lines.append(f"Files in {base}: {sorted(os.listdir(base))[:30]}")
            sub = _ci_child(base, CACHE_DIR)
            if sub:
                lines.append(f"Files in {sub}: {sorted(os.listdir(sub))}")
        except OSError as e:
            lines.append(f"Cannot list {base}: {e}")
    path = find_cache_file(filename)
    if path is None:
        lines.append("Result: the file was not found in any of these places.")
    else:
        data = _load_precomputed(path, os.path.getmtime(path))
        if "__error__" in data:
            lines.append(f"Result: found {path} but could not read it: {data['__error__']}")
        elif "results" not in data:
            lines.append(f"Result: found {path} but it has an unexpected format.")
    return "\n".join(lines)


def get_result(cfg, act, init):
    """One trained network's results: instant in Fast mode, trained in Live mode."""
    if cfg["fast"]:
        return precomputed_for(cfg["dataset"])["results"][f"{act}|{init}"]
    return train_model(cfg["dataset"], act, init, cfg["hidden"],
                       cfg["epochs"], cfg["lr"], cfg["batch"], cfg.get("n_train"))


def describe(cfg):
    if cfg.get("fast"):
        images = "all 60,000 training images (saved results)"
    elif cfg.get("n_train"):
        images = f"{cfg['n_train']:,} training images (live)"
    else:
        images = "all training images (live)"
    return (f"{cfg['dataset']} | {images} | {cfg['hidden']} hidden neurons | "
            f"{cfg['epochs']} epochs | learning rate {cfg['lr']} | "
            f"batch {cfg['batch']}")


def epochs_to_reach(acc_history, target=0.90):
    for epoch, acc in enumerate(acc_history):
        if acc >= target:
            return epoch
    return None


def run_many(specs, cfg, state_key):
    """Fetch every (activation, init) pair in specs (trains them in Live mode)."""
    bar = st.progress(0.0, text="Starting...")
    out = []
    for i, (act, init) in enumerate(specs):
        bar.progress(i / len(specs),
                     text=f"{'Loading' if cfg['fast'] else 'Training'} "
                          f"{act} + {init}  ({i + 1} of {len(specs)})...")
        out.append(get_result(cfg, act, init))
    bar.empty()
    st.session_state[state_key] = (dict(cfg), out)


def show_comparison(state_key, label_key):
    saved = st.session_state.get(state_key)
    if not saved:
        st.info("Choose settings above and press the Run button.")
        return
    cfg_used, results = saved
    st.caption(f"Showing results for: {describe(cfg_used)}")

    loss_df = pd.DataFrame({r[label_key]: r["loss_history"] for r in results})
    acc_df = pd.DataFrame({r[label_key]: [a * 100 for a in r["val_acc_history"]]
                           for r in results})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training loss** (lower is better, x-axis = epoch)")
        st.line_chart(loss_df)
    with c2:
        st.markdown("**Test accuracy %** (higher is better, x-axis = epoch)")
        st.line_chart(acc_df)

    rows = []
    for r in results:
        e90 = epochs_to_reach(r["val_acc_history"])
        rows.append({
            label_key: r[label_key],
            "final loss": round(r["loss_history"][-1], 4),
            "train acc %": round(r["train_acc"] * 100, 2),
            "test acc %": round(r["test_acc"] * 100, 2),
            "epochs to 90% test acc": str(e90) if e90 is not None else "never",
        })
    st.dataframe(pd.DataFrame(rows).set_index(label_key))


def neuron_images(W1, idxs, ablation, side, heading):
    st.markdown(heading)
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.1 * len(idxs), 2.5))
    for ax, i in zip(np.atleast_1d(axes), idxs):
        w = W1[:, i].reshape(side, side)
        lim = np.abs(w).max() or 1.0
        ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"neuron {i}\ndrop {ablation[i] * 100:.2f} pts", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def toy_attention(tokens, scale, d=8):
    """Single-head scaled dot-product attention with fixed random (untrained) weights."""
    def embed(word):
        rng = np.random.default_rng(zlib.crc32(word.lower().encode()))
        return rng.normal(size=d)

    X = np.stack([embed(t) for t in tokens])
    rng = np.random.default_rng(0)
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = (Q @ K.T) / np.sqrt(d) * scale
    weights = softmax_rows(scores)
    return scores, weights, weights @ V


# =====================================================================
# Sidebar: settings shared by every tab
# =====================================================================
st.sidebar.title("Synapse")
st.sidebar.caption("App version 4 (finds saved results next to app.py)")
info_slot = st.sidebar.container()

with st.sidebar.expander("Advanced: train with my own settings"):
    user_live = st.checkbox("Train my own networks (slow)", key="live")
    st.caption("Uses fewer images and epochs by default so it finishes in seconds. "
               "Numbers will differ from the report.")
    dataset_choice = st.radio("Dataset", DATASETS, key="dataset", disabled=not user_live)
    hidden = st.slider("Hidden neurons", 8, 128, 64, step=8, key="hidden",
                       disabled=not user_live)
    epochs = st.slider("Epochs", 1, 30, 5, key="epochs", disabled=not user_live)
    lr = st.select_slider("Learning rate", [0.01, 0.03, 0.1, 0.3, 0.5], value=0.1,
                          key="lr", disabled=not user_live)
    batch = st.select_slider("Batch size", [32, 64, 128, 256], value=128,
                             key="batch", disabled=not user_live)
    n_train = st.select_slider("Training images", [2000, 5000, 10000, 20000, 60000],
                               value=10000, key="n_train", disabled=not user_live)

pre = None if user_live else precomputed_for(DATASETS[0])
use_saved = pre is not None

if use_saved:
    c = pre["config"]
    cfg = {"dataset": DATASETS[0], "fast": True, "hidden": c["hidden"],
           "epochs": c["epochs"], "lr": c["lr"], "batch": c["batch"], "n_train": None}
else:
    cfg = {"dataset": dataset_choice if user_live else DATASETS[1], "fast": False,
           "hidden": hidden, "epochs": epochs, "lr": lr, "batch": batch,
           "n_train": n_train}

with info_slot:
    if use_saved:
        st.markdown("**Showing saved results**")
        st.markdown(f"Trained on all **60,000** MNIST training images; accuracy is "
                    f"measured on 10,000 unseen test images.  \n"
                    f"{c['hidden']} hidden neurons, {c['epochs']} epochs, learning "
                    f"rate {c['lr']}, batch {c['batch']}, seed {c['seed']}.")
        st.caption("These are the exact settings and results of the report. Nothing "
                   "is reduced. They were trained once beforehand, so they load "
                   "instantly.")
    elif user_live:
        st.markdown("**Training live with your settings**")
        st.caption("Each network trains when you press Run. Results will not match "
                   "the report unless you use the same settings and all images.")
    else:
        st.warning("No saved MNIST results found. On your laptop, run "
                   "`python precompute.py` once (about 10 minutes). On a hosted link, "
                   "make sure `precomputed/mnist.pkl` is pushed to GitHub. Until then "
                   "the app trains small networks live on a tiny digits dataset.")
        with st.expander("Details (for debugging)"):
            st.code(cache_diagnosis(DATASETS[0]))

st.title("Synapse")
st.caption("A neural network built from first principles in NumPy. Change one thing "
           "at a time and watch what happens.")

tab_act, tab_init, tab_neuron, tab_attn = st.tabs(
    ["Activation functions", "Weight initialization",
     "Neuron importance", "Attention (toy)"])

# =====================================================================
# Tab 1: activation functions
# =====================================================================
with tab_act:
    st.markdown("**Experiment:** change only the activation function; "
                "everything else stays fixed.")
    c1, c2 = st.columns([2, 1])
    acts = c1.multiselect("Activations to compare", list(ACTIVATIONS),
                          default=list(ACTIVATIONS), key="acts")
    fixed_init = c2.selectbox("Fixed initialization", INITS, index=2, key="act_init")
    if st.button("Run activation comparison", key="run_act", disabled=not acts):
        run_many([(a, fixed_init) for a in acts], cfg, "act_results")
    show_comparison("act_results", "activation")

# =====================================================================
# Tab 2: weight initialization
# =====================================================================
with tab_init:
    st.markdown("**Experiment:** change only how the weights start; "
                "everything else stays fixed.")
    c1, c2 = st.columns([2, 1])
    inits = c1.multiselect("Initializations to compare", INITS,
                           default=INITS, key="inits")
    fixed_act = c2.selectbox("Fixed activation", list(ACTIVATIONS),
                             index=list(ACTIVATIONS).index("relu"), key="init_act")
    if st.button("Run initialization comparison", key="run_init", disabled=not inits):
        run_many([(fixed_act, i) for i in inits], cfg, "init_results")
    show_comparison("init_results", "init")

# =====================================================================
# Tab 3: neuron importance
# =====================================================================
with tab_neuron:
    st.markdown("**Which hidden neurons actually matter?** Two independent measures: "
                "*ablation* (silence one neuron, measure the accuracy drop) and "
                "*activation magnitude* (how strongly it fires on average).")
    c1, c2 = st.columns(2)
    n_act = c1.selectbox("Activation", list(ACTIVATIONS),
                         index=list(ACTIVATIONS).index("relu"), key="n_act")
    n_init = c2.selectbox("Initialization", INITS, index=3, key="n_init")

    if st.button("Show neuron analysis" if cfg["fast"] else "Train and analyze",
                 key="run_neuron"):
        with st.spinner("Working..."):
            res = get_result(cfg, n_act, n_init)
            if "ablation" in res:          # Fast mode: analysis was precomputed
                ablation, magnitude = res["ablation"], res["magnitude"]
                rho, W1 = res["rho"], res["W1"]
            else:                          # Live mode: silence each neuron now
                net = res["net"]
                _, X_test, _, y_test = get_data(cfg["dataset"])
                ablation = net.neuron_importance(X_test, y_test)
                magnitude = net.neuron_activation_magnitude(X_test)
                if np.ptp(ablation) == 0 or np.ptp(magnitude) == 0:
                    rho = float("nan")
                else:
                    rho = float(spearmanr(ablation, magnitude)[0])
                W1 = net.W1.copy()
            st.session_state["neuron"] = (dict(cfg), {
                "label": f"{n_act} + {n_init}",
                "test_acc": res["test_acc"],
                "ablation": ablation,
                "magnitude": magnitude,
                "rho": rho,
                "W1": W1,
            })

    saved = st.session_state.get("neuron")
    if not saved:
        st.info("Press the button to analyze a network's neurons.")
    else:
        cfg_used, r = saved
        ablation, magnitude, W1 = r["ablation"], r["magnitude"], r["W1"]
        st.caption(f"Showing results for: {r['label']} | {describe(cfg_used)}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{r['test_acc'] * 100:.2f}%")
        m2.metric("Neurons with ~no impact", f"{int(np.sum(ablation < 0.001))} of {len(ablation)}")
        m3.metric("Dead neurons (never fire)", f"{int(np.sum(magnitude < 1e-6))}")
        m4.metric("Ablation vs magnitude (Spearman)",
                  "n/a" if np.isnan(r["rho"]) else f"{r['rho']:.2f}")

        order = np.argsort(ablation)[::-1]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar(range(len(order)), ablation[order] * 100)
            ax.set_xlabel("Neurons, sorted by importance")
            ax.set_ylabel("Test accuracy drop (points)")
            ax.set_title("Importance by ablation")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.scatter(magnitude, ablation * 100)
            ax.set_xlabel("Mean |activation|")
            ax.set_ylabel("Test accuracy drop (points)")
            ax.set_title("Is a strongly firing neuron an important one?")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        side = int(round(np.sqrt(W1.shape[0])))
        k = st.slider("Neurons to display", 3, 8, 5, key="k_neurons")
        st.markdown("Each image shows the input weights of one neuron: what pattern "
                    "of pixels it responds to (red = excites it, blue = suppresses it).")
        neuron_images(W1, order[:k], ablation, side, "**Most important neurons**")
        neuron_images(W1, np.argsort(ablation)[:k], ablation, side,
                      "**Least important neurons**")

# =====================================================================
# Tab 4: toy attention
# =====================================================================
with tab_attn:
    st.markdown("**A toy illustration of attention.** Each word asks a question "
                "(query), every word offers a label (key), and the match scores are "
                "turned into weights that blend the words' information (values).")
    st.warning("Toy demo: the embeddings and weight matrices here are random and "
               "untrained, so the pattern shows the *mechanism*, not real language "
               "understanding.")
    sentence = st.text_input("Sentence (2 to 12 words)", "the cat sat on the mat",
                             key="sentence")
    scale = st.slider("Sharpness (higher = more focused attention)", 0.2, 5.0, 1.0,
                      step=0.2, key="sharp")
    words = sentence.split()[:12]
    if len(words) < 2:
        st.info("Enter at least two words.")
    else:
        labels = [f"{i + 1}:{w}" for i, w in enumerate(words)]
        scores, weights, out = toy_attention(words, scale)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(words)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(words)), labels)
        ax.set_xlabel("Word being looked at (key)")
        ax.set_ylabel("Word doing the looking (query)")
        fig.colorbar(im, ax=ax, label="attention weight")
        fig.tight_layout()
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig)
        plt.close(fig)
        with c2:
            q = st.selectbox("Show where this word looks", range(len(words)),
                             format_func=lambda i: labels[i], key="q_word")
            st.bar_chart(pd.DataFrame({"attention weight": weights[q]}, index=labels))
            st.caption("Each row of the heatmap sums to 1: a word has a fixed "
                       "budget of attention to spread across all words.")

        with st.expander("The math behind it"):
            st.latex(r"\text{scores} = \frac{QK^{T}}{\sqrt{d}} \qquad "
                     r"\text{weights} = \mathrm{softmax}(\text{scores}) \qquad "
                     r"\text{output} = \text{weights} \cdot V")
            st.write("Softmax is the same function the network's output layer uses "
                     "to turn scores into probabilities.")
            st.bar_chart(pd.DataFrame({"attention weight": weights[q]}, index=labels))
            st.caption("Each row of the heatmap sums to 1: a word has a fixed "
                       "budget of attention to spread across all words.")

        with st.expander("The math behind it"):
            st.latex(r"\text{scores} = \frac{QK^{T}}{\sqrt{d}} \qquad "
                     r"\text{weights} = \mathrm{softmax}(\text{scores}) \qquad "
                     r"\text{output} = \text{weights} \cdot V")
            st.write("Softmax is the same function the network's output layer uses "
                     "to turn scores into probabilities.")
