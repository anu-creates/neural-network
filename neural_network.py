
"""
Synapse — a from-scratch neural network engine.

This is the core "engine" that every other feature (activation comparison,
neuron importance, initialization comparison) will plug into. It's a single
hidden-layer feedforward network trained with plain gradient descent —
no frameworks, just NumPy.
"""

import numpy as np


# ---------- Activation functions ----------
# Each activation needs a forward function AND its derivative, since
# backprop needs both.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}


# ---------- Weight initialization schemes ----------

def init_weights(n_in, n_out, method="random"):
    if method == "zeros":
        return np.zeros((n_in, n_out))
    elif method == "random":
        return np.random.randn(n_in, n_out) * 0.01
    elif method == "xavier":
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out))
    elif method == "he":
        return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
    else:
        raise ValueError(f"Unknown init method: {method}")


class NeuralNetwork:
    """
    A single-hidden-layer feedforward network.

    Architecture: input -> hidden (activation) -> output (softmax)
    Trained with plain batch gradient descent + cross-entropy loss.
    """

    def __init__(self, n_input, n_hidden, n_output,
                 activation="relu", init_method="random", seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.activation_name = activation
        self.act_fn, self.act_derivative = ACTIVATIONS[activation]

        # Weights and biases
        self.W1 = init_weights(n_input, n_hidden, init_method)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = init_weights(n_hidden, n_output, init_method)
        self.b2 = np.zeros((1, n_output))

        # For loss/accuracy tracking across training
        self.loss_history = []

    # ---------- Forward pass ----------
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.act_fn(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self._softmax(self.Z2)
        return self.A2

    @staticmethod
    def _softmax(z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ---------- Loss ----------
    @staticmethod
    def _cross_entropy(y_pred, y_true_onehot):
        m = y_true_onehot.shape[0]
        eps = 1e-9  # avoid log(0)
        return -np.sum(y_true_onehot * np.log(y_pred + eps)) / m

    # ---------- Backward pass ----------
    def backward(self, X, y_true_onehot, learning_rate=0.1):
        m = X.shape[0]

        # Output layer gradient (softmax + cross-entropy simplifies nicely)
        dZ2 = self.A2 - y_true_onehot
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.act_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # ---------- Training loop ----------
    def train(self, X, y_onehot, epochs=200, learning_rate=0.1,
              batch_size=None, X_val=None, y_val=None,
              verbose=False, shuffle_seed=0):
        """
        batch_size=None  -> full-batch gradient descent (all data per step)
        batch_size=128   -> mini-batch SGD (shuffled each epoch)

        Loss and validation accuracy are both measured at the START of each
        epoch, so they describe the same state of the network.
        """
        n = X.shape[0]
        full_batch = batch_size is None or batch_size >= n
        rng = np.random.default_rng(shuffle_seed)

        self.loss_history = []
        self.val_acc_history = []

        for epoch in range(epochs):
            # --- measure current state ---
            if X_val is not None:
                self.val_acc_history.append(self.accuracy(X_val, y_val))
            y_pred = self.forward(X)
            loss = self._cross_entropy(y_pred, y_onehot)
            self.loss_history.append(loss)

            # --- update weights ---
            if full_batch:
                self.backward(X, y_onehot, learning_rate)  # reuses forward(X) cache
            else:
                order = rng.permutation(n)
                for start in range(0, n, batch_size):
                    idx = order[start:start + batch_size]
                    self.forward(X[idx])
                    self.backward(X[idx], y_onehot[idx], learning_rate)

            if verbose and epoch % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch:4d} | Loss: {loss:.4f}"
                if X_val is not None:
                    msg += f" | Val acc: {self.val_acc_history[-1] * 100:.2f}%"
                print(msg, flush=True)

        return self.loss_history

    # ---------- Evaluation ----------
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y_true_labels):
        preds = self.predict(X)
        return np.mean(preds == y_true_labels)

    # ---------- Neuron activation magnitude ----------
    def neuron_activation_magnitude(self, X):
        """
        Second interpretability measure (the one named in the proposal):
        average absolute activation of each hidden neuron over the dataset.
        A neuron that is rarely/weakly active contributes little.
        """
        self.forward(X)
        return np.mean(np.abs(self.A1), axis=0)

    # ---------- Neuron importance (ablation) ----------
    def neuron_importance(self, X, y_true_labels):
        """
        For each hidden neuron, zero it out temporarily and measure how much
        accuracy drops. Bigger drop = more important neuron.
        """
        baseline_acc = self.accuracy(X, y_true_labels)
        importances = []

        n_hidden = self.W1.shape[1]
        for i in range(n_hidden):
            original_col = self.W2[i, :].copy()
            self.W2[i, :] = 0  # ablate this neuron's contribution to output

            ablated_acc = self.accuracy(X, y_true_labels)
            importances.append(baseline_acc - ablated_acc)

            self.W2[i, :] = original_col  # restore

        return np.array(importances)