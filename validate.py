"""
STEP 2 - VALIDATE: does the from-scratch engine learn real data?

    python validate_mnist.py          # real MNIST (downloads once, ~15MB)
    python validate_mnist.py --small  # tiny offline digits set, quick smoke test

Key idea: we report TEST accuracy (images the network never trained on).
Training accuracy alone can't tell you if the network generalizes.
"""

import sys
import time

from neural_network import NeuralNetwork
from data import load_mnist, load_digits_small, one_hot

use_small = "--small" in sys.argv

# 1. Data
if use_small:
    X_train, X_test, y_train, y_test = load_digits_small()
else:
    X_train, X_test, y_train, y_test = load_mnist()

n_input = X_train.shape[1]          # 784 for MNIST, 64 for small digits
print(f"Train: {X_train.shape}  Test: {X_test.shape}  (inputs per image: {n_input})")

y_train_oh = one_hot(y_train, 10)

# 2. Network: 784 -> 64 -> 10
net = NeuralNetwork(n_input=n_input, n_hidden=64, n_output=10,
                    activation="relu", init_method="he", seed=0)

print("Starting training (20 epochs)...", flush=True)
# 3. Train with mini-batches, tracking test accuracy every epoch
start = time.time()
net.train(X_train, y_train_oh, epochs=20, learning_rate=0.1,
          batch_size=128, X_val=X_test, y_val=y_test, verbose=True)
print(f"\nTraining time: {time.time() - start:.1f}s")

# 4. Final numbers - these go in your report
train_acc = net.accuracy(X_train, y_train)
test_acc = net.accuracy(X_test, y_test)
print(f"Final train accuracy: {train_acc * 100:.2f}%")
print(f"Final test accuracy:  {test_acc * 100:.2f}%")
print(f"Gap (train - test):   {(train_acc - test_acc) * 100:.2f} points")
