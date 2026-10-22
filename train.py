"""
It trains a small network on the toy dataset and prints loss + accuracy.
Once this works, you're ready to start building the tabs (activation
comparison, neuron importance, init comparison) on top of it.
"""

from data import make_toy_dataset, one_hot
from neural_network import NeuralNetwork

# 1. Get data
X, y = make_toy_dataset(n_samples=300, n_classes=3)
y_onehot = one_hot(y, n_classes=3)

# 2. Build the network
net = NeuralNetwork(
    n_input=2,
    n_hidden=16,
    n_output=3,
    activation="relu",
    init_method="xavier",
    seed=0,
)

# 3. Train
net.train(X, y_onehot, epochs=200, learning_rate=0.5, verbose=True)

# 4. Evaluate
acc = net.accuracy(X, y)
print(f"\nFinal training accuracy: {acc * 100:.2f}%")

# 5. Try neuron importance 
importances = net.neuron_importance(X, y)
print("\nTop 5 most important hidden neurons (by accuracy drop when ablated):")
top5 = importances.argsort()[::-1][:5]
for rank, idx in enumerate(top5, 1):
    print(f"  #{rank}: neuron {idx} -> importance {importances[idx]:.4f}")