"""
Checks accuracy of a trained model (saved as model.pkl) against mnist_test.csv.
No training happens here — this just loads the weights and runs the same
forward pass used in app.py, then compares predictions to the true labels.

Run from the same folder as model.pkl and mnist_test.csv:
    python test_accuracy.py
"""

import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "model.pkl"
TEST_CSV_PATH = "mnist_test.csv"

with open(MODEL_PATH, "rb") as f:
    weights, biases, betas = pickle.load(f)

weights = [np.array(w) for w in weights]
biases = [np.array(b).reshape(-1, 1) for b in biases]
betas = [np.array(b).reshape(-1, 1) for b in betas]

NUM_LAYERS = len(weights)


def prelu(z, beta):
    return np.where(z >= 0, z, beta * z)


def soft_max(z):
    z = z - z.max(axis=0, keepdims=True)
    exps = np.exp(z)
    return exps / exps.sum(axis=0)


def forward_batch(X):
    """X: shape (784, num_examples), values in [0, 1]. Returns (10, num_examples) softmax outputs."""
    a = X
    for i, (w, b, beta) in enumerate(zip(weights, biases, betas)):
        z = w @ a + b
        if i != NUM_LAYERS - 1:
            a = prelu(z, beta)
        else:
            a = soft_max(z)
    return a


data_test = pd.read_csv(TEST_CSV_PATH)
test_labels = data_test.iloc[:, 0].to_numpy()
test_data = data_test.iloc[:, 1:].to_numpy() / 255.0  # (num_examples, 784)

outputs = forward_batch(test_data.T)         # (10, num_examples)
predictions = np.argmax(outputs, axis=0)     # (num_examples,)

accuracy = np.mean(predictions == test_labels)
print(f"Test examples: {len(test_labels)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Break down accuracy per digit so you can see if specific digits are weak
print("\nPer-digit accuracy:")
for digit in range(10):
    mask = test_labels == digit
    digit_acc = np.mean(predictions[mask] == test_labels[mask])
    print(f"  {digit}: {digit_acc * 100:.2f}%  ({mask.sum()} examples)")
