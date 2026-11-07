"""
Backend for the digit-recognizer frontend.

Loads a (weights, biases, betas) tuple pickled from the training script's
network: 784 -> 16 -> 16 -> 10, with weights[i] shaped (out, in) and
biases[i]/betas[i] shaped (out,).

Matches the training forward pass exactly:
  - hidden layers: PReLU  ->  z if z >= 0 else beta * z   (per-neuron beta)
  - output layer:  softmax
  - input pixels scaled to [0, 1] (same /255.0 used in training)
"""
import sys, os
print("RUNNING FILE:", os.path.abspath(__file__))
print("PYTHON:", sys.executable)
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = "model.pkl"

with open(MODEL_PATH, "rb") as f:
    weights, biases, betas = pickle.load(f)

weights = [np.array(w) for w in weights]
biases = [np.array(b).reshape(-1, 1) for b in biases]
betas = [np.array(b).reshape(-1, 1) for b in betas]

NUM_LAYERS = len(weights)  # 3: hidden1, hidden2, output


def prelu(z, beta):
    return np.where(z >= 0, z, beta * z)


def softmax(z):
    z = z - z.max(axis=0, keepdims=True)
    exps = np.exp(z)
    return exps / exps.sum(axis=0)


def forward(pixels_784):
    """pixels_784: flat list/array of 784 floats in [0, 1]. Returns 10 softmax probabilities."""
    a = np.array(pixels_784, dtype=np.float64).reshape(-1, 1)  # (784, 1)
    for i, (w, b, beta) in enumerate(zip(weights, biases, betas)):
        z = w @ a + b
        if i != NUM_LAYERS - 1:
            a = prelu(z, beta)
        else:
            a = softmax(z)
    return a.flatten()  # (10,) softmax probabilities, sums to 1


app = Flask(__name__)
CORS(app)  # allow the frontend (served from file:// or another port) to call this


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    pixels = data.get("pixels")

    if pixels is None or len(pixels) != 784:
        return jsonify({"error": "Expected 'pixels': array of 784 floats"}), 400

    outputs = forward(pixels)
    prediction = int(np.argmax(outputs))
    probabilities = outputs.tolist()  # already a softmax distribution, sums to 1

    return jsonify({
        "prediction": prediction,
        "probabilities": probabilities
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "layers": [w.shape for w in weights]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
