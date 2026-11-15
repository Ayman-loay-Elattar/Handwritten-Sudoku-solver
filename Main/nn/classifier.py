"""Small NumPy classifiers used for the MNIST comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    result = np.zeros((len(labels), classes))
    result[np.arange(len(labels)), labels] = 1
    return result


@dataclass
class TrainingHistory:
    losses: list[float]


class MLPClassifier:
    def __init__(self, input_size: int = 784, hidden_size: int = 64, classes: int = 9, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.classes = classes
        self.weights1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, classes))
        self.bias2 = np.zeros(classes)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(0, x @ self.weights1 + self.bias1)
        return hidden, _softmax(hidden @ self.weights2 + self.bias2)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.05,
            batch_size: int = 128, seed: int = 7) -> TrainingHistory:
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        y = np.asarray(y, dtype=int)
        if x.size and x.max() > 1:
            x = x / 255.0
        rng = np.random.default_rng(seed)
        losses = []
        targets = _one_hot(y, self.classes)
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                indices = order[start : start + batch_size]
                xb, target = x[indices], targets[indices]
                hidden, probabilities = self._forward(xb)
                error = (probabilities - target) / len(xb)
                grad_w2 = hidden.T @ error
                grad_b2 = error.sum(axis=0)
                hidden_error = (error @ self.weights2.T) * (hidden > 0)
                grad_w1 = xb.T @ hidden_error
                grad_b1 = hidden_error.sum(axis=0)
                self.weights2 -= learning_rate * grad_w2
                self.bias2 -= learning_rate * grad_b2
                self.weights1 -= learning_rate * grad_w1
                self.bias1 -= learning_rate * grad_b1
            losses.append(self.loss(x, y))
        return TrainingHistory(losses)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        if x.size and x.max() > 1:
            x = x / 255.0
        return self._forward(x)[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1) + 1

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probabilities = self.predict_proba(x)
        return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())


class LogisticRegressionBaseline:
    def __init__(self, input_size: int = 784, classes: int = 9, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.01, (input_size, classes))
        self.bias = np.zeros(classes)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.1,
            batch_size: int = 128, seed: int = 7) -> "LogisticRegressionBaseline":
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        if x.size and x.max() > 1:
            x = x / 255.0
        target = _one_hot(np.asarray(y), self.weights.shape[1])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
                probabilities = _softmax(x[indices] @ self.weights + self.bias)
                error = (probabilities - target[indices]) / len(indices)
                self.weights -= learning_rate * x[indices].T @ error
                self.bias -= learning_rate * error.sum(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        if x.size and x.max() > 1:
            x = x / 255.0
        return _softmax(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1) + 1


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    predictions = probabilities.argmax(axis=1) + 1
    classes = probabilities.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)
    for actual, predicted in zip(labels, predictions):
        confusion[actual - 1, predicted - 1] += 1
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    bins = np.linspace(0, 1, 11)
    calibration_error = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            calibration_error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"accuracy": float(correct.mean()), "confusion_matrix": confusion, "calibration_error": float(calibration_error)}
