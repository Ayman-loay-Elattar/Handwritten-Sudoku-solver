"""Train the NumPy NN and logistic baseline on local MNIST files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.make_data import prepare_mnist
from nn.classifier import LogisticRegressionBaseline, MLPClassifier, classification_metrics


def save_mlp(model: MLPClassifier, path: str | Path = "artifacts/mlp_weights.npz") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        weights1=model.weights1,
        bias1=model.bias1,
        weights2=model.weights2,
        bias2=model.bias2,
    )
    return target


def load_mlp(path: str | Path = "artifacts/mlp_weights.npz") -> MLPClassifier:
    values = np.load(path)
    model = MLPClassifier(
        input_size=values["weights1"].shape[0],
        hidden_size=values["weights1"].shape[1],
        classes=values["weights2"].shape[1],
    )
    model.weights1 = values["weights1"]
    model.bias1 = values["bias1"]
    model.weights2 = values["weights2"]
    model.bias2 = values["bias2"]
    return model


def train(path: str = "data/mnist", limit: int = 10000, seed: int = 7) -> dict:
    data = prepare_mnist(path, seed=seed, limit=limit)
    x_train, y_train = data["x_train"], data["y_train"] - 1
    x_validation, y_validation = data["x_validation"], data["y_validation"] - 1
    x_test, y_test = data["x_test"], data["y_test"] - 1
    nn = MLPClassifier(classes=9, seed=seed)
    baseline = LogisticRegressionBaseline(classes=9, seed=seed)
    nn.fit(x_train, y_train, epochs=5, seed=seed)
    baseline.fit(x_train, y_train, epochs=5, seed=seed)
    results = {
        "split": {"train_source_size": 50000, "validation_source_size": 10000,
                  "train_nonzero": len(x_train), "validation_nonzero": len(x_validation),
                  "test_nonzero": len(x_test)},
        "nn": classification_metrics(nn.predict_proba(x_test), y_test + 1),
        "logistic_regression": classification_metrics(baseline.predict_proba(x_test), y_test + 1),
        "validation": classification_metrics(nn.predict_proba(x_validation), y_validation + 1),
    }
    save_mlp(nn)
    return results


def save_metrics(results: dict, output: str = "artifacts/nn_metrics.json") -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(results)
    for name in ("nn", "logistic_regression", "validation"):
        serializable[name] = dict(serializable[name])
        serializable[name]["confusion_matrix"] = serializable[name]["confusion_matrix"].tolist()
    target.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    np.savetxt(target.with_name("nn_confusion_matrix.csv"), results["nn"]["confusion_matrix"],
               delimiter=",", fmt="%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/mnist")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/nn_metrics.json")
    args = parser.parse_args()
    results = train(args.path, args.limit)
    save_metrics(results, args.output)
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            print(name, "accuracy=", round(metrics["accuracy"], 4),
                  "calibration_error=", round(metrics["calibration_error"], 4))
    print("metrics saved to", args.output)
