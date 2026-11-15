"""Connect the NN confidence model, correction search, and Sudoku CSP."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from pathlib import Path

from csp.sudoku import SudokuSolver
from data.make_data import (
    build_mosaic,
    detect_blank,
    generate_puzzle,
    inject_hard_misreads,
    inject_misreads,
    load_mnist,
)
from nn.classifier import MLPClassifier
from nn.train import load_mlp
from search.correction import correct_puzzle


def confidence_model(observed: np.ndarray, solution: np.ndarray, seed: int = 7) -> np.ndarray:
    """Return deterministic class probabilities, standing in for NN output in demos."""
    rng = np.random.default_rng(seed)
    probabilities = np.full((81, 9), 0.01)
    for cell in range(81):
        row, col = divmod(cell, 9)
        expected = int(solution[row, col])
        predicted = int(observed[row, col]) or expected
        probabilities[cell, predicted - 1] = 0.80 if predicted == expected else 0.52
        probabilities[cell, expected - 1] = 0.80 if predicted == expected else 0.45
        probabilities[cell] += rng.random(9) * 0.001
        probabilities[cell] /= probabilities[cell].sum()
    return probabilities


@lru_cache(maxsize=16)
def _trained_demo_model(seed: int) -> MLPClassifier:
    """Train a tiny digit MLP and use it to score the observed digit symbols."""
    features = np.eye(9, dtype=float)
    labels = np.arange(9)
    model = MLPClassifier(input_size=9, hidden_size=18, classes=9, seed=seed)
    model.fit(features, labels, epochs=20, learning_rate=0.15, batch_size=9, seed=seed)
    return model


def neural_confidence_model(observed: np.ndarray, seed: int = 7) -> np.ndarray:
    values = np.where(observed.ravel() == 0, 1, observed.ravel()) - 1
    return _trained_demo_model(seed).predict_proba(np.eye(9, dtype=float)[values])


def _mnist_path() -> Path | None:
    for candidate in (Path("data/mnist"), Path(r"D:\setups\dataset")):
        if candidate.exists():
            return candidate
    return None


def _real_observation(
    puzzle: np.ndarray, solution: np.ndarray, misreads: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
    path = _mnist_path()
    weights = Path("artifacts/mlp_weights.npz")
    if path is None or not weights.exists():
        raise FileNotFoundError("real MNIST assets are unavailable")
    _, _, test_images, test_labels = load_mnist(path)
    model = load_mlp(weights)
    rng = np.random.default_rng(seed)
    build_mosaic(puzzle, test_images, test_labels, rng)
    observed, mosaic, injected = inject_hard_misreads(
        puzzle, solution, misreads, test_images, test_labels, model, seed,
    )
    probabilities = np.zeros((81, 9), dtype=float)
    classified = np.zeros((9, 9), dtype=int)
    for row in range(9):
        for col in range(9):
            cell = mosaic[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28]
            if detect_blank(cell):
                continue
            probabilities[row * 9 + col] = model.predict_proba(cell[None, ...])[0]
            classified[row, col] = int(probabilities[row * 9 + col].argmax() + 1)
    observed = classified
    return observed, mosaic, injected, probabilities


def run_pipeline(
    seed: int = 7, misreads: int = 2, method: str = "ucs", nn_enabled: bool = True,
    use_forward_checking: bool = True, use_ac3: bool = True, use_backtracking: bool = True,
) -> dict:
    puzzle, solution = generate_puzzle(seed=seed, givens=35)
    if nn_enabled:
        try:
            observed, mosaic, injected, probabilities = _real_observation(
                puzzle, solution, misreads, seed,
            )
            real_nn = True
        except (FileNotFoundError, ValueError) as error:
            print(f"FALLBACK: simulated probabilities, not real NN ({error})")
            observed, injected = inject_misreads(puzzle, solution, misreads, seed=seed + 1)
            probabilities = confidence_model(observed, solution, seed)
            mosaic = None
            real_nn = False
    else:
        observed, injected = inject_misreads(puzzle, solution, misreads, seed=seed + 1)
        probabilities = np.full((81, 9), 1 / 9)
        mosaic = None
        real_nn = False
    csp_solver: SudokuSolver | None = None

    def solve(candidate: np.ndarray) -> list[list[int]] | None:
        nonlocal csp_solver
        csp_solver = SudokuSolver(candidate)
        if not use_backtracking:
            return None
        return csp_solver.solve(use_forward_checking, use_ac3)

    result = correct_puzzle(observed, probabilities, method=method, max_corrections=3, solver=solve)
    stats = csp_solver.stats if csp_solver else SudokuSolver(observed).stats
    return {
        "puzzle": puzzle,
        "observed": observed,
        "solution": solution,
        "injected": injected,
        "probabilities": probabilities,
        "result": result,
        "nn_enabled": nn_enabled,
        "real_nn": real_nn,
        "mosaic": mosaic,
        "csp_stats": stats,
        "csp_flags": {"backtracking": use_backtracking, "forward_checking": use_forward_checking, "ac3": use_ac3},
        "conflicts": sorted((int(row), int(col)) for row, col in _grid_conflicts(observed)),
    }


def _grid_conflicts(grid: np.ndarray) -> set[tuple[int, int]]:
    from search.correction import _conflicts
    return _conflicts(grid)


def render_report(run: dict) -> str:
    result = run["result"]
    lines = ["Input observed grid:", str(run["observed"]), "", f"Search: {result.method}",
             f"Nodes expanded/generated: {result.nodes_expanded}/{result.nodes_generated}",
             f"Status: {result.message}", "Correction log:"]
    lines.extend(str(item) for item in result.corrections)
    if result.message == "no solution":
        confidence = run["probabilities"].max(axis=1)
        least = np.argsort(confidence)[:5]
        lines.append("unreadable; least confident cells: " + str([divmod(int(cell), 9) for cell in least]))
    return "\n".join(lines)
