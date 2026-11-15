from pathlib import Path

import numpy as np
import pytest

from data.make_data import build_mosaic, detect_blank, generate_puzzle, inject_hard_misreads, load_mnist
from integration.pipeline import confidence_model
from nn.train import load_mlp
from search.correction import correct_puzzle


def test_blank_detection_and_mosaic_shape():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[1, 10:18, 10:18] = 255
    labels = np.array([0, 1])
    grid = np.zeros((9, 9), dtype=int)
    grid[0, 0] = 1
    mosaic = build_mosaic(grid, images, labels, np.random.default_rng(1))
    assert mosaic.shape == (252, 252)
    assert detect_blank(np.zeros((28, 28), dtype=np.uint8))
    assert not detect_blank(images[1])


def test_hard_digit_injection_is_a_real_misread():
    dataset = Path(r"D:\setups\dataset")
    weights = Path("artifacts/mlp_weights.npz")
    if not dataset.exists() or not weights.exists():
        pytest.skip("real MNIST dataset and trained weights are required")
    _, _, images, labels = load_mnist(dataset)
    model = load_mlp(weights)
    puzzle, solution = generate_puzzle(7, 35)
    observed, mosaic, log = inject_hard_misreads(
        puzzle, solution, 1, images, labels, model, seed=7,
    )
    row, col, read_digit, true_digit = log[0]
    cell = mosaic[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28]
    assert mosaic.shape == (252, 252)
    assert int(model.predict(cell[None, ...])[0]) == read_digit
    assert read_digit != true_digit
    assert observed[row, col] == read_digit


def test_search_counts_are_valid_when_methods_share_the_same_goal_path():
    puzzle, solution = generate_puzzle(7, 35)
    observed, _ = __import__("data.make_data", fromlist=["inject_misreads"]).inject_misreads(
        puzzle, solution, 1, 8,
    )
    probabilities = confidence_model(observed, solution, 7)
    counts = [correct_puzzle(observed, probabilities, method, 3).nodes_expanded
              for method in ("ucs", "greedy", "astar")]
    # These methods may match when the first cheapest child is already a goal;
    # the test verifies all three implementations still expose valid counters.
    assert all(count >= 1 for count in counts)
