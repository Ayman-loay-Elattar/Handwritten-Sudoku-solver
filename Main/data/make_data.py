"""Download MNIST and generate reproducible Sudoku observations.

The Sudoku images in this project are mosaics made from real MNIST digits;
the puzzle grids and injected mistakes are deliberately synthetic.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

MNIST_URLS = {
    "train_images": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train_labels": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "test_images": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(destination: str | Path = "data/mnist") -> Path:
    """Download the four public MNIST IDX files if they are absent."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        target = destination / f"{name}.gz"
        if not target.exists():
            urllib.request.urlretrieve(url, target)
    return destination


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic == 2051:
            rows, columns = struct.unpack(">II", stream.read(8))
            return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, columns)
        if magic == 2049:
            return np.frombuffer(stream.read(), dtype=np.uint8)[:count]
        raise ValueError(f"Unsupported IDX magic number: {magic}")


def load_mnist(path: str | Path = "data/mnist") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    names = {
        "train_images": ("train_images.gz", "train-images.idx3-ubyte", "train-images-idx3-ubyte"),
        "train_labels": ("train_labels.gz", "train-labels.idx1-ubyte", "train-labels-idx1-ubyte"),
        "test_images": ("test_images.gz", "t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"),
        "test_labels": ("test_labels.gz", "t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"),
    }

    def find(name: str) -> Path | None:
        for candidate in names[name]:
            direct = path / candidate
            nested = path / candidate.replace(".idx", "-idx").replace(".ubyte", "")
            if direct.exists():
                return direct
            if nested.exists():
                return nested
        matches = list(path.rglob(names[name][1])) + list(path.rglob(names[name][2]))
        return matches[0] if matches else None

    files = {name: find(name) for name in names}
    missing = [name for name, file in files.items() if file is None]
    if missing:
        raise FileNotFoundError(f"MNIST files missing in {path}: {', '.join(missing)}; run make_data.py --download")
    return (
        _read_idx(files["train_images"]),
        _read_idx(files["train_labels"]),
        _read_idx(files["test_images"]),
        _read_idx(files["test_labels"]),
    )


def prepare_mnist(
    path: str | Path = "data/mnist", seed: int = 7, limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Scale MNIST and split the original training set into 50k/10k partitions.

    The split is made before dropping class 0, so the requested 50,000/10,000
    source partition sizes remain exact while the classifier receives only 1-9.
    """
    train_images, train_labels, test_images, test_labels = load_mnist(path)
    if len(train_images) != 60000 or len(test_images) != 10000:
        raise ValueError("Expected standard MNIST sizes: 60000 train and 10000 test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_images))
    train_idx, validation_idx = order[:50000], order[50000:]

    def clean(indices: np.ndarray, images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = labels[indices] != 0
        x = images[indices][selected].astype(np.float32) / 255.0
        y = labels[indices][selected].astype(np.int64)
        if limit is not None:
            x, y = x[:limit], y[:limit]
        return x, y

    x_train, y_train = clean(train_idx, train_images, train_labels)
    x_validation, y_validation = clean(validation_idx, train_images, train_labels)
    test_selected = test_labels != 0
    return {
        "x_train": x_train, "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": test_images[test_selected].astype(np.float32) / 255.0,
        "y_test": test_labels[test_selected].astype(np.int64),
    }


def detect_blank(cell_image: np.ndarray, threshold: float = 0.05) -> bool:
    """Return true when mean brightness is below 0.05 on a 0..1 image.

    Raw 0..255 images are converted to the same scale before applying the
    threshold. Blank-cell detection does not use the neural network.
    """
    values = np.asarray(cell_image, dtype=float)
    if values.size == 0:
        raise ValueError("cell_image cannot be empty")
    mean = values.mean() / (255.0 if values.max(initial=0) > 1.0 else 1.0)
    return bool(mean < threshold)


def _choose_image(label: int, images: np.ndarray, labels: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    candidates = np.flatnonzero(labels == label)
    if len(candidates) == 0:
        raise ValueError(f"No MNIST image exists for label {label}")
    return np.asarray(images[int(rng.choice(candidates))])


def build_mosaic(
    grid: np.ndarray, images: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Build a 252x252 MNIST mosaic; zero grid cells become all-zero pixels."""
    grid = np.asarray(grid, dtype=int)
    if grid.shape != (9, 9) or np.asarray(images).shape[1:] != (28, 28):
        raise ValueError("expected a 9x9 grid and MNIST images shaped (n, 28, 28)")
    mosaic = np.zeros((252, 252), dtype=np.uint8)
    for row in range(9):
        for col in range(9):
            if grid[row, col]:
                cell = _choose_image(int(grid[row, col]), images, labels, rng)
                cell = np.asarray(cell, dtype=np.uint8)
                mosaic[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = cell
    return mosaic


def inject_hard_misreads(
    puzzle: np.ndarray, solution: np.ndarray, count: int, images: np.ndarray,
    labels: np.ndarray, model, seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    """Inject real test images that the supplied MLP misclassifies.

    Candidates are ranked first when their wrong prediction conflicts with an
    existing row, column, or box peer, then by model confidence.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    rng = np.random.default_rng(seed)
    puzzle = np.asarray(puzzle, dtype=int)
    solution = np.asarray(solution, dtype=int)
    blanks = list(zip(*np.where(puzzle == 0)))
    if count > len(blanks):
        raise ValueError("count exceeds number of blank cells")
    normalized = np.asarray(images, dtype=float) / (255.0 if np.asarray(images).max() > 1 else 1.0)
    probabilities = model.predict_proba(normalized)
    predictions = probabilities.argmax(axis=1) + 1
    selected: list[tuple[int, int, np.ndarray, int, float]] = []
    available = set(blanks)
    for row, col in blanks:
        true_digit = int(solution[row, col])
        true_ranks = np.argsort(probabilities, axis=1)[:, ::-1]
        indices = np.flatnonzero(
            (labels == true_digit)
            & (predictions != true_digit)
            & np.any(true_ranks[:, :3] == true_digit - 1, axis=1)
        )
        ranked = []
        for index in indices:
            wrong = int(predictions[index])
            peers = set(puzzle[row, :]) | set(puzzle[:, col])
            peers |= set(puzzle[row // 3 * 3 : row // 3 * 3 + 3,
                                col // 3 * 3 : col // 3 * 3 + 3].ravel())
            ranked.append((int(wrong in peers), -float(probabilities[index, wrong - 1]),
                           int(index)))
        if ranked:
            ranked.sort()
            _, negative_confidence, index = ranked[0]
            selected.append((row, col, images[index], int(predictions[index]), -negative_confidence))
    if len(selected) < count:
        raise ValueError("trained MLP has fewer hard misread candidates than requested")
    rng.shuffle(selected)
    selected = selected[:count]
    cell_images: dict[tuple[int, int], np.ndarray] = {}
    for row, col in blanks:
        true_digit = int(solution[row, col])
        candidates = np.flatnonzero((labels == true_digit) & (predictions == true_digit))
        if len(candidates) == 0:
            candidates = np.flatnonzero(labels == true_digit)
        cell_images[(row, col)] = images[int(rng.choice(candidates))]
    observed = puzzle.copy()
    log: list[tuple[int, int, int, int]] = []
    for row, col, image, read_digit, _ in selected:
        observed[row, col] = read_digit
        cell_images[(row, col)] = image
        log.append((int(row), int(col), read_digit, int(solution[row, col])))
    mosaic = np.zeros((252, 252), dtype=np.uint8)
    for (row, col), image in cell_images.items():
        mosaic[row * 28 : (row + 1) * 28, col * 28 : (col + 1) * 28] = image
    return observed, mosaic, log


def solved_grid(rng: np.random.Generator) -> np.ndarray:
    """Create a valid Sudoku solution using randomized backtracking."""
    grid = np.zeros((9, 9), dtype=int)

    def fill(index: int) -> bool:
        if index == 81:
            return True
        row, col = divmod(index, 9)
        values = list(range(1, 10))
        rng.shuffle(values)
        used = set(grid[row, :]) | set(grid[:, col])
        used |= set(grid[row // 3 * 3 : row // 3 * 3 + 3, col // 3 * 3 : col // 3 * 3 + 3].ravel())
        for value in values:
            if value not in used:
                grid[row, col] = value
                if fill(index + 1):
                    return True
        grid[row, col] = 0
        return False

    if not fill(0):
        raise RuntimeError("Could not generate a Sudoku solution")
    return grid


def _has_unique_solution(puzzle: np.ndarray) -> bool:
    from csp.sudoku import SudokuSolver

    solver = SudokuSolver(puzzle)
    return solver.count_solutions(limit=2) == 1


def generate_puzzle(seed: int = 7, givens: int = 35) -> tuple[np.ndarray, np.ndarray]:
    """Return (puzzle, solution), removing cells while preserving uniqueness."""
    if not 17 <= givens <= 81:
        raise ValueError("givens must be between 17 and 81")
    rng = np.random.default_rng(seed)
    solution = solved_grid(rng)
    puzzle = solution.copy()
    order = list(range(81))
    rng.shuffle(order)
    for index in order:
        if np.count_nonzero(puzzle) <= givens:
            break
        row, col = divmod(index, 9)
        old = puzzle[row, col]
        puzzle[row, col] = 0
        if not _has_unique_solution(puzzle):
            puzzle[row, col] = old
    return puzzle, solution


def inject_misreads(
    puzzle: np.ndarray, solution: np.ndarray, count: int, seed: int = 7
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Replace selected blanks with wrong digits and return an error log."""
    if count < 0:
        raise ValueError("count cannot be negative")
    rng = np.random.default_rng(seed)
    observed = puzzle.copy()
    blanks = list(zip(*np.where(puzzle == 0)))
    if count > len(blanks):
        raise ValueError("count exceeds number of blank cells")
    chosen = rng.choice(len(blanks), size=count, replace=False)
    log: list[tuple[int, int, int, int]] = []
    for selected in np.atleast_1d(chosen):
        row, col = blanks[int(selected)]
        used = set(puzzle[row, :]) | set(puzzle[:, col])
        used |= set(puzzle[row // 3 * 3 : row // 3 * 3 + 3,
                            col // 3 * 3 : col // 3 * 3 + 3].ravel())
        alternatives = [digit for digit in used if digit and digit != solution[row, col]]
        if not alternatives:
            alternatives = [digit for digit in range(1, 10) if digit != solution[row, col]]
        wrong = int(rng.choice(alternatives))
        observed[row, col] = wrong
        log.append((int(row), int(col), wrong, int(solution[row, col])))
    return observed, log


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--path", default="data/mnist")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/prepared_data.npz")
    args = parser.parse_args()
    if args.download:
        print(f"MNIST stored in {download_mnist(args.path)}")
    if args.prepare:
        prepared = prepare_mnist(args.path, seed=args.seed)
        np.savez_compressed(args.output, **prepared)
        print(f"Prepared data saved to {args.output}")
    puzzle, solution = generate_puzzle()
    print("Puzzle:\n", puzzle)
    print("Solution:\n", solution)


if __name__ == "__main__":
    main()
