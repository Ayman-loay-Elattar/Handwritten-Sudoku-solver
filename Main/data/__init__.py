"""Data loading and reproducible Sudoku/MNIST data generation."""

from .make_data import generate_puzzle, inject_misreads

__all__ = ["generate_puzzle", "inject_misreads"]
