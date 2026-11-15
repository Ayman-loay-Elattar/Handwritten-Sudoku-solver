"""Uniform-cost, greedy, and A* correction search."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from csp.sudoku import SudokuSolver


@dataclass
class CorrectionResult:
    grid: np.ndarray | None
    corrections: list[dict]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    method: str
    message: str
    trace: list[dict] = field(default_factory=list)


def _conflicts(grid: np.ndarray) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    units = []
    units.extend([[(row, col) for col in range(9)] for row in range(9)])
    units.extend([[(row, col) for row in range(9)] for col in range(9)])
    units.extend([[(row, col) for row in range(box_row, box_row + 3)
                   for col in range(box_col, box_col + 3)]
                  for box_row in (0, 3, 6) for box_col in (0, 3, 6)])
    for unit in units:
        positions: dict[int, list[tuple[int, int]]] = {}
        for row, col in unit:
            value = int(grid[row, col])
            if value:
                positions.setdefault(value, []).append((row, col))
        for duplicate in positions.values():
            if len(duplicate) > 1:
                cells.update(duplicate)
    return cells


def correct_puzzle(
    observed: np.ndarray,
    probabilities: np.ndarray,
    method: str = "ucs",
    max_corrections: int = 3,
    solver: Callable[[np.ndarray], list[list[int]] | None] | None = None,
) -> CorrectionResult:
    """Find a low-probability-cost set of changes that makes the puzzle solvable."""
    if method not in {"ucs", "greedy", "astar"}:
        raise ValueError("method must be ucs, greedy, or astar")
    if max_corrections < 0:
        raise ValueError("max_corrections cannot be negative")
    max_corrections = min(max_corrections, 3)
    grid = np.asarray(observed, dtype=int).copy()
    probabilities = np.asarray(probabilities, dtype=float)
    if grid.shape != (9, 9) or probabilities.shape != (81, 9):
        raise ValueError("expected grid (9,9) and probabilities (81,9)")
    solve = solver or (lambda candidate: SudokuSolver(candidate).solve())
    alternatives: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row, col in _conflicts(grid):
        current = int(grid[row, col])
        choices = [(digit + 1, max(0.0, -float(np.log(np.clip(probabilities[row * 9 + col, digit], 1e-12, 1)))))
                   for digit in range(9) if digit + 1 != current]
        alternatives[(row, col)] = sorted(choices, key=lambda item: item[1])[:3]
    if not alternatives:
        solution = solve(grid)
        return CorrectionResult(np.array(solution) if solution else None, [], 0.0, 1, 1, method,
                                "solution found" if solution else "no solution")

    counter = 0
    queue: list[tuple[float, int, np.ndarray, list[dict], set[tuple[int, int]]]] = []
    heapq.heappush(queue, (0.0, counter, grid, [], set()))
    generated = 1
    trace = [{"cost": 0.0, "corrections": []}]
    while queue:
        priority, _, state, corrections, changed = heapq.heappop(queue)
        conflicts = _conflicts(state)
        solution = solve(state)
        trace.append({"cost": float(priority), "corrections": list(corrections)})
        if solution is not None:
            return CorrectionResult(np.array(solution), corrections, sum(item["cost"] for item in corrections),
                                    generated - len(queue), generated, method, "solution found", trace)
        if len(corrections) >= max_corrections:
            continue
        targets = conflicts or set(alternatives)
        for cell in targets:
            if cell in changed or cell not in alternatives:
                continue
            row, col = cell
            for digit, cost in alternatives[cell]:
                child = state.copy()
                old = int(child[row, col])
                child[row, col] = digit
                entry = {"cell": [row, col], "read_digit": old, "new_digit": digit, "cost": cost}
                child_corrections = corrections + [entry]
                heuristic = 0.0
                if method == "astar":
                    remaining = _conflicts(child)
                    heuristic = min((alternatives[cell][0][1] for cell in remaining if cell in alternatives), default=0.0)
                if method == "greedy":
                    score = cost + heuristic
                else:
                    score = sum(item["cost"] for item in child_corrections) + heuristic
                counter += 1
                heapq.heappush(queue, (score, counter, child, child_corrections, changed | {cell}))
                generated += 1
    return CorrectionResult(None, [], float("inf"), generated - len(queue), generated, method, "no solution", trace)
