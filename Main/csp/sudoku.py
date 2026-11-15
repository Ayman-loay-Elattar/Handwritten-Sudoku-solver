"""Sudoku CSP with switchable backtracking, forward checking, and AC-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

ALL = frozenset(range(1, 10))
CELLS = tuple(range(81))


def peers(cell: int) -> frozenset[int]:
    row, col = divmod(cell, 9)
    result = set()
    result.update(row * 9 + c for c in range(9) if c != col)
    result.update(r * 9 + col for r in range(9) if r != row)
    box_row, box_col = row // 3 * 3, col // 3 * 3
    result.update(r * 9 + c for r in range(box_row, box_row + 3) for c in range(box_col, box_col + 3)
                 if (r, c) != (row, col))
    return frozenset(result)


@dataclass
class SolveStats:
    assignments: int = 0
    backtracks: int = 0
    revisions: int = 0


class SudokuSolver:
    def __init__(self, puzzle: Iterable[Iterable[int]]):
        self.grid = self._validate_grid(puzzle)
        self.stats = SolveStats()

    @staticmethod
    def _validate_grid(puzzle: Iterable[Iterable[int]]) -> list[list[int]]:
        grid = [list(row) for row in puzzle]
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("Sudoku must be a 9x9 grid")
        if any(value < 0 or value > 9 for row in grid for value in row):
            raise ValueError("Sudoku values must be integers from 0 to 9")
        return grid

    def _domains(self) -> Dict[int, set[int]]:
        domains = {cell: set(ALL) for cell in CELLS}
        for cell, value in enumerate(sum(self.grid, [])):
            if value:
                domains[cell] = {value}
        return domains

    @staticmethod
    def _consistent(domains: Dict[int, set[int]]) -> bool:
        if not all(domains[cell] for cell in CELLS):
            return False
        return all(
            not (len(domains[cell]) == len(domains[peer]) == 1 and domains[cell] == domains[peer])
            for cell in CELLS for peer in peers(cell)
        )

    def ac3(self, domains: Dict[int, set[int]]) -> bool:
        queue = [(x, y) for x in CELLS for y in peers(x)]
        while queue:
            x, y = queue.pop(0)
            if self._revise(domains, x, y):
                self.stats.revisions += 1
                if not domains[x]:
                    return False
                queue.extend((z, x) for z in peers(x) if z != y)
        return self._consistent(domains)

    @staticmethod
    def _revise(domains: Dict[int, set[int]], x: int, y: int) -> bool:
        if len(domains[y]) != 1:
            return False
        value = next(iter(domains[y]))
        if value in domains[x] and len(domains[x]) > 1:
            domains[x].remove(value)
            return True
        return False

    def solve(self, use_forward_checking: bool = True, use_ac3: bool = True) -> list[list[int]] | None:
        self.stats = SolveStats()
        domains = self._domains()
        if use_ac3 and not self.ac3(domains):
            return None
        if not use_ac3 and not self._consistent(domains):
            return None
        result = self._backtrack(domains, use_forward_checking, use_ac3)
        if result is None:
            return None
        return [[next(iter(result[row * 9 + col])) for col in range(9)] for row in range(9)]

    def _backtrack(self, domains: Dict[int, set[int]], fc: bool, ac3: bool) -> Dict[int, set[int]] | None:
        unresolved = [cell for cell in CELLS if len(domains[cell]) > 1]
        if not unresolved:
            return domains
        cell = min(unresolved, key=lambda item: len(domains[item]))
        for value in sorted(domains[cell]):
            self.stats.assignments += 1
            branch = {key: set(vals) for key, vals in domains.items()}
            branch[cell] = {value}
            valid = self.ac3(branch) if ac3 else (self._forward_check(branch, cell) if fc else self._consistent(branch))
            if valid:
                result = self._backtrack(branch, fc, ac3)
                if result is not None:
                    return result
            self.stats.backtracks += 1
        return None

    @staticmethod
    def _forward_check(domains: Dict[int, set[int]], cell: int) -> bool:
        value = next(iter(domains[cell]))
        for peer in peers(cell):
            if value in domains[peer]:
                if len(domains[peer]) == 1:
                    return False
                domains[peer].remove(value)
        return all(domains[item] for item in CELLS)

    def count_solutions(self, limit: int = 2) -> int:
        count = 0

        def visit(domains: Dict[int, set[int]]) -> None:
            nonlocal count
            if count >= limit:
                return
            if not self.ac3(domains):
                return
            unresolved = [cell for cell in CELLS if len(domains[cell]) > 1]
            if not unresolved:
                count += 1
                return
            cell = min(unresolved, key=lambda item: len(domains[item]))
            for value in sorted(domains[cell]):
                branch = {key: set(vals) for key, vals in domains.items()}
                branch[cell] = {value}
                visit(branch)

        visit(self._domains())
        return count

    def explain(self) -> str:
        result = self.solve()
        return "no solution" if result is None else "solution found"
