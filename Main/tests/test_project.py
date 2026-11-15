import numpy as np
import pytest

from csp.sudoku import SudokuSolver
from data.make_data import generate_puzzle, inject_misreads
from integration.pipeline import confidence_model, run_pipeline
from search.correction import correct_puzzle


VALID = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])


@pytest.mark.parametrize("mode", [(False, False), (True, False), (True, True)])
def test_csp_modes_solve(mode):
    result = SudokuSolver(VALID).solve(*mode)
    assert result is not None
    assert all(sorted(row) == list(range(1, 10)) for row in result)


def test_generated_puzzle_is_reproducible_and_unique():
    first = generate_puzzle(11, 35)
    second = generate_puzzle(11, 35)
    assert np.array_equal(first[0], second[0])
    assert SudokuSolver(first[0]).count_solutions() == 1


def test_search_corrects_a_misread():
    puzzle, solution = generate_puzzle(13, 35)
    observed, _ = inject_misreads(puzzle, solution, 1, 14)
    result = correct_puzzle(observed, confidence_model(observed, solution, 13), max_corrections=2)
    assert result.message == "solution found"
    assert np.array_equal(result.grid, solution)
    assert result.nodes_generated >= result.nodes_expanded


def test_pipeline_has_correction_log():
    result = run_pipeline(17, 2)
    assert result["result"].message == "solution found"
    assert len(result["result"].corrections) <= 3


def test_impossible_duplicate_givens_returns_no_solution():
    impossible = VALID.copy()
    impossible[0, 2] = 5
    assert SudokuSolver(impossible).solve() is None


def test_impossible_conflicting_full_grid_returns_no_solution():
    impossible = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 5],
    ])
    assert SudokuSolver(impossible).explain() == "no solution"


def test_impossible_shape_is_explicit():
    with pytest.raises(ValueError, match="9x9"):
        SudokuSolver([[1]])


def test_invalid_misread_count_is_explicit():
    with pytest.raises(ValueError, match="exceeds"):
        inject_misreads(VALID, VALID, 100, 1)


def test_invalid_search_method_is_explicit():
    with pytest.raises(ValueError, match="method"):
        correct_puzzle(VALID, np.ones((81, 9)) / 9, "bad")


def test_search_known_small_answer():
    puzzle = VALID.copy()
    puzzle[0, 2] = 5
    probabilities = np.ones((81, 9)) / 9
    probabilities[2, 3] = 0.9
    result = correct_puzzle(puzzle, probabilities, "ucs", 1)
    assert result.message == "solution found"


def test_pipeline_flags_and_methods_are_observable():
    first = run_pipeline(7, 1, "ucs", True, True, True)
    second = run_pipeline(8, 3, "greedy", False, False, False)
    assert not np.array_equal(first["observed"], second["observed"])
    assert first["nn_enabled"] is True and second["nn_enabled"] is False
    assert first["result"].method != second["result"].method
    assert all(item["cost"] >= 0 for item in first["result"].corrections)
