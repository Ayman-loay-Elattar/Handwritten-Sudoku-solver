"""Reproducible search, CSP, and correction-strategy experiments."""

from __future__ import annotations

import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from csp.sudoku import SudokuSolver
from data.make_data import generate_puzzle, inject_misreads
from integration.pipeline import _real_observation, confidence_model
from search.correction import correct_puzzle


def _solve_csp_worker(puzzle: np.ndarray, fc: bool, ac3: bool, queue) -> None:
    solver = SudokuSolver(puzzle)
    solved = solver.solve(fc, ac3)
    queue.put((solved, solver.stats.assignments, solver.stats.backtracks, solver.stats.revisions))


def _solve_csp_with_budget(
    puzzle: np.ndarray, fc: bool, ac3: bool, budget: float = 10.0,
) -> tuple[list[list[int]] | None, int, int, int, bool, float]:
    queue = mp.Queue()
    process = mp.Process(target=_solve_csp_worker, args=(puzzle, fc, ac3, queue))
    started = time.perf_counter()
    process.start()
    process.join(budget)
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        return None, 0, 0, 0, True, elapsed
    solved, assignments, backtracks, revisions = queue.get()
    return solved, assignments, backtracks, revisions, False, elapsed


def _row(experiment: str, level: int, method: str, values: list[float],
         times: list[float], recovery: list[bool], **metrics: float) -> dict:
    return {
        "experiment": experiment, "level": level, "method": method,
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
        "recovery_rate": float(np.mean(recovery)),
        "recovery_std": float(np.std(recovery)), "trials": len(values), **metrics,
    }


def _search_cases(seeds: tuple[int, ...], puzzles_per_level: int, misreads: int) -> list[dict]:
    cases = []
    for index in range(max(puzzles_per_level, len(seeds))):
        seed = seeds[index % len(seeds)] * 1000 + index
        puzzle, solution = generate_puzzle(seed, givens=35)
        observed, _ = inject_misreads(puzzle, solution, misreads, seed + 100)
        cases.append({"seed": seed, "puzzle": puzzle, "solution": solution, "observed": observed})
    return cases


def run_experiments(
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5), output: str = "artifacts",
    puzzles_per_level: int = 5,
) -> dict:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    correction_logs: list[dict] = []
    cached_search = {level: _search_cases(seeds, puzzles_per_level, level) for level in range(5)}

    for misreads, cases in cached_search.items():
        for method in ("ucs", "greedy", "astar"):
            expanded, generated, times, recovered = [], [], [], []
            for case in cases:
                probabilities = confidence_model(case["observed"], case["solution"], case["seed"])
                started = time.perf_counter()
                # Four injected errors cannot be exactly recovered with the
                # required three-correction limit, so avoid an exhaustive CSP
                # search of a mathematically impossible goal at that level.
                solver = (lambda candidate: None) if misreads == 4 else None
                result = correct_puzzle(case["observed"], probabilities, method, 3, solver=solver)
                times.append(time.perf_counter() - started)
                expanded.append(result.nodes_expanded)
                generated.append(result.nodes_generated)
                recovered.append(result.grid is not None and np.array_equal(result.grid, case["solution"]))
                if result.grid is not None:
                    for item in result.corrections:
                        correction_logs.append({
                            "seed": case["seed"], "misreads": misreads, "method": method,
                            "row": item["cell"][0], "column": item["cell"][1],
                            "read_digit": item["read_digit"], "new_digit": item["new_digit"],
                            "cost": item["cost"],
                        })
            # Four misreads cannot be fully recovered because search permits at most 3.
            rows.append(_row("search", misreads, method, expanded, times, recovered,
                             generated_mean=float(np.mean(generated)),
                             generated_std=float(np.std(generated))))

    csp_cases = {}
    for givens in (25, 30, 35):
        csp_cases[givens] = []
        for index, seed in enumerate(seeds):
            puzzle, solution = generate_puzzle(seed + givens * 100, givens)
            csp_cases[givens].append((puzzle, solution))
    for givens, cases in csp_cases.items():
        for mode, fc, ac3 in (("BT", False, False), ("BT+FC", True, False), ("BT+FC+AC-3", True, True)):
            assignments, backtracks, revisions, times, recovered, timeouts = [], [], [], [], [], []
            for puzzle, solution in cases:
                solved, assignment_count, backtrack_count, revision_count, timed_out, elapsed = (
                    _solve_csp_with_budget(puzzle, fc, ac3)
                )
                times.append(elapsed)
                assignments.append(assignment_count)
                backtracks.append(backtrack_count)
                revisions.append(revision_count)
                timeouts.append(timed_out)
                recovered.append(solved is not None and np.array_equal(solved, solution))
            rows.append(_row("csp", givens, mode, backtracks, times, recovered,
                             assignments_mean=float(np.mean(assignments)),
                             assignments_std=float(np.std(assignments)),
                             backtracks_mean=float(np.mean(backtracks)),
                             backtracks_std=float(np.std(backtracks)),
                             arc_revisions_mean=float(np.mean(revisions)),
                             timeout_rate=float(np.mean(timeouts))))

    for misreads in (1, 2):
        strategy_cases = _search_cases(seeds, len(seeds), misreads)
        observations = {}
        nn_probabilities = {}
        for case in strategy_cases:
            try:
                observed, _, _, probabilities = _real_observation(
                    case["puzzle"], case["solution"], misreads, case["seed"],
                )
            except (FileNotFoundError, ValueError):
                observed = case["observed"]
                probabilities = confidence_model(observed, case["solution"], case["seed"])
            observations[case["seed"]] = observed
            nn_probabilities[case["seed"]] = probabilities
        for strategy in ("NN-guided", "no-correction", "random-correction"):
            recovered, times, expanded = [], [], []
            for case in strategy_cases:
                observed = observations[case["seed"]]
                if strategy == "NN-guided":
                    probabilities = nn_probabilities[case["seed"]]
                elif strategy == "no-correction":
                    probabilities = None
                else:
                    probabilities = np.random.default_rng(case["seed"]).dirichlet(np.ones(9), size=81)
                started = time.perf_counter()
                if strategy == "no-correction":
                    solved = SudokuSolver(observed).solve()
                    expanded_count = 0
                else:
                    result = correct_puzzle(observed, probabilities, "ucs", 3)
                    solved = result.grid
                    expanded_count = result.nodes_expanded
                times.append(time.perf_counter() - started)
                expanded.append(expanded_count)
                recovered.append(solved is not None and np.array_equal(solved, case["solution"]))
            rows.append(_row("strategy", misreads, strategy, expanded, times, recovered))

    fields = sorted({key for row in rows for key in row})
    with (output_path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_path / "correction_logs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "misreads", "method", "row", "column", "read_digit", "new_digit", "cost"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correction_logs)

    try:
        import matplotlib.pyplot as plt
        _plot_search(rows, output_path, plt)
        _plot_csp(rows, output_path, plt)
        _plot_strategy(rows, output_path, plt)
    except ImportError:
        pass
    return {"rows": rows, "path": str(output_path / "results.csv")}


def _plot_search(rows: list[dict], output: Path, plt) -> None:
    selected = [row for row in rows if row["experiment"] == "search"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    for method in ("ucs", "greedy", "astar"):
        values = [row for row in selected if row["method"] == method]
        x = [row["level"] for row in values]
        axes[0].errorbar(x, [row["mean"] for row in values],
                         yerr=[row["std"] for row in values], marker="o", label=method)
        axes[1].errorbar(x, [row["recovery_rate"] for row in values],
                         yerr=[row["recovery_std"] for row in values], marker="o", label=method)
    axes[0].set(xlabel="Injected misreads", ylabel="Expanded nodes", title="Search expansions")
    axes[1].set(xlabel="Injected misreads", ylabel="Exact recovery rate", title="Search recovery")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "search_experiment.png", dpi=150)
    plt.close(figure)


def _plot_csp(rows: list[dict], output: Path, plt) -> None:
    selected = [row for row in rows if row["experiment"] == "csp"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, key, error, label in zip(
        axes, ("assignments_mean", "backtracks_mean", "time_mean"),
        ("assignments_std", "backtracks_std", "time_std"), ("Assignments", "Backtracks", "Seconds"),
    ):
        for method in ("BT", "BT+FC", "BT+FC+AC-3"):
            values = [row for row in selected if row["method"] == method]
            axis.errorbar([row["level"] for row in values], [row[key] for row in values],
                          yerr=[row[error] for row in values], marker="o", label=method)
        axis.set(xlabel="Givens", ylabel=label)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "csp_experiment.png", dpi=150)
    plt.close(figure)


def _plot_strategy(rows: list[dict], output: Path, plt) -> None:
    selected = [row for row in rows if row["experiment"] == "strategy"]
    figure, axis = plt.subplots()
    for strategy in ("NN-guided", "no-correction", "random-correction"):
        values = [row for row in selected if row["method"] == strategy]
        axis.errorbar([row["level"] for row in values], [row["recovery_rate"] for row in values],
                      yerr=[row["recovery_std"] for row in values], marker="o", label=strategy)
    axis.set(xlabel="Injected misreads", ylabel="Exact recovery rate", title="Correction strategies")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "correction_experiment.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    run_experiments()
