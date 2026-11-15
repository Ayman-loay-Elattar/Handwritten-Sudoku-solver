# Handwritten Sudoku Reader and Solver

This is a reproducible Python prototype for the required pipeline:

1. generate a unique Sudoku and simulated handwritten-digit mistakes;
2. build a 252x252 mosaic from real MNIST test images, detect blank cells by
   brightness, and classify nonblank cells with the trained NumPy MLP; NN can
   be switched off in the GUI;
3. search for low-cost corrections with UCS, Greedy, or A*; and
4. validate candidates with Sudoku CSP backtracking, Forward Checking, and AC-3.

The project is intentionally transparent: every stage has its own module,
counters, tests, and generated outputs. Sudoku layouts are generated, while
the cell images in the real-assets path are MNIST test images.

## 1. Install

Use PowerShell from this folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.11+ is supported. If `py -3.14` is unavailable, use the Python
executable installed on your machine.

## 2. Run the GUI (recommended)

```powershell
python gui.py
```

The GUI shows the observed NN input, corrected CSP result, and ground truth.
Choose a seed, 0-3 injected misreads, and a search algorithm, then click
**Run pipeline**. The status line shows whether a solution was found and the
search node counters. The correction log appears below the boards.

## 3. Run the text/image demo

```powershell
python run_demo.py
```

This prints the observed grid, search method, expanded/generated nodes, status,
and correction log. It creates `artifacts/demo.png`. If correction is
impossible, the command still completes and labels the middle panel as the
unmodified input rather than crashing.

## 4. Run experiments

```powershell
python run_experiments.py
```

This runs five fast seeded trials per level by default and writes mean, standard deviation, and recovery rate to
`artifacts/results.csv`. It also creates:

* `artifacts/search_experiment.png`: UCS vs Greedy vs A*;
* `artifacts/csp_experiment.png`: BT vs BT+FC vs BT+FC+AC-3;
* `artifacts/correction_experiment.png`: correction strategy visualization.

The search experiment covers 0-4 injected errors. The CSP experiment covers
25, 30, and 35 givens.

For the project-specific exact-recovery requirement of 100 puzzles per
misread level, run:

```powershell
python run_experiments.py --puzzles-per-level 100
```

This is more expensive because it performs the actual correction search on
every puzzle.

## 5. Train and evaluate the MNIST classifiers

MNIST is optional and is not needed for the GUI/demo:

```powershell
python make_data.py --download --path data/mnist
python make_data.py --prepare --path data/mnist --output artifacts/prepared_data.npz
python -m nn.train --path D:\setups\dataset --limit 10000
```

The first command downloads the four public IDX files. The second scales
pixels, makes the fixed-seed 50,000/10,000 source split, drops digit 0, and
saves a compressed prepared-data file. The third trains the NumPy MLP and
logistic-regression baseline on the nonzero training subset and evaluates the
untouched MNIST test set. It prints accuracy and calibration error and saves
`artifacts/nn_metrics.json`, `artifacts/nn_confusion_matrix.csv`, and
`artifacts/mlp_weights.npz`, which the real image pipeline loads.

## 6. Run tests

```powershell
python -m pytest -q
```

The suite currently covers 16 cases: normal CSP modes, reproducible generation,
correction search, integration, an impossible duplicate-given puzzle,
invalid input, a known small search answer, and pipeline option changes.
The tests are distributed across `tests/test_project.py` and
`tests/test_fixes.py`, with at least 5 normal, 3 edge, and 2 impossible cases.
Impossible cases explicitly return `no solution`.

## Requirement coverage

| Requirement | Implementation |
|---|---|
| Separate data/NN/search/CSP/integration/eval/tests modules | `data/`, `nn/`, `search/`, `csp/`, `integration/`, `eval/`, `tests/` |
| Reproducible data generation | `data/make_data.py`, fixed seeds |
| Data dictionary | `data/data_dictionary.md` |
| NN and baseline, metrics artifact | `nn/classifier.py`, `nn/train.py`, `artifacts/nn_metrics.json` |
| Search counters and known test | `search/correction.py`, `tests/test_project.py` |
| BT, FC, AC-3 switchable CSP | `csp/sudoku.py` |
| Integrated demo and GUI | `run_demo.py`, `gui.py`; real image assets are used when weights and MNIST are present |
| Pinned dependencies | `requirements.txt` |
| Normal, edge, and impossible tests | `tests/` (`python -m pytest -q`) |
| Five-seed tables and plots, assignments/backtracks/time | `eval/experiments.py`, `artifacts/` |
| Report | `REPORT.md` |
| Contribution statement | `CONTRIBUTIONS.md` |

The original presentation and git-log requirement is a delivery activity:
slides/live demonstration should use the GUI and generated artifacts. This
workspace is not a Git repository, so a git log cannot be generated here.
