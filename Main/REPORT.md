# P7 Handwritten Sudoku Reader and Solver

## Problem

The project reads a 9x9 Sudoku containing simulated handwritten-digit
misreads. A digit classifier supplies probabilities, correction search changes
conflicting uncertain cells, and a Sudoku constraint solver validates the
candidate. Blank cells are generated puzzle blanks rather than classifier
predictions.

## Design choices

The code is split into `data`, `nn`, `search`, `csp`, `integration`, `eval`, and
`tests`. NumPy was chosen to keep the project reproducible in a clean venv.
UCS, Greedy, and A* share one correction-state implementation. Correction costs
are `-log(probability)` and are clipped at zero. The A* heuristic is the
minimum available correction cost among remaining conflicts, so it cannot
overestimate the cost of at least one required correction. The search limit is
three corrections.

## Neural network results

Command run:

```powershell
.\.venv\Scripts\python.exe -m nn.train --path D:\setups\dataset --limit 10000
```

The supplied IDX files were read successfully. The source split is 50,000
training and 10,000 validation examples before dropping class 0; the test set
is untouched. With the requested 10,000 training limit, 10,000 nonzero
training examples, 9,002 nonzero validation examples, and 9,020 nonzero test
examples were used.

| Model | Test accuracy | Calibration error |
|---|---:|---:|
| NumPy MLP | 0.8972 | 0.0633 |
| Logistic regression | 0.8929 | 0.0881 |

The full 9x9 MLP confusion matrix is in
`artifacts/nn_metrics.json` and `artifacts/nn_confusion_matrix.csv`.

## Search and CSP results

The five-seed default experiment was run with:

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

Search results below are expanded-node mean +/- standard deviation from
`artifacts/audit_run/results.csv`:

| Misreads | UCS | Greedy | A* |
|---:|---:|---:|---:|
| 0 | 1.0 +/- 0.0 | 1.0 +/- 0.0 | 1.0 +/- 0.0 |
| 1 | 2.0 +/- 0.0 | 2.0 +/- 0.0 | 2.0 +/- 0.0 |
| 2 | 4.0 +/- 0.0 | 4.0 +/- 0.0 | 4.0 +/- 0.0 |
| 3 | 11.0 +/- 0.0 | 9.0 +/- 0.0 | 11.0 +/- 0.0 |
| 4 | 15181.6 +/- 3708.0 | 15181.6 +/- 3708.0 | 15181.6 +/- 3708.0 |

CSP outputs include assignments, backtracks, arc revisions, and elapsed time
for 25, 30, and 35 givens in `artifacts/results.csv`. AC-3 substantially
reduces assignments and backtracks in these generated cases. The strategy
experiment records measured NN-guided, no-correction, and random-correction
recovery rather than illustrative constants.

## Integration and GUI

`run_demo.py` builds and classifies a real MNIST mosaic when
`artifacts/mlp_weights.npz` and MNIST are available; it prints the observed
input, correction log, counters, and solution, then writes `artifacts/demo.png`.
`gui.py` provides three boards, red conflict
and green correction highlighting, a scrollable correction/search trace, live
BT/FC/AC-3 and NN switches, counters, validation messages, and a Step button.
The GUI's NN switch changes between real MLP probabilities and uniform
probabilities. If the real assets are unavailable, the pipeline prints
`FALLBACK: simulated probabilities, not real NN` and uses the documented
simulation fallback.

## Limits and conclusions

The Sudoku layouts and injected error locations are generated; the image
pipeline uses real MNIST test images and is not a photographed-page benchmark.
Full exact recovery on 100 puzzles per error level is available with
`run_experiments.py --puzzles-per-level 100 --output artifacts/full_run`.
The default five-seed run is shorter for routine validation and writes separate
expanded-node, generated-node, recovery, CSP, strategy, and correction-log
columns/files.
