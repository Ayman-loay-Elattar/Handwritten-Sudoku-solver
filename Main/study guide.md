# Study Guide: Handwritten Sudoku Reader and Solver

## 1. What the project does

This project reads a Sudoku puzzle containing handwritten-style digit images,
detects the digits, corrects likely recognition mistakes, and solves the
Sudoku.

The complete pipeline is:

1. Generate a valid Sudoku puzzle and its solution.
2. Create a 9x9 grid of digit images. The images come from MNIST when the
   real-data files and trained model are available.
3. Use a NumPy neural network to classify each nonblank digit as 1-9.
4. Detect conflicts caused by incorrect classifications.
5. Search for inexpensive corrections using Uniform-Cost Search (UCS),
   Greedy Search, or A*.
6. Validate and solve each candidate with a Sudoku constraint-satisfaction
   problem (CSP) solver.
7. Display the observed grid, corrected solution, ground truth, correction
   log, and search/CSP counters.

The project is a reproducible prototype rather than a photographed-page
recognition system. Sudoku layouts and injected mistakes are generated, while
the optional digit images are real MNIST images.

## 2. Important project parts to discuss

### Data generation

`data/make_data.py` generates a complete Sudoku solution, removes cells while
preserving a unique solution, and can inject simulated misreads. It also
downloads and prepares MNIST data.

Important points:

- Sudoku puzzles are generated with a random seed.
- A puzzle normally contains 17-81 givens.
- Blank cells are represented by `0`.
- MNIST images are 28x28 pixels.
- Pixel values are scaled from 0-255 to 0-1.
- Digit 0 is removed because Sudoku uses digits 1-9.

### Neural network and baseline

`nn/classifier.py` and `nn/train.py` contain a NumPy multilayer perceptron
(MLP) and a logistic-regression baseline.

The MLP classifies each image and produces probabilities for digits 1-9.
Those probabilities are important because they become correction costs:

```text
correction cost = -log(probability)
```

A likely alternative has a smaller cost. The project compares the MLP with
logistic regression using accuracy, calibration error, and a confusion matrix.

### Search algorithms

`search/correction.py` treats a possible correction set as a search state.
An action changes one uncertain cell to an alternative digit.

- **UCS:** expands the lowest total correction cost first.
- **Greedy:** expands the state with the lowest estimated remaining cost.
- **A\*:** combines the cost already paid with a conflict-based heuristic.

The goal is to find a corrected puzzle that the CSP solver can solve. The
normal correction limit is three changes.

Important counters are:

- `nodes expanded`: states actually processed.
- `nodes generated`: candidate states created.
- correction cost: how unlikely the changed digit is according to the model.

### Sudoku CSP solver

`csp/sudoku.py` models the puzzle as 81 variables with values 1-9.
Cells in the same row, column, or 3x3 box must have different values.

- **Backtracking (BT):** assigns values and goes back when a choice fails.
- **Forward Checking (FC):** removes an assigned value from neighboring
  domains before continuing.
- **AC-3:** applies arc consistency to shrink domains and expose conflicts
  earlier.

The solver reports assignments, backtracks, arc revisions, elapsed time, and
whether a solution was found.

### Integration and GUI

`integration/pipeline.py` connects recognition, conflict detection, search,
and CSP solving.

`gui.py` provides three boards:

- Observed NN input
- Corrected CSP result
- Ground truth

The GUI also shows conflict/correction highlighting, search traces, counters,
and switches for NN, BT, FC, AC-3, and the search method.

### Testing and experiments

The tests cover normal solutions, reproducibility, correction search,
impossible puzzles, invalid inputs, and pipeline options.

The experiments compare:

1. UCS, Greedy, and A* for different numbers of misreads.
2. BT, BT+FC, and BT+FC+AC-3 for different numbers of givens.
3. NN-guided correction, no correction, and random correction.

## 3. Installation commands

Run these commands from the project folder in PowerShell.

### Create a virtual environment

```powershell
py -3.14 -m venv .venv
```

Expected output:

```text
Usually no output when the environment is created successfully.
```

Explanation: This creates an isolated Python environment in `.venv`. If
Python 3.14 is not installed, replace `py -3.14` with the Python version
installed on the computer.

### Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected output:

```text
The prompt normally changes to show (.venv).
```

Explanation: Commands using `python` now use the project environment instead
of the system-wide Python installation.

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Expected output:

```text
Successfully installed numpy matplotlib pytest
```

Explanation: NumPy is used for numerical work and the neural network,
Matplotlib creates plots, and pytest runs the automated tests. Exact version
and installation messages may vary.

## 4. Main commands and expected output

### Run the graphical interface

```powershell
python gui.py
```

Expected result:

```text
A window titled "P7 Handwritten Sudoku Reader and Solver" opens.
```

Explanation: Choose a seed, choose 0-3 misreads, select `ucs`, `greedy`, or
`astar`, and click **Run pipeline**. The window displays the observed puzzle,
the corrected CSP result, the ground truth, a status message, counters, and a
correction/search trace. If the required real NN assets are unavailable, the
pipeline uses its documented simulated-probability fallback.

### Run the text and image demo

```powershell
python run_demo.py
```

Expected output includes:

```text
Observed/input grid
Search method and correction information
Expanded/generated node counters
Solution or a no-solution message
Plot saved to artifacts/demo.png
```

Explanation: The command runs one seeded example using seed 7, two misreads,
and UCS. It prints the pipeline report and saves a three-panel image showing
the NN input, corrected result, and ground truth.

### Run the default experiments

```powershell
python run_experiments.py
```

Expected output has this form:

```text
Experiment rows: <number>
Results saved to artifacts/results.csv
```

Explanation: The default run performs five seeded trials per level. It writes
CSV results and plots such as:

- `artifacts/results.csv`
- `artifacts/search_experiment.png`
- `artifacts/csp_experiment.png`
- `artifacts/correction_experiment.png`

The exact row count and measured values depend on the configured experiments.

### Run the full 100-puzzle recovery requirement

```powershell
python run_experiments.py --puzzles-per-level 100 --output artifacts/full_run
```

Expected output:

```text
Experiment rows: <number>
Results saved to artifacts/full_run/results.csv
```

Explanation: This is the project-specific exact-recovery experiment. It is
more expensive because it runs correction search on 100 puzzles for each
misread level. The output directory keeps the full-run results separate from
the default results.

### Download MNIST

```powershell
python make_data.py --download --path data/mnist
```

Expected output:

```text
MNIST stored in data\mnist
Puzzle:
 ...
Solution:
 ...
```

Explanation: The command downloads the four public MNIST IDX files if they
are missing. The script also prints a generated sample puzzle and solution.
The exact grid values depend on the seed and may vary in formatting.

### Prepare MNIST data

```powershell
python make_data.py --prepare --path data/mnist --output artifacts/prepared_data.npz
```

Expected output:

```text
Prepared data saved to artifacts/prepared_data.npz
Puzzle:
 ...
Solution:
 ...
```

Explanation: The command checks the standard MNIST sizes, creates the
50,000/10,000 source split, removes digit 0, scales pixels to 0-1, and saves
the prepared arrays in a compressed NumPy file.

### Train and evaluate the classifiers

```powershell
python -m nn.train --path data/mnist --limit 10000
```

Expected output has this form:

```text
nn accuracy= <value> calibration_error= <value>
logistic_regression accuracy= <value> calibration_error= <value>
validation accuracy= <value> calibration_error= <value>
metrics saved to artifacts/nn_metrics.json
```

Explanation: The command trains the NumPy MLP and logistic-regression
baseline, evaluates them, saves the metrics JSON, saves the MLP weights, and
writes the 9x9 test confusion matrix to
`artifacts/nn_confusion_matrix.csv`. Exact metric values can change with data,
seed, or training settings.

### Run automated tests

```powershell
python -m pytest -q
```

Expected output has this form:

```text
................                                                         [100%]
16 passed
```

Explanation: Each dot represents a passing test. The tests verify CSP modes,
reproducible puzzle generation, correction, integration, invalid input, and
impossible-puzzle behavior. The exact number may change if tests are added.

## 5. How to explain a successful result

For a successful pipeline run, explain the result in this order:

1. The observed grid is the classifier's first reading and may contain
   conflicts.
2. Conflicting cells are identified using Sudoku row, column, and box rules.
3. Search tries alternative digits, preferring alternatives with low
   probability cost.
4. CSP checks whether each candidate is a valid solvable Sudoku.
5. The first accepted candidate is reported as `solution found`.
6. The correction log identifies each changed cell, its original digit, its
   replacement digit, and its cost.
7. The counters show how much work the selected search and CSP strategy did.

For an unsuccessful result, explain that `no solution` means no candidate
within the configured correction limit passed the CSP solver. It does not
necessarily mean that the original Sudoku has no mathematical solution; the
recognition errors may require more corrections than the search limit.

## 6. Files worth mentioning during a presentation

| File/folder | Purpose |
|---|---|
| `README.md` | Installation and usage overview |
| `prj.txt` | Original project idea and requirements |
| `data/` | MNIST preparation and Sudoku data generation |
| `nn/` | MLP, baseline, training, and metrics |
| `search/` | UCS, Greedy, and A* correction search |
| `csp/` | Backtracking, Forward Checking, and AC-3 |
| `integration/` | End-to-end pipeline |
| `eval/` | Experiments and plots |
| `tests/` | Automated tests |
| `artifacts/` | Generated plots, logs, metrics, and model weights |
| `REPORT.md` | Recorded results, design choices, and limitations |
