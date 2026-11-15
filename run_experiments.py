import argparse

from eval.experiments import run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles-per-level", type=int, default=5,
                        help="number of search puzzles per misread level; use 100 for the full requirement")
    parser.add_argument("--output", default="artifacts",
                        help="directory for results.csv, logs, and plots")
    args = parser.parse_args()
    result = run_experiments(puzzles_per_level=args.puzzles_per_level, output=args.output)
    print(f"Experiment rows: {len(result['rows'])}")
    print(f"Results saved to {result['path']}")
