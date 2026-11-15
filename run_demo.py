"""Run the integrated demo and save a three-panel visualization."""

from pathlib import Path

import numpy as np

from integration.pipeline import render_report, run_pipeline


def main() -> None:
    run = run_pipeline(seed=7, misreads=2, method="ucs")
    print(render_report(run))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; textual demo completed")
        return
    result = run["result"]
    corrected = result.grid if result.grid is not None else run["observed"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, [run["observed"], corrected, run["solution"]],
                                  ["NN input", "Corrected/CSP solution", "Ground truth"]):
        axis.imshow(image, cmap="Blues", vmin=0, vmax=9)
        axis.set_title(title)
        axis.set_xticks(range(9))
        axis.set_yticks(range(9))
        axis.grid(True, color="white")
        for row in range(9):
            for col in range(9):
                axis.text(col, row, str(image[row, col]), ha="center", va="center")
    figure.tight_layout()
    Path("artifacts").mkdir(exist_ok=True)
    figure.savefig("artifacts/demo.png", dpi=150)
    print("Plot saved to artifacts/demo.png")


if __name__ == "__main__":
    main()
