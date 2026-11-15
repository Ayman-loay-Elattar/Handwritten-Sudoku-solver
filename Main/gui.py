"""Interactive Tkinter visualizer for the complete Sudoku pipeline."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from integration.pipeline import render_report, run_pipeline


class SudokuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("P7 Handwritten Sudoku Reader and Solver")
        root.geometry("1120x820")
        self.seed_var = tk.StringVar(value="7")
        self.misreads_var = tk.StringVar(value="2")
        self.method_var = tk.StringVar(value="ucs")
        self.bt_var = tk.BooleanVar(value=True)
        self.fc_var = tk.BooleanVar(value=True)
        self.ac3_var = tk.BooleanVar(value=True)
        self.nn_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Set options and click Run pipeline.")
        self.last_run = None

        controls = ttk.LabelFrame(root, text="Pipeline controls", padding=8)
        controls.pack(fill="x", padx=10, pady=8)
        self._entry(controls, "Seed", self.seed_var, 6)
        self._entry(controls, "Misreads (0-3)", self.misreads_var, 6)
        ttk.Label(controls, text="Search").pack(side="left", padx=(12, 4))
        ttk.Combobox(controls, textvariable=self.method_var, values=("ucs", "greedy", "astar"),
                     state="readonly", width=8).pack(side="left")
        for label, variable in (("BT", self.bt_var), ("FC", self.fc_var),
                                ("AC-3", self.ac3_var), ("NN on", self.nn_var)):
            ttk.Checkbutton(controls, text=label, variable=variable).pack(side="left", padx=5)
        ttk.Button(controls, text="Run pipeline", command=self.run).pack(side="left", padx=8)
        ttk.Button(controls, text="Step", command=self.step).pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=5)

        boards = ttk.Frame(root, padding=8)
        boards.pack(fill="x")
        self.canvases = []
        for title in ("Observed NN input", "Corrected CSP result", "Ground truth"):
            panel = ttk.Frame(boards)
            panel.pack(side="left", expand=True)
            ttk.Label(panel, text=title).pack()
            canvas = tk.Canvas(panel, width=320, height=320, background="white")
            canvas.pack(padx=8)
            self.canvases.append(canvas)

        ttk.Label(root, textvariable=self.status_var, padding=8).pack(fill="x")
        counters = ttk.LabelFrame(root, text="Live counters", padding=6)
        counters.pack(fill="x", padx=10)
        self.counter_var = tk.StringVar(value="No run yet.")
        ttk.Label(counters, textvariable=self.counter_var).pack(anchor="w")
        log_frame = ttk.LabelFrame(root, text="Correction/search trace", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.log = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    @staticmethod
    def _entry(parent: ttk.Widget, label: str, variable: tk.StringVar, width: int) -> None:
        ttk.Label(parent, text=label).pack(side="left", padx=(4, 3))
        ttk.Entry(parent, textvariable=variable, width=width).pack(side="left")

    def _read_options(self) -> tuple[int, int, str]:
        try:
            seed = int(self.seed_var.get())
            misreads = int(self.misreads_var.get())
        except ValueError as error:
            raise ValueError("Seed must be an integer and misreads must be an integer from 0 to 3") from error
        method = self.method_var.get().strip().lower()
        if not 0 <= misreads <= 3:
            raise ValueError("Misreads must be between 0 and 3")
        if method not in {"ucs", "greedy", "astar"}:
            raise ValueError("Search must be ucs, greedy, or astar")
        return seed, misreads, method

    def draw(self, canvas: tk.Canvas, grid: np.ndarray, highlights: dict[tuple[int, int], str] | None = None) -> None:
        canvas.delete("all")
        size = 320 / 9
        highlights = highlights or {}
        for row in range(10):
            width = 3 if row % 3 == 0 else 1
            canvas.create_line(0, row * size, 320, row * size, width=width)
            canvas.create_line(row * size, 0, row * size, 320, width=width)
        for row in range(9):
            for col in range(9):
                colour = highlights.get((row, col))
                if colour:
                    canvas.create_rectangle(col * size + 2, row * size + 2,
                                            (col + 1) * size - 2, (row + 1) * size - 2,
                                            fill=colour, outline="")
                value = int(grid[row, col])
                if value:
                    canvas.create_text((col + 0.5) * size, (row + 0.5) * size,
                                       text=str(value), font=("Segoe UI", 18))

    def run(self) -> None:
        try:
            seed, misreads, method = self._read_options()
            run = run_pipeline(seed, misreads, method, self.nn_var.get(),
                               self.fc_var.get(), self.ac3_var.get(), self.bt_var.get())
        except (ValueError, RuntimeError) as error:
            self.status_var.set(f"Error: {error}")
            return
        self.last_run = run
        result = run["result"]
        corrected = result.grid if result.grid is not None else run["observed"]
        conflicts = {tuple(cell): "tomato" for cell in run["conflicts"]}
        changed = {tuple(item["cell"]): "lightgreen" for item in result.corrections}
        self.draw(self.canvases[0], run["observed"], conflicts)
        self.draw(self.canvases[1], corrected, {**conflicts, **changed})
        self.draw(self.canvases[2], run["solution"])
        stats = run["csp_stats"]
        self.status_var.set(f"{result.message} | NN={'on' if run['nn_enabled'] else 'off'}")
        self.counter_var.set(
            f"assignments={stats.assignments} | backtracks={stats.backtracks} | "
            f"arc revisions={stats.revisions} | nodes expanded={result.nodes_expanded} | "
            f"nodes generated={result.nodes_generated}"
        )
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", render_report(run) + "\n\nSearch trace:\n")
        for index, item in enumerate(result.trace):
            self.log.insert("end", f"step {index}: cost={item['cost']:.4f}, corrections={item['corrections']}\n")
        self.log.configure(state="disabled")

    def step(self) -> None:
        if self.last_run is None:
            self.status_var.set("Run the pipeline before stepping through the trace.")
            return
        trace = self.last_run["result"].trace
        shown = self.last_run.get("trace_step", 0)
        self.last_run["trace_step"] = min(shown + 1, max(0, len(trace) - 1))
        item = trace[self.last_run["trace_step"]]
        self.status_var.set(f"Trace step {self.last_run['trace_step']}: cost={item['cost']:.4f}")

    def clear(self) -> None:
        self.last_run = None
        for canvas in self.canvases:
            canvas.delete("all")
        self.status_var.set("Set options and click Run pipeline.")
        self.counter_var.set("No run yet.")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    SudokuApp(root)
    root.mainloop()
