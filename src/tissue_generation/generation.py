"""Public interfaces for summarizing and generating annotated tissue samples."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ._anneal import AnnealingDiagnostics, anneal
from .statistics import _size, summarize
from .summary import TissueSummary, nonnegative_int, positive_float


def _print_diagnostics(report):
    reason = {
        "no_cells": "no cells requested",
        "no_targets": "no defined FunCN targets for the placed cells",
        "zero_steps": "zero proposals requested",
        "zero_energy": "zero energy reached",
        "step_budget": "proposal budget reached",
    }[report["stop_reason"]]
    print(
        f"Generation finished in {report['total_seconds']:.3f} s "
        f"(annealing setup and search {report['annealing_seconds']:.3f} s) · {reason}"
    )
    if report["final_energy"] is not None:
        print(
            f"Energy {report['initial_energy']:.6g} → {report['final_energy']:.6g} "
            f"(best at proposal {report['best_step']:,}, last {report['last_energy']:.6g})"
        )
    attempted = report["steps_attempted"]
    evaluated = report["steps_evaluated"]
    accepted = report["steps_accepted"]
    acceptance = f" ({accepted / evaluated:.1%})" if evaluated else ""
    print(
        f"Proposals {attempted:,}/{report['steps_requested']:,} · "
        f"accepted {accepted:,}/{evaluated:,} evaluated{acceptance} · "
        f"skipped {attempted - evaluated:,} · target pairs {report['target_pairs']:,}"
    )


def generate_from_summary(
    summary: TissueSummary | str | Path,
    *,
    domain_size: tuple[float, float],
    n_cells: int | None = None,
    grid_size: float | None = None,
    steps: int = 2000,
    seed: int | None = None,
    progress: bool = False,
    verbose: bool | None = None,
) -> pd.DataFrame:
    """Generate x, y, cell_type rows from a summary object or CSV/YAML path.

    Set progress=True to show the annealing proposal count and energy.
    The completion summary follows progress by default. Set verbose explicitly
    to control it independently. Diagnostics are returned in attrs["annealing"].
    """
    started = perf_counter()
    if isinstance(summary, (str, Path)):
        summary = TissueSummary.load(summary)
    if not isinstance(summary, TissueSummary):
        raise TypeError("summary must be a TissueSummary or CSV/YAML path")

    size = _size(domain_size, "domain_size")
    steps = nonnegative_int(steps, "steps")
    if grid_size is not None:
        grid_size = positive_float(grid_size, "grid_size")
    if n_cells is None:
        with np.errstate(over="ignore", under="ignore"):
            expected_count = float(size.prod() * sum(summary.densities.values()))
        if not np.isfinite(expected_count):
            raise ValueError("domain area times density must give a finite cell count")
        n_cells = round(expected_count)
    n_cells = nonnegative_int(n_cells, "n_cells")
    types = list(summary.densities)
    annealing_seconds = 0.0
    if n_cells:
        ideal = n_cells * np.array(list(summary.abundances.values()))
        counts = np.floor(ideal).astype(np.int64)
        remainders = np.argsort(counts - ideal, kind="stable")
        counts[remainders[:n_cells - counts.sum()]] += 1
        labels = np.repeat(np.arange(len(types)), counts)
        target = summary.funcn.loc[types, types].to_numpy()
        annealing_started = perf_counter()
        points, diagnostics = anneal(
            size, labels, target, summary.sigma,
            summary.statistic, summary.permutations, grid_size, steps,
            np.random.default_rng(seed), show_progress=progress,
        )
        annealing_seconds = perf_counter() - annealing_started
    else:
        labels = np.empty(0, dtype=int)
        points = np.empty((0, 2))
        diagnostics = AnnealingDiagnostics(steps, stop_reason="no_cells")

    result = pd.DataFrame(points, columns=["x", "y"])
    result["cell_type"] = pd.Categorical.from_codes(labels, categories=types)
    report = asdict(diagnostics)
    report.update(annealing_seconds=annealing_seconds, total_seconds=perf_counter() - started)
    result.attrs.update(
        domain_size=tuple(size.tolist()), grid_size=grid_size, statistic=summary.statistic,
        initial_loss=diagnostics.initial_energy, final_loss=diagnostics.final_energy,
        annealing=report,
    )
    if verbose or (verbose is None and progress):
        _print_diagnostics(report)
    return result


def generate_from_data(
    data,
    *,
    domain_size: tuple[float, float],
    sigma: float,
    statistic: str = "median",
    n_cells: int | None = None,
    grid_size: float | None = None,
    steps: int = 2000,
    seed: int | None = None,
    progress: bool = False,
    verbose: bool | None = None,
    **summary_options,
) -> pd.DataFrame:
    """Summarize a spatial sample and generate a tissue using those targets.

    Set progress=True to show progress and diagnostics, or verbose=True for
    diagnostics alone. Additional keyword arguments are forwarded to summarize().
    """
    started = perf_counter()
    summary = summarize(data, sigma=sigma, statistic=statistic, seed=seed, **summary_options)
    result = generate_from_summary(
        summary, domain_size=domain_size, n_cells=n_cells, grid_size=grid_size,
        steps=steps, seed=seed, progress=progress, verbose=False,
    )
    report = result.attrs["annealing"]
    report["total_seconds"] = perf_counter() - started
    if verbose or (verbose is None and progress):
        _print_diagnostics(report)
    return result


def to_anndata(cells: pd.DataFrame):
    """Convert generated cells to AnnData with obsm['spatial'] and obs['cell_type']."""
    try:
        from anndata import AnnData
    except ImportError as exc:
        raise ImportError("AnnData export requires tissue-generation[spatial]") from exc

    obs = cells[["cell_type"]].reset_index(drop=True).copy()
    obs["cell_type"] = obs["cell_type"].astype("category")
    obs.index = obs.index.astype(str)
    return AnnData(
        X=np.empty((len(cells), 0), dtype=np.float32),
        obs=obs,
        obsm={"spatial": cells[["x", "y"]].to_numpy(dtype=float)},
    )
