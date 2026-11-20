"""Public interfaces for summarizing and generating annotated tissue samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._anneal import anneal
from .statistics import _size, summarize
from .summary import TissueSummary, nonnegative_int, positive_float


def generate_from_summary(
    summary: TissueSummary | str | Path,
    *,
    domain_size: tuple[float, float],
    n_cells: int | None = None,
    grid_size: float | None = None,
    steps: int = 2000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate x, y, cell_type rows from a summary object or CSV/YAML path."""
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
    if n_cells:
        ideal = n_cells * np.array(list(summary.abundances.values()))
        counts = np.floor(ideal).astype(np.int64)
        remainders = np.argsort(counts - ideal, kind="stable")
        counts[remainders[:n_cells - counts.sum()]] += 1
        labels = np.repeat(np.arange(len(types)), counts)
        points, initial, final = anneal(
            size, labels, summary.funcn.loc[types, types].to_numpy(), summary.sigma,
            summary.statistic, summary.permutations, grid_size, steps,
            np.random.default_rng(seed),
        )
    else:
        labels = np.empty(0, dtype=int)
        points = np.empty((0, 2))
        initial = final = None

    result = pd.DataFrame(points, columns=["x", "y"])
    result["cell_type"] = pd.Categorical.from_codes(labels, categories=types)
    result.attrs.update(
        domain_size=tuple(size.tolist()), grid_size=grid_size, statistic=summary.statistic,
        initial_loss=initial, final_loss=final,
    )
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
    **summary_options,
) -> pd.DataFrame:
    """Summarize a spatial sample and generate a tissue using those targets.

    Additional keyword arguments are forwarded to summarize().
    """
    summary = summarize(data, sigma=sigma, statistic=statistic, seed=seed, **summary_options)
    return generate_from_summary(
        summary, domain_size=domain_size, n_cells=n_cells, grid_size=grid_size,
        steps=steps, seed=seed,
    )


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
