'''Simulated annealing in continuous space or on square lattice sites.'''

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from ._funcn import Neighborhoods


@dataclass
class AnnealingDiagnostics:
    steps_requested: int
    stop_reason: str = "no_targets"
    target_pairs: int = 0
    steps_attempted: int = 0
    steps_evaluated: int = 0
    steps_accepted: int = 0
    initial_energy: float | None = None
    final_energy: float | None = None
    last_energy: float | None = None
    best_step: int | None = None
    energy_history: list[tuple[int, float, float]] = field(default_factory=list)


def anneal(
    size, labels, target, sigma, statistic, permutations, grid_size, steps, rng,
    *, show_progress=False,
):
    diagnostics = AnnealingDiagnostics(steps)
    n_cells = len(labels)
    if grid_size is None:
        points = rng.random((n_cells, 2)) * size
    else:
        shape = tuple(int(np.floor(np.nextafter(length / grid_size, np.inf))) for length in size)
        capacity = shape[0] * shape[1]
        if capacity < n_cells:
            raise ValueError(f"lattice has {capacity} sites for {n_cells} cells")
        sites = rng.choice(capacity, n_cells, replace=False)
        occupied = {site: index for index, site in enumerate(sites)}

        def center(site):
            return (np.stack(np.divmod(site, shape[1]), axis=-1) + 0.5) * grid_size

        points = center(sites)

    counts = np.bincount(labels, minlength=len(target))
    possible = (counts[:, None] > 0) & (counts[None, :] - np.eye(len(target)) > 0)
    fit = np.isfinite(target) & possible
    diagnostics.target_pairs = int(fit.sum())
    if not diagnostics.target_pairs:
        return points, diagnostics
    active = counts > 0
    # Only allocate fields for the types represented in the output.
    active_codes = np.cumsum(active) - 1
    model = Neighborhoods(
        points, active_codes[labels], active.sum(), sigma, statistic, permutations, rng
    )
    fit = fit[np.ix_(active, active)]
    expected = target[np.ix_(active, active)][fit]
    scale = np.maximum(1, expected)

    def loss():
        observed = model.ratios()[fit]
        error = np.abs(observed - expected) / scale
        # Give missing ratios a larger penalty than any defined ratio.
        squared_error = np.square(error)
        return float(np.mean(np.where(np.isfinite(error), squared_error / (1 + squared_error), 1)))

    initial = current = best = loss()
    best_points = points.copy()
    diagnostics.initial_energy = initial
    diagnostics.best_step = 0
    diagnostics.energy_history.append((0, current, best))
    sample_interval = max(1, (steps + 199) // 200)
    with tqdm(
        total=steps, desc="Annealing", unit="proposal",
        disable=not show_progress or steps == 0 or best == 0,
    ) as proposals:
        if show_progress and steps and best > 0:
            proposals.set_postfix_str(f"current={current:.4g}, best={best:.4g}", refresh=False)
        for step in range(steps):
            if best == 0:
                break
            if step and step % sample_interval == 0:
                diagnostics.energy_history.append((step, current, best))
                if show_progress:
                    proposals.set_postfix_str(
                        f"current={current:.4g}, best={best:.4g}", refresh=False
                    )
            if show_progress and step:
                proposals.update()
            diagnostics.steps_attempted = step + 1
            progress = step / max(1, steps - 1)
            temperature = max(initial, 0.01) * 0.6 * 0.001 ** progress
            index = int(rng.integers(n_cells))
            swap = None
            if grid_size is None:
                if rng.random() < 0.25:
                    destination = rng.random(2) * size
                else:
                    radius = min(size.max(), sigma) * (1.5 - progress)
                    destination = points[index] + rng.normal(0, radius, 2)
                    if np.any(destination < 0) or np.any(destination >= size):
                        continue
            else:
                site = int(rng.integers(capacity))
                swap = occupied.get(site)
                if swap is not None and labels[swap] == labels[index]:
                    continue
                destination = center(site)

            original = points[index].copy()
            changes = [model.move(index, destination, points)]
            points[index] = destination
            if swap is not None:
                changes.append(model.move(swap, original, points))
                points[swap] = original
            diagnostics.steps_evaluated += 1
            candidate = loss()
            delta = candidate - current
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current = candidate
                diagnostics.steps_accepted += 1
                if grid_size is not None:
                    original_site = sites[index]
                    if swap is None:
                        del occupied[original_site]
                    else:
                        sites[swap] = original_site
                        occupied[original_site] = swap
                    sites[index] = site
                    occupied[site] = index
                if current < best:
                    best = current
                    diagnostics.best_step = step + 1
                    best_points = points.copy()
            else:
                for change in reversed(changes):
                    model.undo(change)
                points[index] = original
                if swap is not None:
                    points[swap] = destination

        if show_progress and diagnostics.steps_attempted:
            proposals.set_postfix_str(f"current={current:.4g}, best={best:.4g}", refresh=False)
            proposals.update()

    if diagnostics.steps_attempted:
        diagnostics.energy_history.append((diagnostics.steps_attempted, current, best))
    diagnostics.last_energy = current
    diagnostics.final_energy = best
    if steps == 0:
        diagnostics.stop_reason = "zero_steps"
    elif best == 0:
        diagnostics.stop_reason = "zero_energy"
    else:
        diagnostics.stop_reason = "step_budget"
    return best_points, diagnostics
