'''Simulated annealing in continuous space or on square lattice sites.'''

from __future__ import annotations

import numpy as np

from ._funcn import Neighborhoods


def anneal(size, labels, target, sigma, statistic, permutations, grid_size, steps, rng):
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
    if not fit.any():
        return points, None, None
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
    for step in range(steps):
        if best == 0:
            break
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
        candidate = loss()
        delta = candidate - current
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current = candidate
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
                best_points = points.copy()
        else:
            for change in reversed(changes):
                model.undo(change)
            points[index] = original
            if swap is not None:
                points[swap] = destination

    return best_points, initial, best
