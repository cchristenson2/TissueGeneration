'''Gaussian FunCN with composition-preserving label permutations.'''

from __future__ import annotations

import numpy as np


def gaussian_weights(delta: np.ndarray, sigma: float) -> np.ndarray:
    # Truncation makes zero support well defined and avoids relying on tiny tails.
    with np.errstate(over="ignore"):
        squared = np.square(delta / sigma).sum(axis=-1)
    return np.exp(-0.5 * squared) * (squared <= 16)


class Neighborhoods:
    def __init__(self, points, labels, n_types, sigma, statistic, permutations, rng):
        self.sigma = sigma
        self.statistic = statistic
        self.counts = np.bincount(labels, minlength=n_types)
        self.labels = np.stack(
            [labels] + [rng.permutation(labels) for _ in range(
                permutations if statistic == "median" else 0
            )]
        )
        self.draws = np.arange(len(self.labels))
        self.group_indices = [
            np.argsort(self.labels != group, axis=1, kind="stable")[:, :count]
            for group, count in enumerate(self.counts)
        ]
        self.bins = self.labels + n_types * self.draws[:, None]
        self.fields = np.empty((len(self.labels), len(points), n_types))
        for start in range(0, len(points), 128):
            stop = min(start + 128, len(points))
            weights = gaussian_weights(points[start:stop, None] - points, sigma)
            weights[np.arange(stop - start), np.arange(start, stop)] = 0
            for draw, groups in enumerate(self.labels):
                for group in range(n_types):
                    self.fields[draw, start:stop, group] = weights[:, groups == group].sum(axis=1)

    def ratios(self) -> np.ndarray:
        n_types = len(self.counts)
        medians_or_means = np.full((len(self.labels), n_types, n_types), np.nan)
        for target, indices in enumerate(self.group_indices):
            if indices.shape[1]:
                influence = self.fields[self.draws[:, None], indices, :]
                reduce = np.median if self.statistic == "median" else np.mean
                medians_or_means[:, target] = reduce(influence, axis=1)
        n_cells = len(self.labels[0])
        if self.statistic == "median":
            baseline = np.median(medians_or_means[1:], axis=0)
        elif n_cells > 1:
            # Each other cell has this probability of carrying the source label.
            mean_weight = self.fields[0].sum() / (n_cells * (n_cells - 1))
            baseline = mean_weight * (self.counts[None, :] - np.eye(n_types))
        else:
            baseline = np.zeros((n_types, n_types))
        possible = (self.counts[:, None] > 0) & (
            self.counts[None, :] - np.eye(n_types) > 0
        )
        observed = np.maximum(medians_or_means[0], 0)
        observed[observed < 1e-10] = 0
        return np.divide(
            observed, baseline, out=np.full((n_types, n_types), np.nan),
            where=possible & (baseline > 1e-10),
        )

    def move(self, index, destination, points):
        '''Update affected influences and return the values needed to undo the move.'''
        rows = np.arange(len(points))
        sources = self.labels[:, index]
        column = (self.draws[:, None], rows, sources[:, None])
        old_column = self.fields[column]
        old_row = self.fields[:, index].copy()
        weights = gaussian_weights(destination - points, self.sigma)
        weights[index] = 0
        old_weights = gaussian_weights(points[index] - points, self.sigma)
        old_weights[index] = 0
        self.fields[column] = old_column + (weights - old_weights)
        totals = np.bincount(
            self.bins.ravel(), weights=np.tile(weights, len(self.draws)),
            minlength=len(self.draws) * len(self.counts),
        )
        self.fields[:, index] = totals.reshape(old_row.shape)
        return column, old_column, index, old_row

    def undo(self, change):
        column, old_column, index, old_row = change
        self.fields[column] = old_column
        self.fields[:, index] = old_row
