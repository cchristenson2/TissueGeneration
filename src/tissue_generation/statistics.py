"""Public interfaces for measuring annotated spatial samples."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ._funcn import Neighborhoods
from .summary import TissueSummary, positive_float, validate_metric


def _size(value, name):
    size = np.asarray(value, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError(f"{name} must be a pair of positive finite lengths")
    return size


def _read_cells(data, x_col, y_col, type_col, spatial_key, cell_type_map):
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data, dtype={type_col: str}, keep_default_na=False)
    if isinstance(data, pd.DataFrame):
        coordinates = data[[x_col, y_col]]
        annotations = data[type_col]
    elif hasattr(data, "obs") and hasattr(data, "obsm"):
        coordinates = data.obsm[spatial_key]
        annotations = data.obs[type_col]
    else:
        raise TypeError("data must be a DataFrame, spatial CSV path, or AnnData-like object")

    points = np.asarray(coordinates, dtype=float)
    if points.shape != (len(annotations), 2) or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("input must contain finite x and y coordinates for each cell")
    if annotations.isna().any():
        raise ValueError("every cell must have a cell type")
    annotations = annotations.astype(str)
    if cell_type_map is not None:
        annotations = annotations.map(lambda label: cell_type_map.get(label, label))
    if any(not isinstance(label, str) or not label.strip() for label in annotations):
        raise ValueError("every cell must have a nonempty cell type string")

    with np.errstate(over="ignore"):
        points = points - points.min(axis=0)
    if not np.isfinite(points).all():
        raise ValueError("coordinate spans must be finite")
    labels, codes = np.unique(annotations.to_numpy(dtype=str), return_inverse=True)
    return points, labels.tolist(), codes


def _composition(points, types, codes, reference_size):
    span = points.max(axis=0)
    size = _size(span if reference_size is None else reference_size, "reference_size")
    if not np.allclose(np.minimum(span, size), span, rtol=1e-12, atol=0):
        raise ValueError("reference_size must cover the observed coordinate spans")
    with np.errstate(over="ignore", under="ignore"):
        area = positive_float(size.prod(), "reference area")
    counts = np.bincount(codes)
    return pd.DataFrame(
        {"count": counts, "abundance": counts / len(codes), "density": counts / area},
        index=pd.Index(types, name="cell_type"),
    )


def _funcn(points, types, codes, sigma, statistic, permutations, seed):
    model = Neighborhoods(
        points, codes, len(types), sigma, statistic, permutations, np.random.default_rng(seed)
    )
    return pd.DataFrame(model.ratios(), index=types, columns=types)


def cell_composition(
    data,
    *,
    reference_size: tuple[float, float] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    type_col: str = "cell_type",
    spatial_key: str = "spatial",
    cell_type_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return count, relative abundance, and area density for each observed type.

    Accepts a DataFrame, spatial CSV, or AnnData-like object. Area comes from
    reference_size (width, height) or the observed coordinate spans.
    """
    points, types, codes = _read_cells(data, x_col, y_col, type_col, spatial_key, cell_type_map)
    return _composition(points, types, codes, reference_size)


def funcn_matrix(
    data,
    *,
    sigma: float,
    statistic: str = "median",
    permutations: int = 19,
    x_col: str = "x",
    y_col: str = "y",
    type_col: str = "cell_type",
    spatial_key: str = "spatial",
    cell_type_map: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Measure directed FunCN ratios for a DataFrame, spatial CSV, or AnnData.

    Rows are target types and columns are source types. Undefined ratios are
    NaN. Sigma uses the input coordinate units.
    """
    sigma, permutations = validate_metric(sigma, statistic, permutations)
    points, types, codes = _read_cells(data, x_col, y_col, type_col, spatial_key, cell_type_map)
    return _funcn(points, types, codes, sigma, statistic, permutations, seed)


def summarize(
    data,
    *,
    sigma: float,
    statistic: str = "median",
    permutations: int = 19,
    reference_size: tuple[float, float] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    type_col: str = "cell_type",
    spatial_key: str = "spatial",
    cell_type_map: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> TissueSummary:
    """Measure cell densities and FunCN from a DataFrame, spatial CSV, or AnnData."""
    sigma, permutations = validate_metric(sigma, statistic, permutations)
    points, types, codes = _read_cells(data, x_col, y_col, type_col, spatial_key, cell_type_map)
    composition = _composition(points, types, codes, reference_size)
    ratios = _funcn(points, types, codes, sigma, statistic, permutations, seed)
    return TissueSummary(composition["density"].to_dict(), ratios, sigma, statistic, permutations)
