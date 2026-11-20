# Tissue Generation

Generate virtual 2D tissue samples with one row per cell and columns `x`, `y`, and
`cell_type`. The package uses measured cell abundances and Gaussian neighborhood
statistics to place cells in a rectangular domain with simulated annealing.

## Install

```bash
pip install .
```

## Generate from spatial data

Supply a pandas DataFrame with one row per cell, a spatial CSV, or an AnnData
object. For AnnData, coordinates come from `obsm["spatial"]` and cell labels
from `obs["cell_type"]`. Use `x_col`, `y_col`, `type_col`, and `spatial_key`
to override those names as applicable.

```python
from tissue_generation import generate_from_data

cells = generate_from_data(
    "reference_cells.csv",
    domain_size=(400, 300),
    reference_size=(400, 300),
    n_cells=200,
    sigma=25,
    statistic="median",
    cell_type_map={"CD4 T": "T cell", "CD8 T": "T cell"},
    seed=42,
)
cells.to_csv("virtual_cells.csv", index=False)
```

Use matching units for coordinates, reference and output dimensions, `sigma`,
and `grid_size`. When supplied, `reference_size=(width, height)` gives the area
used for density, including any margins. Otherwise the reference area is
inferred from the coordinate spans. When `n_cells` is omitted, the package
rounds the reference density times output area. When `grid_size` is omitted,
coordinates are continuous. Set it to a positive spacing to place at most one
cell at the center of each square that fully fits in the domain.

## Generate from a summary

```python
from tissue_generation import generate_from_summary, summarize

summary = summarize("reference_cells.csv", sigma=25, statistic="mean", seed=42)
summary.save("reference_summary.yaml")
summary.save("reference_summary.csv")

cells = generate_from_summary(
    "reference_summary.yaml", domain_size=(400, 300), n_cells=200, grid_size=10, seed=42
)
```

You can also pass a `TissueSummary` object directly. The YAML format has
`densities` (cells per unit area by type), `funcn` (a mapping of target type to
source type to ratio), and `sigma`. Include `statistic` and `permutations` to
override the defaults of `median` and 19. CSV has one row for every ordered
target/source pair with columns `target,source,density,funcn,sigma,statistic,permutations`.
Repeat the target's density for each source and use the same settings on every
row. Use YAML `null` or an empty CSV field for an undefined ratio. Relative
abundances are each type's density divided by the total density. Apply type
mapping before measuring targets, while the individual cells are available.

## FunCN and annealing

Each target cell receives the summed Gaussian influence of cells of each
source type within four kernel standard deviations, excluding its own
contribution. The median ratio divides the median influence over the target
cells by the median of the same statistic over label permutations. The mean
ratio uses the mean influence and the exact expected mean under random
labeling. Both preserve type counts under permutation. Ratios are undefined
when a pair cannot contribute or its permutation baseline is zero.

Annealing keeps the requested cell count and apportions it across types by
largest remainder. It fits the defined ratios for represented cell types and
returns the best placement found within the `steps` proposal budget. Use
`steps=0` for a random initial placement. The returned DataFrame exposes
`initial_loss` and `final_loss` in `attrs`. Both are `None` when no ratios
can be fitted. Some targets are hard to reproduce on a different domain or at
different density. This implementation initializes all cell pairs in blocks
and is intended for small to moderate samples.

## Measure a tissue directly

`cell_composition()` returns counts, relative abundances, and area densities by
cell type. `funcn_matrix()` returns the directed FunCN ratios for a chosen
kernel width and statistic. Both accept the same spatial inputs and cell-type
mapping options as `summarize()`.

```python
from tissue_generation import cell_composition, funcn_matrix

composition = cell_composition(cells, reference_size=(400, 300))
organization = funcn_matrix(cells, sigma=25, statistic="mean")
```

The [observed vs. virtual tissue notebook](notebooks/observed_vs_virtual.ipynb)
loads one sample from `data/backup_output.rds`, generates a tissue with the
same cell count and domain, and compares composition and FunCN. Install the
notebook dependencies in a Python 3.11+ kernel environment with
`pip install -e '.[notebook]'`. Reading this RDS archive uses about 6 GB of RAM.

## Visualize with Squidpy

Install the optional AnnData dependency with `pip install '.[spatial]'` and
install Squidpy in the plotting environment.

```python
import squidpy as sq
from tissue_generation import to_anndata

sq.pl.spatial_scatter(to_anndata(cells), color="cell_type", shape=None, img=False, size=10)
```

[Squidpy spatial scatter documentation](https://squidpy.readthedocs.io/en/stable/api/squidpy.pl.spatial_scatter.html)
