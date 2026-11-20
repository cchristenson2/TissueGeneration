"""Tissue composition and directed FunCN targets with CSV and YAML I/O."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def positive_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def validate_metric(sigma, statistic, permutations):
    sigma = positive_float(sigma, "sigma")
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'")
    permutations = nonnegative_int(permutations, "permutations")
    if statistic == "median" and permutations == 0:
        raise ValueError("median FunCN requires at least one permutation")
    return sigma, permutations


@dataclass
class TissueSummary:
    """Area densities per type and a FunCN matrix with target rows and source columns."""

    densities: Mapping[str, float]
    funcn: pd.DataFrame
    sigma: float
    statistic: str = "median"
    permutations: int = 19

    def __post_init__(self):
        self.sigma, self.permutations = validate_metric(
            self.sigma, self.statistic, self.permutations
        )
        if not isinstance(self.densities, Mapping) or not self.densities:
            raise ValueError("densities must map at least one cell type to a density")
        if any(not isinstance(label, str) or not label.strip() for label in self.densities):
            raise ValueError("cell types must be nonempty strings")
        self.densities = {
            label: positive_float(value, f"density for {label!r}")
            for label, value in self.densities.items()
        }
        positive_float(sum(self.densities.values()), "total density")
        labels = list(self.densities)
        if (
            not isinstance(self.funcn, pd.DataFrame)
            or not self.funcn.index.is_unique
            or not self.funcn.columns.is_unique
            or set(self.funcn.index) != set(labels)
            or set(self.funcn.columns) != set(labels)
        ):
            raise ValueError("FunCN rows and columns must each match the density labels")
        self.funcn = self.funcn.loc[labels, labels].astype(float)
        values = self.funcn.to_numpy()
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError("FunCN values must be nonnegative or missing")

    @property
    def abundances(self) -> dict[str, float]:
        """Return relative cell-type abundances inferred from the area densities."""
        total = sum(self.densities.values())
        return {label: density / total for label, density in self.densities.items()}

    def save(self, path: str | Path) -> None:
        """Write this summary as YAML (.yaml or .yml) or CSV."""
        path = Path(path)
        metadata = {
            "sigma": self.sigma, "statistic": self.statistic, "permutations": self.permutations
        }
        if path.suffix.lower() in (".yaml", ".yml"):
            data = {
                **metadata,
                "densities": self.densities,
                "funcn": {
                    target: {
                        source: None if pd.isna(value) else float(value)
                        for source, value in row.items()
                    }
                    for target, row in self.funcn.iterrows()
                },
            }
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        elif path.suffix.lower() == ".csv":
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["target", "source", "density", "funcn", *metadata])
                for target, row in self.funcn.iterrows():
                    for source, value in row.items():
                        writer.writerow([
                            target, source, self.densities[target],
                            "" if pd.isna(value) else value, *metadata.values(),
                        ])
        else:
            raise ValueError("summary path must end in .csv, .yaml, or .yml")

    @classmethod
    def load(cls, path: str | Path) -> TissueSummary:
        """Read a summary written in the documented CSV or YAML format."""
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not {"densities", "funcn", "sigma"} <= data.keys():
                raise ValueError("YAML summary requires densities, funcn, and sigma")
            matrix = data["funcn"]
            if not isinstance(matrix, dict) or any(
                not isinstance(row, dict) or row.keys() != matrix.keys()
                for row in matrix.values()
            ):
                raise ValueError("YAML FunCN must list every ordered pair")
            return cls(
                data["densities"], pd.DataFrame.from_dict(matrix, orient="index"),
                data["sigma"], data.get("statistic", "median"), data.get("permutations", 19),
            )
        if path.suffix.lower() != ".csv":
            raise ValueError("summary path must end in .csv, .yaml, or .yml")

        columns = {"target", "source", "density", "funcn", "sigma", "statistic", "permutations"}
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if not columns <= set(reader.fieldnames or ()):
                raise ValueError(f"CSV summary requires columns {', '.join(sorted(columns))}")
            densities, matrix, metadata = {}, {}, set()
            for row in reader:
                if any(row[key] is None for key in columns):
                    raise ValueError("CSV summary row is missing columns")
                target, source_type = row["target"], row["source"]
                density = positive_float(row["density"], f"density for {target!r}")
                if densities.setdefault(target, density) != density:
                    raise ValueError(f"inconsistent density for {target!r}")
                pairs = matrix.setdefault(target, {})
                if source_type in pairs:
                    raise ValueError(f"duplicate FunCN pair {(target, source_type)!r}")
                pairs[source_type] = float(row["funcn"]) if row["funcn"].strip() else np.nan
                metadata.add((float(row["sigma"]), row["statistic"], int(row["permutations"])))
        if len(metadata) != 1 or any(row.keys() != matrix.keys() for row in matrix.values()):
            raise ValueError("CSV summary needs consistent settings and every ordered pair")
        sigma, statistic, permutations = metadata.pop()
        return cls(
            densities, pd.DataFrame.from_dict(matrix, orient="index"),
            sigma, statistic, permutations,
        )
