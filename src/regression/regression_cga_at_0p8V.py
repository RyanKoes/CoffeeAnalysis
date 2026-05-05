from __future__ import annotations

# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---


import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from util import read_coffehub


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find the project root (contains 'voltammetry-files')."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "voltammetry-files").exists():
            return candidate
    raise FileNotFoundError("Could not find 'voltammetry-files' by walking upward from CWD")


def _load_voltammetry_txt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 3-column voltammetry text file: time, potential (V), current (µA)."""

    times: List[float] = []
    potentials: List[float] = []
    currents: List[float] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            t_str, e_str, i_str = (cell.strip() for cell in row[:3])
            times.append(float(t_str))
            potentials.append(float(e_str))
            currents.append(float(i_str))

    return np.asarray(times, dtype=float), np.asarray(potentials, dtype=float), np.asarray(currents, dtype=float)


def _moving_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, preserving length."""
    if window <= 1:
        return values.copy()

    n = int(values.size)
    if n == 0:
        return values.copy()

    left = window // 2
    right = window - left

    out = np.empty_like(values, dtype=float)
    for i in range(n):
        start = max(0, i - left)
        end = min(n, i + right)
        out[i] = float(np.mean(values[start:end]))
    return out


def _oxidation_segment(potential_v: np.ndarray, current_ua: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return only the oxidation sweep (initial up-scan): start -> first max potential."""

    if potential_v.size == 0:
        return potential_v, current_ua

    i_max = int(np.argmax(potential_v))
    if i_max == 0 and potential_v.size > 1:
        # Likely reversed ordering; flip and re-identify.
        potential_v = potential_v[::-1]
        current_ua = current_ua[::-1]
        i_max = int(np.argmax(potential_v))

    return potential_v[: i_max + 1], current_ua[: i_max + 1]


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


@dataclass(frozen=True)
class SamplePoint:
    coffee_name: str
    volt_file: str
    hplc_cga: float
    v_target: float
    v_actual: float
    i_at_v: float


def build_dataset(
    repo_root: Path,
    *,
    v_target: float = 0.8,
    smooth_window: int = 20,
) -> List[SamplePoint]:
    """Join CoffeeHub truth to voltammetry files and extract I(0.8V) per curve."""

    # NOTE: util.read_coffehub() has a quirk: when require_columns is provided it
    # always also requires all cupping-attribute columns, which can unnecessarily
    # force a Google Sheets refresh. For this regression helper, we only need a
    # small set of columns, so we load with cache enabled and validate columns
    # ourselves.
    df = read_coffehub(use_cache=True)

    required_cols = {"Name", "HPLC_CGA", "cv_data1", "cv_data2", "cv_data3"}
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        raise KeyError(
            "CoffeeHub data is missing required columns: "
            + ", ".join(missing)
            + ". If you recently changed the sheet schema, delete data/raw_data_cache.pkl to refresh."
        )

    volt_dirs = [repo_root / "voltammetry-files", repo_root / "extra-voltammetry-files"]

    points: List[SamplePoint] = []

    for _, row in df.iterrows():
        name = str(row["Name"])
        hplc_cga = float(row["HPLC_CGA"])

        for col in ("cv_data1", "cv_data2", "cv_data3"):
            volt_file = row.get(col)
            if pd.isna(volt_file):
                continue

            volt_file = str(volt_file).strip()
            if not volt_file:
                continue

            path = None
            for d in volt_dirs:
                candidate = d / volt_file
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                # Skip missing voltammetry files; the truth row may still be useful elsewhere.
                continue

            _, potential_v, current_ua = _load_voltammetry_txt(path)

            # Keep only non-negative potential region to match notebook conventions.
            mask_nonneg = potential_v >= 0.0
            potential_v = potential_v[mask_nonneg]
            current_ua = current_ua[mask_nonneg]

            # Oxidation only.
            potential_ox, current_ox = _oxidation_segment(potential_v, current_ua)

            # Smooth current and take the point closest to v_target.
            current_ox_sm = _moving_window_mean(current_ox, smooth_window)

            idx = _nearest_index(potential_ox, v_target)
            v_actual = float(potential_ox[idx])
            i_at_v = float(current_ox_sm[idx])

            points.append(
                SamplePoint(
                    coffee_name=name,
                    volt_file=volt_file,
                    hplc_cga=hplc_cga,
                    v_target=v_target,
                    v_actual=v_actual,
                    i_at_v=i_at_v,
                )
            )

    return points


def fit_and_plot(points: List[SamplePoint], out_path: Path, *, show: bool = False) -> None:
    if not points:
        raise ValueError("No usable sample points were found (check voltammetry filenames + CoffeeHub cache)")

    x = np.asarray([p.i_at_v for p in points], dtype=float).reshape(-1, 1)
    y = np.asarray([p.hplc_cga for p in points], dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    r2 = float(r2_score(y, y_pred))
    mae = float(mean_absolute_error(y, y_pred))

    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], y, s=12, alpha=0.75, zorder=2)

    # Regression line across observed x-range
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_line = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line[:, 0], y_line, color="0.2", linewidth=2.0, zorder=3)

    ax.set_xlabel("Current (uA) at 0.8 V")
    ax.set_ylabel("CGAs (ppm)")
    ax.set_title("CGA Prediction")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")

    print(f"Saved: {out_path}")
    print(f"N={len(points)} curves")
    print(f"Fit: HPLC_CGA = {model.coef_[0]:.6g} * I0.8 + {model.intercept_:.6g}")
    print(f"R²={r2:.6g}, MAE={mae:.6g} ppm")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit and plot CGA regression using current at 0.8 V.")
    p.add_argument("--v", type=float, default=0.8, help="Target voltage (V) for feature extraction (default: 0.8)")
    p.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Centered moving average window for current (default: 20)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF path (default: Plots/CGA_regression_I0p8V.pdf)",
    )
    p.add_argument("--show", action="store_true", help="Show the plot window")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = _find_repo_root()

    out_path = args.out
    if out_path is None:
        out_path = repo_root / "Plots" / "CGA_regression_I0p8V.pdf"

    points = build_dataset(repo_root, v_target=float(args.v), smooth_window=int(args.smooth_window))
    fit_and_plot(points, out_path, show=bool(args.show))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
