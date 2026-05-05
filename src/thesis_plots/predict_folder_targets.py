"""Predict coffee targets (TDS / Caffeine / CGA) for new CV files.

This repo's window-search artifacts (data/BEST_*.pkl) store the *best window +
architecture*, but they do not store an inference-ready scaler. To predict
new files reliably, this script can:

  1) Train a single model on *all* available labeled CoffeeHub data for a chosen
     target, using the stored best window+architecture.
  2) Use that trained model to predict a folder of new CV files.

Expected input file format matches util.read_cv_data_bins():
  CSV-like text with 3 columns per row: t, v, i (no header), comma-separated.

Examples
--------

Train (if needed) + predict a folder:

  python ThesisPlotGeneration/predict_folder_targets.py \
    --input-dir ./my_new_scans --target caffeine --output predictions.csv

Predict using an already-trained inference model:

  python ThesisPlotGeneration/predict_folder_targets.py \
    --input-dir ./my_new_scans --target cga \
    --model ThesisPlotGeneration/data/inference_models/HPLC_CGA__...pth
"""

from __future__ import annotations

# --- repo-root bootstrap (added by reorg) ---
import sys as _sys
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
# --- end bootstrap ---


import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------

# This file lives in ThesisPlotGeneration/. The repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = Path(__file__).resolve().parent

# Make relative-path defaults (./data, ./voltammetry-files, etc.) behave as in
# the original training/search scripts, no matter where the user runs this from.
os.chdir(REPO_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import pandas as pd
import torch

from nn_0_synthetic_data_gen import build_model_data
from nn_1_train_model import CoffeeNetBase, train_coffeenet
from nn_model_window_search import get_network_architectures
from util import read_cv_data_bins


CanonicalTarget = Literal["HPLC_Caff", "HPLC_CGA", "TDS"]


TARGET_ALIASES: dict[str, CanonicalTarget] = {
    "caff": "HPLC_Caff",
    "caffeine": "HPLC_Caff",
    "hplc_caff": "HPLC_Caff",
    "cga": "HPLC_CGA",
    "hplc_cga": "HPLC_CGA",
    "tds": "TDS",
}


def _canonical_target(raw: str) -> CanonicalTarget:
    key = raw.strip().lower()
    if key in TARGET_ALIASES:
        return TARGET_ALIASES[key]
    raise ValueError(f"Unknown target '{raw}'. Use one of: {sorted(set(TARGET_ALIASES))}")


def _default_best_params_path(target: CanonicalTarget) -> Path:
    # User requested the script live next to ThesisPlotGeneration/data.
    return THESIS_DIR / "data" / f"BEST_{target}_fixed.pkl"


def _default_model_out_path(target: CanonicalTarget, window: tuple[float, float], architecture: str) -> Path:
    vmin, vmax = window
    safe_arch = architecture.replace("/", "-").replace(" ", "_")
    out_dir = THESIS_DIR / "data" / "inference_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{target}__{safe_arch}__V{vmin:.2f}-{vmax:.2f}.pth"


def _load_best_window_and_arch(target: CanonicalTarget, best_params_path: Path) -> tuple[tuple[float, float], str, float | None]:
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best-params file not found: {best_params_path}")

    data = torch.load(best_params_path, map_location=torch.device("cpu"), weights_only=False)
    metadata = data.get("metadata", {})
    window = metadata.get("window")
    architecture = metadata.get("architecture")
    r2 = metadata.get("r2")

    if window is None or architecture is None:
        raise ValueError(f"Missing metadata in {best_params_path}")

    vmin, vmax = float(window[0]), float(window[1])
    return (vmin, vmax), str(architecture), (float(r2) if r2 is not None else None)


def _voltage_indices(n_points: int, window: tuple[float, float]) -> np.ndarray:
    vmin, vmax = window
    full_voltage = np.linspace(0.0, 2.0, n_points)
    mask = (full_voltage >= vmin) & (full_voltage <= vmax)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"Window {window} contains no points for n_points={n_points}")
    return idx


def _get_arch_builder(architecture_name: str):
    architectures = get_network_architectures()
    arch_def = next((a for a in architectures if a.get("network_name") == architecture_name), None)
    if arch_def is None:
        known = sorted({a.get("network_name") for a in architectures})
        raise ValueError(
            f"Architecture '{architecture_name}' not found in get_network_architectures().\n"
            f"Known architectures include: {known[:12]}{' ...' if len(known) > 12 else ''}"
        )
    return arch_def["network"]


@dataclass(frozen=True)
class InferenceBundle:
    target: CanonicalTarget
    window: tuple[float, float]
    architecture: str
    n_points: int
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    model: CoffeeNetBase


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe


def _destandardize_y(y_s: np.ndarray, mean: float, std: float) -> np.ndarray:
    return y_s * std + mean


def train_inference_model(
    *,
    target: CanonicalTarget,
    window: tuple[float, float],
    architecture: str,
    num_epochs: int,
    out_path: Path,
    device: str,
) -> Path:
    df_all = build_model_data(
        NORMALIZE=False,
        REDOX=False,
        USE_BINS=False,
        test_train_split=False,
    )

    df_all = df_all.dropna(subset=[target])
    if len(df_all) == 0:
        raise ValueError(f"No rows available with non-null target '{target}'.")

    n_points = len(df_all["cv_raw"].iloc[0])
    idx = _voltage_indices(n_points, window)

    X = np.array([np.asarray(cv)[idx] for cv in df_all["cv_raw"]], dtype=float)
    y = df_all[target].values.reshape(-1, 1).astype(float)

    # Match training scripts: StandardScaler (mean/std) for X and y.
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if y_std == 0:
        raise ValueError(f"Target '{target}' has zero variance; cannot train.")

    X_s = _standardize(X, x_mean, x_std)
    y_s = (y - y_mean) / y_std

    arch_builder = _get_arch_builder(architecture)
    model = CoffeeNetBase()
    model.network = arch_builder(X_s.shape[1])
    model.to(torch.device(device))

    model = train_coffeenet(model, X_s, y_s, None, None, num_epochs=num_epochs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target": target,
            "window": (float(window[0]), float(window[1])),
            "architecture": architecture,
            "n_points": int(n_points),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "num_epochs": int(num_epochs),
        },
        out_path,
    )

    return out_path


def load_inference_model(model_path: Path, device: str) -> InferenceBundle:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    target = _canonical_target(str(data["target"]))
    window = (float(data["window"][0]), float(data["window"][1]))
    architecture = str(data["architecture"])
    n_points = int(data["n_points"])

    x_mean = np.asarray(data["x_mean"], dtype=float)
    x_std = np.asarray(data["x_std"], dtype=float)
    y_mean = float(data["y_mean"])
    y_std = float(data["y_std"])

    idx = _voltage_indices(n_points, window)
    input_size = int(idx.size)

    arch_builder = _get_arch_builder(architecture)
    model = CoffeeNetBase()
    model.network = arch_builder(input_size)
    model.load_state_dict(data["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    return InferenceBundle(
        target=target,
        window=window,
        architecture=architecture,
        n_points=n_points,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        model=model,
    )


def _iter_input_files(input_dir: Path, glob: str) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir must be a directory: {input_dir}")

    files = sorted([p for p in input_dir.glob(glob) if p.is_file()])
    # If user used the default, filter to typical extensions.
    if glob == "*":
        files = [p for p in files if p.suffix.lower() in {".txt", ".csv", ".dat"}]
    return files


def _load_raw_trace(file_path: Path) -> np.ndarray:
    # read_cv_data_bins expects (filename, datadir) rather than full path.
    _, raw = read_cv_data_bins(
        file_path.name,
        normalize=False,
        redox=False,
        datadir=file_path.parent,
        num_bins=64,
    )
    return np.asarray(raw, dtype=float)


def _resample_1d(y: np.ndarray, n: int) -> np.ndarray:
    if y.size == n:
        return y
    if y.size < 2:
        raise ValueError("Cannot resample a trace with <2 points")
    x_old = np.linspace(0.0, 1.0, y.size)
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, y)


def predict_folder(
    *,
    bundle: InferenceBundle,
    input_dir: Path,
    glob: str,
    resample: bool,
    device: str,
) -> pd.DataFrame:
    idx = _voltage_indices(bundle.n_points, bundle.window)

    rows: list[dict[str, object]] = []
    files = list(_iter_input_files(input_dir, glob=glob))
    if not files:
        raise FileNotFoundError(f"No files matched in {input_dir} with glob '{glob}'")

    for fp in files:
        try:
            raw = _load_raw_trace(fp)
            if raw.size != bundle.n_points:
                if resample:
                    raw = _resample_1d(raw, bundle.n_points)
                else:
                    raise ValueError(
                        f"Raw trace length {raw.size} != expected {bundle.n_points}. "
                        f"Re-run with --resample to force interpolation."
                    )

            x = raw[idx]
            x_s = _standardize(x, bundle.x_mean, bundle.x_std)

            with torch.no_grad():
                pred_s = (
                    bundle.model(torch.tensor(x_s, dtype=torch.float32, device=torch.device(device)).unsqueeze(0))
                    .detach()
                    .cpu()
                    .numpy()
                )

            pred = float(_destandardize_y(pred_s, bundle.y_mean, bundle.y_std).ravel()[0])

            rows.append(
                {
                    "file": fp.name,
                    "path": str(fp),
                    "target": bundle.target,
                    "prediction": pred,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "file": fp.name,
                    "path": str(fp),
                    "target": bundle.target,
                    "prediction": np.nan,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing CV files to predict")
    p.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target to predict: tds|caffeine|cga (aliases accepted)",
    )
    p.add_argument(
        "--best-params",
        type=Path,
        default=None,
        help="Path to BEST_<target>_fixed.pkl (default: ThesisPlotGeneration/data)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to an inference model .pth. If omitted, uses ThesisPlotGeneration/data/inference_models/.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Epochs when training a new inference model (default: 2000)",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Do not auto-train if the model file is missing (error instead)",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*",
        help="Glob pattern within --input-dir (default: '*'; filtered to .txt/.csv/.dat)",
    )
    p.add_argument(
        "--resample",
        action="store_true",
        help="If input raw trace length differs, linearly resample to expected length",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path (default: predictions.csv)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device to use (default: cuda if available else cpu)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    target = _canonical_target(args.target)

    best_params_path = args.best_params or _default_best_params_path(target)
    window, architecture, best_r2 = _load_best_window_and_arch(target, best_params_path)

    model_path = args.model or _default_model_out_path(target, window, architecture)

    if model_path.exists():
        print(f"Loading inference model: {model_path}")
    else:
        if args.no_train:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Re-run without --no-train to train automatically."
            )
        print(
            "Inference model missing; training a new one from CoffeeHub data.\n"
            f"  Target: {target}\n"
            f"  Best window: {window}\n"
            f"  Best architecture: {architecture}\n"
            + (f"  Best LOCO R² (from search): {best_r2:.3f}\n" if best_r2 is not None else "")
            + f"  Epochs: {args.epochs}\n"
            f"  Output: {model_path}"
        )
        model_path = train_inference_model(
            target=target,
            window=window,
            architecture=architecture,
            num_epochs=args.epochs,
            out_path=model_path,
            device=args.device,
        )
        print(f"Saved inference model: {model_path}")

    bundle = load_inference_model(model_path, device=args.device)

    print(
        f"Predicting folder '{args.input_dir}' for {bundle.target} "
        f"(window {bundle.window[0]:.2f}-{bundle.window[1]:.2f} V, arch {bundle.architecture})"
    )

    df_pred = predict_folder(
        bundle=bundle,
        input_dir=args.input_dir,
        glob=args.glob,
        resample=args.resample,
        device=args.device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(args.output, index=False)
    print(f"Wrote predictions: {args.output} ({len(df_pred)} rows)")


if __name__ == "__main__":
    main()
