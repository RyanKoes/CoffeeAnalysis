from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

try:
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"This script requires 'matplotlib'. Install it into your active environment "
		"(e.g., pip install matplotlib)."
	) from exc

from plot_generator import FIGSIZE_SINGLE_COLUMN, apply_publication_style, savefig_pdf


def _load_voltammetry_txt(path: Path) -> Tuple[List[float], List[float], List[float]]:
	"""Load a 3-column voltammetry text file: time, potential (V), current."""

	times: List[float] = []
	potentials: List[float] = []
	currents: List[float] = []

	with path.open("r", newline="") as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue
			# Expected format (examples in repo):
			#   0.01000, -0.8000, -143.3
			t_str, e_str, i_str = (cell.strip() for cell in row[:3])
			times.append(float(t_str))
			potentials.append(float(e_str))
			currents.append(float(i_str))

	return times, potentials, currents


def main() -> None:
	apply_publication_style(figsize=FIGSIZE_SINGLE_COLUMN)

	repo_root = Path(__file__).resolve().parents[1]
	data_path = repo_root / "voltammetry-files" / "AlabasterColumbiaReg1.txt"

	_, potential_v, current = _load_voltammetry_txt(data_path)

	# Ignore any data in negative voltage (keep potential/current aligned)
	nonneg_pairs = [(v, i) for v, i in zip(potential_v, current) if v >= 0.0]
	potential_v, current = (list(t) for t in zip(*nonneg_pairs))

	fig, ax = plt.subplots()

	# Highlight region: 0.0 to 1.0 V
	ax.axvspan(0.0, 1.0, alpha=0.18, zorder=0)

	ax.plot(potential_v, current, linewidth=1.2, zorder=2)

	ax.set_xlabel("Potential (V)")
	ax.set_ylabel("Current (µA)")
	ax.set_title("Alabaster Columbia (Reg 1)")

	x_min = 0.0
	x_max = max(max(potential_v), 1.0)
	ax.set_xlim(x_min, x_max)

	out_dir = repo_root / "Plots"
	out_dir.mkdir(exist_ok=True)

	out_path = out_dir / "AlabasterColumbiaReg1.pdf"
	savefig_pdf(str(out_path))

	plt.show()


if __name__ == "__main__":
	main()

