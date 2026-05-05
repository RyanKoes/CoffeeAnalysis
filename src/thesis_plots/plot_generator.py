"""ThesisPlotGeneration plotting defaults.

Import and call :func:`apply_publication_style` at the top of any script in this
folder to get consistent, publication-friendly Matplotlib output.

Key goals (matching typical journal submission requirements):
- Embed TrueType fonts in PDFs (editable/searchable): ``pdf.fonttype = 42``
- Prevent labels from getting clipped: ``savefig.bbox = 'tight'``
- Use serif typography (journal-like)
- Provide sensible default ``figsize`` presets for single/double column figures
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

try:
	import matplotlib as mpl  # type: ignore[reportMissingImports]
	import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
	raise ImportError(
		"ThesisPlotGeneration.plot_generator requires 'matplotlib'. "
		"Install it into your active environment (e.g., pip install matplotlib)."
	) from exc

# --- Figure size presets (inches) ---
# These are conservative defaults that generally print well.
# You can override per-figure by passing figsize=(w, h) to plt.subplots(...)
# or by calling apply_publication_style(figsize=...).
FIGSIZE_SINGLE_COLUMN: Tuple[float, float] = (3.35, 2.40)
FIGSIZE_DOUBLE_COLUMN: Tuple[float, float] = (6.90, 3.10)


_BASE_RCPARAMS: Mapping[str, Any] = {
	# Typography: journals usually prefer serif fonts.
	"font.family": "serif",
	# Ordered fallbacks; Matplotlib will use the first available on the system.
	"font.serif": [
		"Times New Roman",
		"Times",
		"Nimbus Roman",
		"STIX Two Text",
		"STIXGeneral",
		"DejaVu Serif",
	],
	"font.size": 11,
	"axes.labelsize": 12,
	"axes.titlesize": 12,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"legend.fontsize": 10,
	"figure.titlesize": 14,

	# Output: make PDF text editable/searchable (TrueType fonts).
	"pdf.fonttype": 42,
	"ps.fonttype": 42,

	# LaTeX: keep off by default (set True only if you have a full LaTeX install).
	"text.usetex": False,

	# Saving: high-res raster fallback; default format + tight bounds.
	"savefig.dpi": 300,
	"savefig.format": "pdf",
	"savefig.bbox": "tight",
}


def apply_publication_style(
	*,
	figsize: Tuple[float, float] = FIGSIZE_SINGLE_COLUMN,
	use_tex: bool = False,
	extra_rcparams: Mapping[str, Any] | None = None,
) -> None:
	"""Apply consistent Matplotlib settings for publication-quality figures.

	Args:
		figsize: Default figure size in inches. This sets the global
			``figure.figsize`` rcParam (you can still override per-figure).
		use_tex: Set True to render text via LaTeX (requires a working LaTeX
			installation); defaults to False.
		extra_rcparams: Optional overrides/additions applied last.
	"""

	rc = dict(_BASE_RCPARAMS)
	rc["figure.figsize"] = figsize
	rc["text.usetex"] = use_tex
	if extra_rcparams:
		rc.update(dict(extra_rcparams))

	mpl.rcParams.update(rc)


def savefig_pdf(path: str, *, dpi: int = 300, **kwargs: Any) -> None:
	"""Save the current figure as a PDF with publication defaults.

	This is a convenience wrapper around ``plt.savefig`` that enforces the two
	most common "oops" settings: tight bounding box and PDF output.
	"""

	kwargs.setdefault("format", "pdf")
	kwargs.setdefault("bbox_inches", "tight")
	kwargs.setdefault("dpi", dpi)
	plt.savefig(path, **kwargs)

