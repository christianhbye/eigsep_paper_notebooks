"""Shared figure geometry for the EIGSEP instrument paper.

Every paper figure is built through this module, so that the saved PDF box is
identically the figure size and the render scale in the manuscript is 1.000.

The defect this closes: ``bbox_inches="tight"`` shrinks the canvas to the ink
and then adds ``pad_inches``, so the saved box is not a predictable function of
the figure size -- in either direction. ``save`` below takes no keyword
arguments, which is what makes a tight bounding box unreachable.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

TEX_PT_PER_IN = 72.27  # a TeX pt; a PDF MediaBox pt is 1/72 in
COLUMN_PT = 244.0      # \columnwidth  = (\textwidth - \columnsep) / 2
TEXT_PT = 508.0        # \textwidth

COL_IN = COLUMN_PT / TEX_PT_PER_IN    # 3.376228
FULL_IN = TEXT_PT / TEX_PT_PER_IN     # 7.029196

RC = {
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "savefig.dpi": 600,
    "savefig.bbox": None,      # never "tight"
    "savefig.pad_inches": 0.0,
}


def apply_rc():
    """Apply the paper's font and output scheme to the global rcParams."""
    mpl.rcParams.update(RC)


def figsize(cols=1, aspect=0.75, height_in=None):
    """Figure size in inches for a ``cols``-column figure.

    ``cols=1`` is a ``figure`` float, ``cols=2`` a ``figure*``. Height is
    ``height_in`` when given, else ``width * aspect``.
    """
    width = COL_IN if cols == 1 else FULL_IN
    height = float(height_in) if height_in is not None else width * aspect
    return (width, height)


def figure(cols=1, aspect=0.75, height_in=None, **subplots_kw):
    """``plt.subplots`` at the paper's geometry, constrained layout by default."""
    apply_rc()
    subplots_kw.setdefault("layout", "constrained")
    return plt.subplots(figsize=figsize(cols, aspect, height_in), **subplots_kw)


def save(fig, path):
    """Write ``fig`` to ``path``.

    Takes no keyword arguments on purpose: there is no way to ask for a tight
    bounding box through this function, so the saved box stays identical to the
    figure size.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
