"""Matplotlib style shared by the thesis figures (CONDOR palette)."""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

GAMMA = "#6E1423"        # dark red -- signal, matches thesis citation colour
PROTON = "#1A1A1A"       # ink -- background
ACCENT = "#F81B84"       # magenta, for highlights
MUTED = "#8A8A8A"
GRID = "#D9D9D9"

PRIMARY_COLORS = {"gamma": GAMMA, "proton": PROTON}

# Sequential: white -> dark red. For densities, footprints, hit maps.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "condor_seq", ["#FFFFFF", "#F2D5D9", "#D08A94", "#A33B4C", GAMMA, "#3D0B14"]
)



def apply_style(context: str = "notebook") -> None:
    """Install CONDOR defaults into matplotlib.

    `context` scales font sizes: 'notebook' (default), 'paper' (compact,
    for LaTeX figures) or 'talk' (large, for slides).
    """
    scale = {"paper": 0.85, "notebook": 1.0, "talk": 1.35}[context]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Palatino Linotype", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10 * scale,
        "axes.titlesize": 11 * scale,
        "axes.labelsize": 10 * scale,
        "xtick.labelsize": 9 * scale,
        "ytick.labelsize": 9 * scale,
        "legend.fontsize": 9 * scale,
        "figure.titlesize": 12 * scale,

        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "axes.edgecolor": "#4A4A4A",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "image.cmap": "condor_seq",
    })
    for name, cmap in (("condor_seq", CMAP_SEQ),):
        try:
            mpl.colormaps.register(cmap, name=name)
        except ValueError:
            pass  # already registered
