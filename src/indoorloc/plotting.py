"""Shared matplotlib style so every figure in the report looks the same."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .data import ROOT

FIG_DIR = ROOT / "results" / "figures"

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Categorical slots, fixed order. Buildings use the first three.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BUILDING_COLORS = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2]}

# Single-hue blue ramp for magnitude (heatmaps, error maps) and ordinal floors.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
FLOOR_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
FLOOR_CMAP = ListedColormap(FLOOR_COLORS)


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2, "legend.frameon": False,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
