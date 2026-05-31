"""
plot_utils.py
-------------
Matplotlib figure builders for the early warning dashboard.
All functions accept pre-processed grids and return a Figure.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure


# ── Shared style ──────────────────────────────────────────────────────────────

BG    = "#0e1117"
TEXT  = "#e8eaf0"
SPINE = "#2d3140"

ALERT_COLORS = {
    "None":   "#2a4858",
    "Orange": "#ff8c00",
    "Red":    "#e63946",
}

ALERT_ORDER = ["None", "Orange", "Red"]


def _apply_dark_style(fig: Figure, axes) -> None:
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)


# ── Probability pcolormesh ────────────────────────────────────────────────────

def plot_probability(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    prob_grid: np.ndarray,
    date: str = "",
    figsize: tuple[int, int] = (8, 6),
    boundaries_gdf=None,
    prob_label: str = "P(event)",
    thresholds: dict | None = None,
) -> Figure:
    """
    Pcolormesh of continuous pred_target values.

    Parameters
    ----------
    prob_label  : colorbar label, e.g. "P(coldspell)" or "P(extreme rainfall)"
    thresholds  : {"Red": float, "Orange": float} — draws contour lines at each level
    """
    if thresholds is None:
        thresholds = {"Red": 0.66, "Orange": 0.33}

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _apply_dark_style(fig, ax)

    cmap = plt.cm.RdYlBu_r
    norm = mcolors.Normalize(vmin=0, vmax=1)

    pcm = ax.pcolormesh(lon_grid, lat_grid, prob_grid,
                        cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label(prob_label, color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    # Contour lines at alert thresholds
    for level in ["Orange", "Red"]:
        thresh = thresholds.get(level)
        if thresh is None:
            continue
        cs = ax.contour(lon_grid, lat_grid, prob_grid,
                        levels=[thresh],
                        colors=[ALERT_COLORS[level]],
                        linewidths=1.0, linestyles="--")
        ax.clabel(cs, fmt={thresh: f"{level} ({thresh:.0%})"},
                  colors=[ALERT_COLORS[level]], fontsize=7)

    if boundaries_gdf is not None:
        boundaries_gdf.boundary.plot(ax=ax, color=TEXT, linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = prob_label
    if date:
        title += f"  ·  {date}"
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_aspect("equal")
    return fig


# ── Alert-level pcolormesh ────────────────────────────────────────────────────

def plot_alert_level(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    alert_grid: np.ndarray,
    date: str = "",
    figsize: tuple[int, int] = (8, 6),
    boundaries_gdf=None,
    peril_label: str = "Event",
) -> Figure:
    """
    Pcolormesh of categorical alert levels (None / Orange / Red).

    Parameters
    ----------
    peril_label : used in the plot title, e.g. "Coldspell" or "Rainfall"
    """
    level_to_int = {lvl: i for i, lvl in enumerate(ALERT_ORDER)}
    int_grid = np.vectorize(level_to_int.get)(alert_grid).astype(float)

    cmap = mcolors.ListedColormap([ALERT_COLORS[l] for l in ALERT_ORDER])
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _apply_dark_style(fig, ax)

    ax.pcolormesh(lon_grid, lat_grid, int_grid,
                  cmap=cmap, norm=norm, shading="auto")

    patches = [
        mpatches.Patch(color=ALERT_COLORS[lvl], label=lvl)
        for lvl in ALERT_ORDER
    ]
    ax.legend(
        handles=patches, loc="lower right", framealpha=0.25,
        facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, fontsize=9,
    )

    if boundaries_gdf is not None:
        boundaries_gdf.boundary.plot(ax=ax, color=TEXT, linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = f"{peril_label} alert level"
    if date:
        title += f"  ·  {date}"
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_aspect("equal")
    return fig