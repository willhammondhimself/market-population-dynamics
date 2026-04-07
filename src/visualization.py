"""Visualization functions for ternary phase diagrams, price paths, and distributions."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import List


def ternary_coords(p1, p2, p3):
    """Convert simplex coordinates to 2D Cartesian for plotting.

    Vertices: p1=1 at (0,0), p2=1 at (1,0), p3=1 at (0.5, sqrt(3)/2).
    """
    x = p2 + p3 * 0.5
    y = p3 * np.sqrt(3) / 2
    return x, y


def plot_ternary_heatmap(
    results: List[dict],
    metric_key: str,
    title: str,
    cmap: str = "RdYlGn_r",
    vmin: float = None,
    vmax: float = None,
    ax=None,
):
    """Plot a metric on the ternary simplex.

    Args:
        results: list of dicts with p1, p2, p3 and metric fields.
        metric_key: which metric to plot.
        title: plot title and colorbar label.
        cmap: matplotlib colormap.
        vmin, vmax: colorbar range.
        ax: matplotlib axes (creates new figure if None).

    Returns:
        (fig, ax) tuple.
    """
    p1_arr = np.array([r["p1"] for r in results])
    p2_arr = np.array([r["p2"] for r in results])
    p3_arr = np.array([r["p3"] for r in results])
    values = np.array([r[metric_key] for r in results])

    x, y = ternary_coords(p1_arr, p2_arr, p3_arr)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    triang = mtri.Triangulation(x, y)

    levels = 20
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 20)

    tcf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend="both")
    ax.tricontour(triang, values, levels=levels, colors="k", linewidths=0.3, alpha=0.3)

    # Simplex edges
    triangle = plt.Polygon(
        [ternary_coords(1, 0, 0), ternary_coords(0, 1, 0), ternary_coords(0, 0, 1)],
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(triangle)

    # Vertex labels
    ax.text(*ternary_coords(1, 0, 0), "\nType I\n(Value)", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(*ternary_coords(0, 1, 0), "\nType II\n(Emotion)", ha="center", va="top", fontsize=12, fontweight="bold")
    tx, ty = ternary_coords(0, 0, 1)
    ax.text(tx, ty + 0.05, "Type III\n(Noise)", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.colorbar(tcf, ax=ax, label=title, shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_aspect("equal")
    ax.axis("off")

    if own_fig:
        plt.tight_layout()

    return fig, ax


def plot_price_paths(
    results_list: list,
    title: str = "Price Paths",
    colors=None,
    ax=None,
):
    """Plot multiple simulation price paths.

    Args:
        results_list: list of simulate_market() result dicts.
        title: plot title.
        colors: list of colors (one per path).
        ax: matplotlib axes.

    Returns:
        (fig, ax) tuple.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    for i, result in enumerate(results_list):
        color = colors[i] if colors else None
        ax.plot(result["time"], result["price"], color=color, alpha=0.5, linewidth=0.8)
        if i == 0:
            ax.plot(result["time"], result["fair_value"], "k--", alpha=0.5, linewidth=1, label="Fair value $V^*$")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.legend(loc="upper right")

    if own_fig:
        plt.tight_layout()

    return fig, ax


def plot_demand_decomposition(result: dict, p: tuple, ax=None):
    """Plot 4-panel demand decomposition for a single simulation path.

    Args:
        result: simulate_market() output dict.
        p: (p1, p2, p3) population fractions used.
        ax: array of 4 axes (creates new figure if None).

    Returns:
        (fig, axes) tuple.
    """
    p1, p2, p3 = p

    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    else:
        axes = ax
        fig = axes[0].get_figure()

    # Price vs fair value
    axes[0].plot(result["time"], result["price"], "b-", linewidth=1, label="Market price")
    axes[0].plot(result["time"], result["fair_value"], "k--", linewidth=1, label="Fair value $V^*$")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title(f"Demand Decomposition — $\\mathbf{{p}}$ = ({p1}, {p2}, {p3})", fontsize=13, fontweight="bold")

    # Weighted demands
    axes[1].plot(result["time"], p1 * result["demand_value"], "b-", alpha=0.7, linewidth=0.8, label="Value demand")
    axes[1].plot(result["time"], p2 * result["demand_emotion"], "r-", alpha=0.7, linewidth=0.8, label="Emotion demand")
    axes[1].plot(result["time"], p3 * result["demand_noise"], "gray", alpha=0.5, linewidth=0.8, label="Noise demand")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("Weighted demand")
    axes[1].legend()

    # Sentiment
    axes[2].plot(result["time"], result["sentiment"], "r-", linewidth=0.8)
    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].set_ylabel("Sentiment $S(t)$")

    # Mispricing
    axes[3].fill_between(
        result["time"], result["mispricing"], 0,
        where=result["mispricing"] >= 0, alpha=0.4, color="green", label="Overvalued",
    )
    axes[3].fill_between(
        result["time"], result["mispricing"], 0,
        where=result["mispricing"] < 0, alpha=0.4, color="red", label="Undervalued",
    )
    axes[3].axhline(0, color="k", linewidth=0.5)
    axes[3].set_ylabel("$\\log(P/V^*)$")
    axes[3].set_xlabel("Time (years)")
    axes[3].legend()

    if own_fig:
        plt.tight_layout()

    return fig, axes


def plot_mispricing_distributions(
    mixes: list,
    params=None,
    n_paths: int = 100,
):
    """Plot mispricing histograms for multiple population mixes.

    Args:
        mixes: list of ((p1,p2,p3), label) tuples.
        params: ModelParams instance.
        n_paths: number of MC paths per mix.

    Returns:
        (fig, axes) tuple.
    """
    from src.simulation import simulate_market
    from src.agents import ModelParams as MP

    if params is None:
        params = MP()

    nrows = (len(mixes) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 5 * nrows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, (pop, title) in zip(axes_flat, mixes):
        all_mispricings = []
        for i in range(n_paths):
            result = simulate_market(pop, params, seed=i)
            half = len(result["mispricing"]) // 2
            all_mispricings.extend(result["mispricing"][half:])

        all_mispricings = np.array(all_mispricings)
        ax.hist(all_mispricings, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Fair value")
        ax.set_title(f"{title}\n$\\mathbf{{p}}$ = {pop}", fontsize=11, fontweight="bold")
        ax.set_xlabel("$\\log(P/V^*)$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # Hide unused axes
    for ax in list(axes_flat)[len(mixes):]:
        ax.set_visible(False)

    fig.suptitle(
        "Mispricing Distributions by Population Mix",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig, axes
