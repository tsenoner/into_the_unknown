from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(
    ax: plt.Axes,
    results_df: "pd.DataFrame",
    title: str,
    x_col: str = "x_col",
    y_col: str = "r_square",
    yerr_col: Optional[str] = "std_err",
    group_col: str = "plm",
    fontsize: int = 16,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """Plot bar chart results on the given axes."""
    if color_map is None:
        color_map = plt.cm.get_cmap("tab10")

    bar_width = 0.8 / len(results_df[group_col].unique())
    groups = sorted(
        results_df[group_col].unique(),
        key=lambda x: (
            list(color_map.keys()).index(x) if x in color_map else float("inf")
        ),
    )
    x = np.arange(len(results_df[x_col].unique()))

    for i, group in enumerate(groups):
        group_data = results_df[results_df[group_col] == group]
        yerr = group_data[yerr_col] if yerr_col in group_data.columns else None
        color = (
            color_map[group]
            if isinstance(color_map, dict)
            else color_map(i / len(groups))
        )

        ax.bar(
            x + i * bar_width,
            group_data[y_col],
            bar_width,
            yerr=yerr,
            label=group,
            color=color,
        )

    ax.set_xlabel(x_col, fontsize=fontsize)
    ax.set_ylabel(y_col, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 4)
    ax.set_xticks(x + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(results_df[x_col].unique(), fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def create_plots(
    data: Union["pd.DataFrame", List["pd.DataFrame"]],
    titles: Union[str, List[str]],
    output_file: str,
    x_col: str = "x_col",
    y_col: str = "r_square",
    yerr_col: Optional[str] = "std_err",
    group_col: str = "plm",
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (20, 10),
    fontsize: int = 16,
) -> None:
    """Create one or two plots based on the provided data and save the figure."""
    if isinstance(data, list) and len(data) == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        data = [data] if not isinstance(data, list) else data

    if isinstance(titles, str):
        titles = [titles]

    handles, labels = [], []
    for ax, df, title in zip(axes, data, titles):
        plot_results(
            ax,
            df,
            title,
            x_col,
            y_col,
            yerr_col,
            group_col,
            fontsize,
            color_map,
        )
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=fontsize + 4,
        title="pLMs",
        title_fontsize=fontsize,
        ncol=4,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


# Example color map (you can modify or extend this as needed)
COLOR_MAP = {
    "prott5": "#377eb8",
    "prottucker": "#373bb7",
    "prostt5": "#1217b5",
    "esm1b": "#5fd35b",
    "clean": "#4daf4a",
    "esm2_8m": "#fdae61",
    "esm2_35m": "#ff7f00",
    "esm2_150m": "#f46d43",
    "esm2_650m": "#d73027",
    "esm2_3b": "#a50026",
    "ankh_base": "#ffd700",
    "ankh_large": "#a88c01",
}
