from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
BAR_WIDTH = 0.8
GROUP_SPACING = 0.5
FEATURE_SPACING = 2

# PLM groups
PLM_GROUPS = [
    ["prott5", "prottucker", "prostt5"],
    ["esm1b", "clean"],
    ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3b"],
    ["ankh_base", "ankh_large"],
]

# Color map
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

# X-axis renaming
X_AXIS_RENAME = {
    "alntmscore": "Alignment TM-Score",
    "fident": "Sequence Identity",
    "hfsp": "HFSP",
    "lddt": "lDDT",
    "rmsd": "RMSD",
}


def get_color(plm: str, color_map: Dict[str, str]) -> str:
    """Get color for a given PLM."""
    return color_map.get(
        plm,
        plt.cm.get_cmap("tab10")(
            list(color_map.keys()).index(plm) / len(color_map)
        ),
    )


def calculate_bar_positions(
    n_features: int, groups: List[List[str]]
) -> np.ndarray:
    """Calculate the x-positions for bars."""
    n_groups = len(groups)
    total_group_width = (
        sum(len(group) * BAR_WIDTH for group in groups)
        + (n_groups - 1) * GROUP_SPACING
    )
    return np.arange(n_features) * (total_group_width + FEATURE_SPACING)


def plot_group_bars(
    ax: plt.Axes,
    group: List[str],
    start_position: float,
    feature: str,
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    yerr_col: Optional[str],
    color_map: Dict[str, str],
    is_first_feature: bool,
) -> None:
    """Plot bars for a single group."""
    for i, plm in enumerate(group):
        plm_data = results_df[
            (results_df[group_col] == plm) & (results_df[x_col] == feature)
        ]
        if not plm_data.empty:
            yerr = (
                plm_data[yerr_col].values
                if yerr_col in plm_data.columns
                else None
            )
            color = get_color(plm, color_map)
            ax.bar(
                start_position + i * BAR_WIDTH,
                plm_data[y_col].values[0],
                BAR_WIDTH,
                yerr=yerr,
                label=plm if is_first_feature else "",
                color=color,
            )


def plot_results(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    title: str,
    x_col: str = "x_col",
    y_col: str = "r_square",
    group_col: str = "plm",
    yerr_col: Optional[str] = "std_err",
    fontsize: int = 16,
    color_map: Optional[Dict[str, str]] = None,
    x_axis_rename: Optional[Dict[str, str]] = None,
    selected_x_values: Optional[List[str]] = None,
) -> None:
    """Plot bar chart results on the given axes."""
    if color_map is None:
        color_map = COLOR_MAP

    if selected_x_values is None:
        selected_x_values = results_df[x_col].unique()
    else:
        results_df = results_df[results_df[x_col].isin(selected_x_values)]

    n_features = len(selected_x_values)
    x = calculate_bar_positions(n_features, PLM_GROUPS)

    for feature_idx, feature in enumerate(selected_x_values):
        feature_start = x[feature_idx]
        group_start = feature_start

        for group in PLM_GROUPS:
            plot_group_bars(
                ax,
                group,
                group_start,
                feature,
                results_df,
                x_col,
                y_col,
                group_col,
                yerr_col,
                color_map,
                feature_idx == 0,
            )
            group_start += len(group) * BAR_WIDTH + GROUP_SPACING

    # Set up axes labels and title
    ax.set_xlabel("Features", fontsize=fontsize + 2)
    ax.set_ylabel("R$^2$", fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 4)

    # Set x-ticks and labels
    total_group_width = (
        sum(len(group) * BAR_WIDTH for group in PLM_GROUPS)
        + (len(PLM_GROUPS) - 1) * GROUP_SPACING
    )
    ax.set_xticks(x + total_group_width / 2 - GROUP_SPACING / 2)
    x_labels = (
        [x_axis_rename.get(label, label) for label in selected_x_values]
        if x_axis_rename
        else selected_x_values
    )
    ax.set_xticklabels(x_labels, fontsize=fontsize)

    # Set y-axis tick font size
    ax.tick_params(axis="y", labelsize=fontsize)

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust x-axis limits to prevent cutting off edge bars
    ax.set_xlim(x[0] - BAR_WIDTH, x[-1] + total_group_width + BAR_WIDTH)
    ax.set_ylim(0, 1)


def create_plots(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    titles: Union[str, List[str]],
    output_file: str,
    x_col: str = "x_col",
    y_col: str = "r_square",
    group_col: str = "plm",
    yerr_col: Optional[str] = "std_err",
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (20, 10),
    fontsize: int = 16,
    x_axis_rename: Optional[Dict[str, str]] = None,
    selected_x_values: Optional[List[str]] = None,
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
            group_col,
            yerr_col,
            fontsize,
            color_map,
            x_axis_rename,
            selected_x_values,
        )
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Create legend
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fontsize=fontsize + 4,
        title="pLMs",
        title_fontsize=fontsize,
        ncol=4,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


# Usage example
if __name__ == "__main__":
    # Load your data here
    # data = pd.read_csv("your_data.csv")
    # selected_x_values = ["value1", "value2", "value3"]  # Replace with your desired x-axis values
    # create_plots(data, "Your Title", "output.png", x_col="feature", y_col="r_square", group_col="plm", yerr_col="std_err", color_map=COLOR_MAP, x_axis_rename=X_AXIS_RENAME, selected_x_values=selected_x_values)
    pass
