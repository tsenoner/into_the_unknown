import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from into_the_unknown.utils.datashader_plot import generate_ds_image


def plot_correlations(df_fs, out_dir, r_square_file, color_map):
    x_columns = ["fident", "lddt", "alntmscore", "rmsd", "hfsp", "aa_comp_diff"]
    plms = list(color_map.keys())
    results = []
    total_iterations = len(plms) * 2 * len(x_columns)

    with tqdm(total=total_iterations, desc="Generating plots") as pbar:
        for is_pca in [False, True]:
            for x_col in x_columns:
                out_subdir = out_dir / x_col / ("pca" if is_pca else "no_pca")
                out_subdir.mkdir(parents=True, exist_ok=True)

                for plm in plms:
                    pbar.set_postfix_str(
                        f"PLM: {plm}, PCA: {is_pca}, Correlation: {x_col}"
                    )
                    dist_col = f"{'pca_' if is_pca else ''}dist_{plm}"
                    output_file = out_subdir / f"{plm}.png"

                    if dist_col in df_fs.columns:
                        r_square, std_err = generate_ds_image(
                            df=df_fs,
                            x_col=x_col,
                            y_col=dist_col,
                            agg_func="count",
                            plot_width=1200,
                            plot_height=800,
                            point_px_size=1,
                            add_regression=True,
                            output_file=output_file,
                        )
                        results.append(
                            {
                                "plm": plm,
                                "is_pca": is_pca,
                                "dist_col": dist_col,
                                "x_col": x_col,
                                "r_square": r_square,
                                "std_err": std_err,
                            }
                        )
                    else:
                        print(
                            f"Warning: Column {dist_col} not found in the dataframe. Skipping this PLM."
                        )

                    pbar.update(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(r_square_file, index=False)
    return results_df


def plot_results(ax, results_df, title, fontsize, color_map):
    bar_width = 0.1
    group_spacing = 0.2
    num_groups = len(results_df["x_col"].unique())
    num_bars_per_group = len(color_map)

    for i, pLM in enumerate(color_map.keys()):
        group = results_df[results_df["plm"] == pLM]
        if not group.empty:
            indices = np.arange(num_groups) * (
                bar_width * num_bars_per_group + group_spacing
            )
            ax.bar(
                indices + i * bar_width,
                group["r_square"],
                yerr=group["std_err"],
                width=bar_width,
                label=pLM,
                color=color_map[pLM],
            )
    ax.set_xlabel("Comparison", fontsize=fontsize)
    ax.set_ylabel("R^2", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 4)
    ax.set_xticks(indices + bar_width * num_bars_per_group / 2)
    ax.set_xticklabels(results_df["x_col"].unique(), fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(axis="y", linestyle="--")


def create_plots(
    non_pca_results_df, pca_results_df, r_square_plot, color_map, fontsize=16
):
    if not pca_results_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
        plot_results(
            axes[0],
            non_pca_results_df,
            "R^2 for Non-PCA Distance",
            fontsize,
            color_map,
        )
        plot_results(
            axes[1], pca_results_df, "R^2 for PCA Distance", fontsize, color_map
        )
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_results(
            ax,
            non_pca_results_df,
            "R^2 for Non-PCA Distance",
            fontsize,
            color_map,
        )
        handles, labels = ax.get_legend_handles_labels()

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
    plt.savefig(r_square_plot, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create correlation plots")
    parser.add_argument(
        "-i", "--input_csv", required=True, help="Input CSV file from Part 2"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory for plots"
    )
    args = parser.parse_args()

    df_fs = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r_square_file = output_dir / "r_square.csv"
    r_square_plot = output_dir / "r_square.png"

    color_map = {
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

    results_df = plot_correlations(df_fs, output_dir, r_square_file, color_map)
    non_pca_results_df = results_df[~results_df["is_pca"]]
    pca_results_df = results_df[results_df["is_pca"]]
    create_plots(
        non_pca_results_df,
        pca_results_df,
        r_square_plot,
        color_map,
        fontsize=20,
    )


if __name__ == "__main__":
    main()
