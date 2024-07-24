from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from into_the_unknown.graph.datashader_plot_tmp import generate_ds_image
from tqdm import tqdm

class ProteinAnalysis:
    def __init__(self, dataset_dir: str, out_dir: str):
        self.BASE_DIR = Path(dataset_dir)
        self.BASE_NAME = self.BASE_DIR.stem
        self.CORR_DIR = Path(f"{out_dir}/corr_{self.BASE_NAME}")
        self.CORR_DIR.mkdir(parents=True, exist_ok=True)
        self.R_SQUARE_FILE = self.CORR_DIR / "r_square.csv"
        self.R_SQUARE_PLOT = self.CORR_DIR / "r_square.png"
        self.color_map = {
            "prott5": "#377eb8",
            "prottucker": "#373bb7", # 1e2060
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
        self.plms = list(self.color_map.keys())

    def plot_correlations(self, df_fs):
        x_columns = [
            "fident",
            "lddt",
            "alntmscore",
            "rmsd",
            "hfsp",
            # "aa_comp_diff",
        ]

        results = []
        total_iterations = len(self.plms) * 2 * len(x_columns)  # 2 for PCA and non-PCA

        with tqdm(total=total_iterations, desc="Generating plots") as pbar:
            for is_pca in [False, True]:
                for x_col in x_columns:
                    corr_dir = self.CORR_DIR / x_col / ("pca" if is_pca else "no_pca")
                    corr_dir.mkdir(parents=True, exist_ok=True)

                    for plm in self.plms:
                        pbar.set_postfix_str(f"PLM: {plm}, PCA: {is_pca}, Correlation: {x_col}")

                        dist_col = f"{'pca_' if is_pca else ''}distance_{plm}"
                        output_file = corr_dir / f"{plm}.png"

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
                                    "distance_col": dist_col,
                                    "x_col": x_col,
                                    "r_square": r_square,
                                    "std_err": std_err,
                                }
                            )
                        else:
                            print(f"Warning: Column {dist_col} not found in the dataframe. Skipping this PLM.")

                        pbar.update(1)

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.R_SQUARE_FILE, index=False)

        non_pca_results_df = results_df[~results_df["is_pca"]]
        pca_results_df = results_df[results_df["is_pca"]]
        self.create_plots(non_pca_results_df, pca_results_df, fontsize=20)

    def plot_results(self, ax, results_df, title, fontsize):
        bar_width = 0.1
        group_spacing = 0.2
        num_groups = len(results_df["x_col"].unique())
        num_bars_per_group = len(self.plms)

        for i, pLM in enumerate(self.plms):
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
                    color=self.color_map[pLM],
                )
        ax.set_xlabel("Comparison", fontsize=fontsize)
        ax.set_ylabel("R^2", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize + 4)
        ax.set_xticks(indices + bar_width * num_bars_per_group / 2)
        ax.set_xticklabels(results_df["x_col"].unique(), fontsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.grid(axis="y", linestyle="--")

    def create_plots(
        self, non_pca_results_df, pca_results_df=None, fontsize=16
    ):
        if pca_results_df is not None and not pca_results_df.empty:
            fig, axes = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
            self.plot_results(
                axes[0],
                non_pca_results_df,
                "R^2 for Non-PCA Distance",
                fontsize,
            )
            self.plot_results(
                axes[1],
                pca_results_df,
                "R^2 for PCA Distance",
                fontsize,
            )
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            fig, ax = plt.subplots(figsize=(20, 10))
            self.plot_results(
                ax,
                non_pca_results_df,
                "R^2 for Non-PCA Distance",
                fontsize,
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
        plt.savefig(self.R_SQUARE_PLOT, bbox_inches="tight")
        plt.close(fig)  # Close the figure instead of showing it

    def run_analysis(self, df_fs):
        self.plot_correlations(df_fs)

def main():
    base_dir = "data/s_pombe"
    out_dir = "out"
    protein_analysis = ProteinAnalysis(dataset_dir=base_dir, out_dir=out_dir)

    # Load the DataFrame from Part 2
    df_fs = pd.read_csv(Path(base_dir) / f"{Path(base_dir).stem}_final.csv")

    protein_analysis.run_analysis(df_fs)

if __name__ == "__main__":
    main()