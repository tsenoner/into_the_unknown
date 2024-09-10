from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from into_the_unknown.plots.utils import COLOR_MAP, X_AXIS_RENAME, create_plots


def generate_r2_file(data_dir: Path, r2_file: Path, nr_test_set: int) -> None:
    results = []
    for metric_path in data_dir.glob("**/*test_metrics.txt"):
        plm = metric_path.parent.stem
        feature = metric_path.parents[1].stem
        with metric_path.open() as f:
            content = f.read()
            std_err = float(
                content.split("RMSE:")[1].split("\n")[0].strip()
            ) / (nr_test_set**0.5)
            r_square = float(content.split("R2:")[1].split("\n")[0].strip())
        results.append(f"{plm},{feature},{r_square},{std_err}")

    with r2_file.open("w") as out_handle:
        out_handle.write("plm,feature,r_square,std_err\n")
        out_handle.write("\n".join(results))


def create_plot(
    base_path: Path,
    data_subdir: str,
    title: str,
    selected_x_values: List[str],
    plot_func: callable,
    **kwargs,
) -> None:
    data_dir = base_path / "out" / data_subdir
    r2_file = data_dir / "r2.csv"
    r2_png = (
        data_dir / f"r2{'_extended' if len(selected_x_values) > 3 else ''}.png"
    )

    if not r2_file.exists():
        generate_r2_file(data_dir, r2_file, nr_test_set=564_832)

    df = pd.read_csv(r2_file)
    plot_func(
        data=df,
        titles=title,
        output_file=r2_png,
        x_col="feature",
        y_col="r_square",
        yerr_col="std_err",
        group_col="plm",
        color_map=COLOR_MAP,
        x_axis_rename=X_AXIS_RENAME,
        selected_x_values=selected_x_values,
        fontsize=22,
        **kwargs,
    )


def create_distance_correlation_plot(
    base_path: Path, use_pca: bool, selected_x_values: List[str]
) -> None:
    data_dir = base_path / "out" / "corr_swissprot"
    r2_file = data_dir / "r_square.csv"
    r2_png = (
        data_dir
        / f"r2_{'PCA' if use_pca else 'noPCA'}{'_extended' if len(selected_x_values) > 3 else ''}.png"
    )

    df = pd.read_csv(r2_file)
    df = df[df["is_pca"] == use_pca]

    create_plots(
        data=df,
        titles=f"Euclidean distance {'with' if use_pca else 'without'} PCA",
        output_file=r2_png,
        x_col="x_col",
        y_col="r_square",
        yerr_col="std_err",
        group_col="plm",
        color_map=COLOR_MAP,
        x_axis_rename=X_AXIS_RENAME,
        selected_x_values=selected_x_values,
        fontsize=22,
    )


def main():
    base_path = Path(".")

    original_x_values = ["fident", "alntmscore", "hfsp"]
    extended_x_values = ["fident", "alntmscore", "lddt", "rmsd", "hfsp"]

    plot_configs = [
        ("feature_swissprot", "Trained models"),
        ("linreg_swissprot", "Linear regression models"),
    ]

    for data_subdir, title_prefix in plot_configs:
        for x_values in [original_x_values, extended_x_values]:
            suffix = " (extended)" if len(x_values) > 3 else ""
            print(f"Generating {title_prefix}{suffix}...")
            create_plot(
                base_path,
                data_subdir,
                f"{title_prefix}{suffix}",
                x_values,
                create_plots,
            )

    for use_pca in [False, True]:
        pca_status = "with" if use_pca else "without"
        print(f"Generating distance correlation plots {pca_status} PCA...")
        for x_values in [original_x_values, extended_x_values]:
            create_distance_correlation_plot(base_path, use_pca, x_values)

    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
