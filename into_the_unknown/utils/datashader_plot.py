"""
datashader_plot.py

This script generates a plot using Datashader to visualize large datasets.
It allows for custom aggregation functions (mean or count) and can optionally
add a linear regression line using Scipy. The plot can be saved to a file or
displayed directly.

Parameters:
- data: DataFrame containing the data to be plotted.
- x_col: Column name for the x-axis values.
- y_col: Column name for the y-axis values.
- agg_func: Aggregation function ('mean' or 'count') to be applied to the data.
- plot_width: Width of the plot in pixels.
- plot_height: Height of the plot in pixels.
- cmap: Colormap to be used for the plot.
- add_regression: Boolean flag to add a regression line to the plot.
- output_file: Path to save the plot. If None, the plot will be displayed.

Example usage:
    np.random.seed(42)
    n = 100000
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'value': np.random.rand(n)
    })
    generate_ds_image(df, 'x', 'y', agg_func='mean', add_regression=True)
"""

import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import linregress


def generate_ds_image(
    df,
    x_col,
    y_col,
    agg_col=None,
    agg_func="mean",
    plot_width=1000,
    plot_height=1000,
    point_px_size=1,
    cmap=cc.fire,
    add_regression=False,
    output_file=None,
):
    # Define the canvas
    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(df[x_col].min(), df[x_col].max()),
        y_range=(df[y_col].min(), df[y_col].max()),
    )

    # Aggregate the data based on the provided aggregation function
    if agg_func == "mean":
        agg = canvas.points(df, x_col, y_col, ds.mean(agg_col))
    elif agg_func == "count":
        agg = canvas.points(df, x_col, y_col, ds.count())
    else:
        raise ValueError("agg_func must be 'mean' or 'count'")

    # Create the plot
    plot = tf.shade(tf.spread(agg, px=point_px_size), cmap=cmap)

    # Convert the plot to an image
    image = plot.to_pil()

    # Create a matplotlib figure and axis without extra whitespace
    fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100), tight_layout=True)

    # Set the extent to match the x and y ranges
    extent = [
        df[x_col].min(),
        df[x_col].max(),
        df[y_col].min(),
        df[y_col].max(),
    ]
    ax.imshow(image, aspect="auto", extent=extent)

    # Create a colorbar
    norm = Normalize(vmin=np.nanmin(agg.data), vmax=np.nanmax(agg.data))
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([])

    # Add the colorbar to the right of the plot
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Average Value" if agg_func == "mean" else "Count")

    # Optionally add the regression line
    if add_regression:
        X = df[x_col].values
        y = df[y_col].values

        # Remove NaN values
        mask = ~np.isnan(X) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        slope, intercept, r_value, p_value, std_err = linregress(X, y)
        r_square = r_value**2

        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=2)

        # Add the R^2 value text
        ax.text(
            0.05,
            0.95,
            f"$R^2 = {r_square:.2f}$",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            color="black",
        )

    # Set axis labels
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return r_square, std_err


# Example usage with a sample dataframe
if __name__ == "__main__":
    np.random.seed(42)
    n = 100000
    df = pd.DataFrame(
        {
            "x": np.random.randn(n),
            "y": np.random.randn(n),
            "value": np.random.rand(n),
        }
    )
    generate_ds_image(
        df, "x", "y", agg_col="value", agg_func="mean", add_regression=True
    )
