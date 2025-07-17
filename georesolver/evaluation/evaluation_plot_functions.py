"""methods for plotting graph"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.use("Agg")

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from georesolver.common.settings import PathSettings

path_settings = PathSettings()

font = {"weight": "bold", "size": 16}  #'family' : 'normal',
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
fontsize_axis = 17
font_size_alone = 14
matplotlib.rc("font", **font)

markers = ["o", "s", "v", "^"]
linestyles = ["-", "--", "-.", ":"]


def homogenize_legend(ax, legend_location, legend_size=14):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, Line2D):
            new_handles.append(h)
        elif isinstance(h, Polygon):
            new_handles.append(
                Line2D([], [], linestyle=h.get_linestyle(), color=h.get_edgecolor())
            )
    ax.legend(
        loc=legend_location,
        prop={"size": legend_size},
        handles=new_handles,
        labels=labels,
    )


colors_blind = [
    ["blue", (0, 114.0 / 255, 178.0 / 255)],
    ["reddish_purple", (204.0 / 255, 121.0 / 255, 167.0 / 255)],
    ["black", (0, 0, 0)],
    ["orange", (230.0 / 255, 159.0 / 255, 0)],
    ["sky_blue", (86.0 / 255, 180.0 / 255, 233.0 / 255)],
    ["vermillon", (213.0 / 255, 94.0 / 255, 0)],
    ["bluish_green", (0, 158.0 / 255, 115.0 / 255)],
    ["dark_green", (41.0 / 255, 94.0 / 255, 17.0 / 255)],
    ["yellow", (240.0 / 255, 228.0 / 255, 66.0 / 255)],
    ["brown", (110.0 / 255, 74.0 / 255, 60.0 / 255)],
    ["grey", (221.0 / 255, 221.0 / 255, 221.0 / 255)],
    ["light_green", (000.0 / 255, 158.0 / 255, 115.0 / 255)],
]


def ecdf(data: list, array: bool = True):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    if not array:
        return pd.DataFrame({"x": x, "y": y})
    else:
        return x, y


def get_median(target_results: dict, key: str, metric: str = "d_error") -> float:
    """return the median distance error"""
    return round(
        np.median([r[key][metric] for r in target_results.values()]),
        2,
    )


def get_x_label(metric_evaluated) -> str:
    x_label = ""
    if metric_evaluated == "d_error":
        x_label = "Geolocation error (km)"
    if metric_evaluated == "rtt":
        x_label = "RTT (ms)"

    return x_label


def plot_cdf(
    x: list,
    y: list,
    output_path: str,
    x_label: str,
    y_label: str,
    x_lim_right: int = None,
    x_lim_left: int = None,
    y_lim: int = 1,
    legend_pos: str = "upper left",
    legend_size: int = 10,
    x_log_scale: bool = False,
    metric_evaluated: str = "",
) -> None:

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, y, color=colors_blind[0][1])
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel(y_label, fontsize=fontsize_axis)

    if metric_evaluated == "d_error":
        plot_limit(limit=40, metric_evaluated=metric_evaluated)
    elif metric_evaluated == "rtt":
        plot_limit(limit=2, metric_evaluated=metric_evaluated)
    else:
        pass

    homogenize_legend(ax1, legend_pos, legend_size=legend_size)

    plt.tight_layout()
    if x_log_scale:
        plt.xscale("log")
    if x_lim_right:
        plt.xlim(right=x_lim_right)
    plt.xlim(left=x_lim_left)
    plt.ylim((0, y_lim))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_multiple_cdf(
    cdfs: list,
    output_path: str,
    metric_evaluated: str = "",
    legend_outside: str = False,
    legend_pos: str = "upper left",
    legend_size: int = 10,
    under_padding: int = 0,
    x_limit_left: int = 1,
    x_limit_right: int = None,
    x_label: str = "",
    y_label: str = "CDF of targets",
    x_log_scale: bool = True,
    y_log_scale: bool = False,
    fontsize_axis: int = 17,
    legend_fontsize: int = 8,
) -> None:

    fig, ax1 = plt.subplots(1, 1)

    for i, (x, y, label) in enumerate(cdfs):
        ax1.plot(x, y, label=label, color=colors_blind[i][1])

    x_label = get_x_label(metric_evaluated) if not x_label else x_label
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel(y_label, fontsize=fontsize_axis)

    if metric_evaluated == "d_error":
        plot_limit(
            limit=40, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    elif metric_evaluated == "rtt":
        plot_limit(
            limit=2, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    else:
        pass

    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1), fontsize=legend_fontsize)
    else:
        plt.legend(loc=legend_pos, fontsize=legend_fontsize)
    if x_log_scale:
        plt.xscale("log")
    if y_log_scale:
        plt.yscale("log")
    if x_limit_left:
        plt.xlim(left=x_limit_left)
    if x_limit_right:
        plt.xlim(right=x_limit_right)

    homogenize_legend(ax1, legend_pos, legend_size=legend_size)

    plt.tight_layout()
    plt.ylim((0, 1))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_multiple_cdfs_with_dates(
    cdfs: list,
    output_path: str,
    metric_evaluated: str,
    legend_outside: str = False,
    legend_pos: str = "upper left",
    legend_size: int = 10,
    under_padding: int = 0,
    x_limit_left: int = 1,
    x_label: str = "",
    y_label: str = "CDF of targets",
    x_log_scale: bool = True,
    y_log_scale: bool = False,
    legend_fontisize: int = 10,
) -> None:

    _, ax1 = plt.subplots(1, 1)
    x = cdfs[0][0]
    for i, (_, y, label) in enumerate(cdfs):
        ax1.plot(x, y, label=label, color=colors_blind[i][1])

    x_label = get_x_label(metric_evaluated) if not x_label else x_label
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel(y_label, fontsize=fontsize_axis)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for label in ax1.get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    if metric_evaluated == "d_error":
        plot_limit(
            limit=40, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    elif metric_evaluated == "rtt":
        plot_limit(
            limit=2, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    else:
        pass

    homogenize_legend(ax1, legend_pos, legend_size=legend_fontisize)

    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
    else:
        plt.legend(loc=legend_pos, fontsize=legend_fontisize)
    if x_log_scale:
        plt.xscale("log")
    if x_limit_left:
        plt.xlim(left=x_limit_left)

    plt.tight_layout()
    plt.ylim((0, 1))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def get_proportion_under(x, y, threshold: int = 40) -> int:
    proportion_of_ip = 1
    for i, distance in enumerate(x):
        if distance > threshold:
            proportion_of_ip = y[i]
            break

    return proportion_of_ip


def get_proportion_over(x, y, threshold: int = 40) -> int:
    proportion_of_ip = 1
    for i, value in enumerate(x):
        if value >= threshold:
            proportion_of_ip = y[i]
            break

    return proportion_of_ip


def plot_limit(
    limit: float,
    metric_evaluated: str = "d_error",
    under_padding_d: int = 29,
    under_padding_rtt: int = 0.95,
) -> None:
    x = [limit, limit]
    y = [0, 1]

    # Plotting the line with dots
    plt.plot(x, y, linestyle="dotted", color="grey")
    plt.annotate(
        f"{x[0]} {'km' if metric_evaluated == 'd_error' else 'ms'}",
        xy=(
            (
                x[0] - under_padding_d
                if metric_evaluated == "d_error"
                else x[0] - under_padding_rtt
            ),
            1 - 0.1,
        ),
        size=12,
    )
