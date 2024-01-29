"""methods for plotting graph"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from geogiant.common.settings import PathSettings

path_settings = PathSettings()


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


def get_error_bars(results: dict) -> tuple:
    """parse data for plot"""
    x = []
    y = []
    e = []
    for budget, (median_distance, deviation) in results.items():
        x.append(budget)
        y.append(median_distance)
        e.append(deviation)

    return x, y, e


def get_plot(results: dict) -> tuple[tuple, tuple]:
    """parse data for plot"""
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for budget, (m_d, w_m_d) in results.items():
        x1.append(budget)
        y1.append(m_d)

        x2.append(budget)
        y2.append(w_m_d)

    return (x1, y1), (x2, y2)


def plot_no_pings(results: dict, out_file: str) -> None:
    fig, ax1 = plt.subplots(1, 1)

    p1, p2 = get_plot(results["answers"])
    ax1.plot(p1[0], p1[1], label="frontend fingerprint")
    ax1.plot(p2[0], p2[1], label="w frontend fingerprint")

    p1, p2 = get_plot(results["subnet"])
    ax1.plot(p1[0], p1[1], label="subnet fingerprint")
    ax1.plot(p2[0], p2[1], label="w subnet fingerprint")

    p1, p2 = get_plot(results["bgp_prefix"])
    ax1.plot(p1[0], p1[1], label="bgp fingerprint")
    ax1.plot(p2[0], p2[1], label="w bgp fingerprint")

    plt.xlabel("Probing Budget [nb pings]")
    plt.ylabel("Median Error [km]")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()
    plt.yscale("log")
    plt.title(f"Geolocation Error Function of Probing Budget", fontsize=13)
    plt.savefig(path_settings.FIGURE_PATH / out_file)
    plt.show()


def plot_median_error_per_finger_printing_method(results: dict, out_file: str) -> None:
    fig, ax1 = plt.subplots(1, 1)

    for granularity, result in results.items():
        x, y, _ = get_error_bars(result)
        ax1.plot(x, y, label=f"{granularity} fingerprint")

    plt.xlabel("Probing Budget [nb pings]")
    plt.ylabel("Median Error [km]")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()
    plt.yscale("log")
    plt.title(f"Geolocation Error Function of Probing Budget", fontsize=13)
    plt.savefig(path_settings.FIGURE_PATH / out_file)
    plt.show()
