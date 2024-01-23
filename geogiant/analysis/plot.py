"""methods for plotting graph"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def ecdf(data, array: bool = True):
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


def get_error_bars(results: dict, max_probing_budget: int) -> tuple:
    """parse data for plot"""
    x = []
    y = []
    e = []
    print(results)
    for budget, (median_distance, deviation) in results.items():
        x.append(budget)
        y.append(median_distance)
        e.append(deviation)

        if budget > max_probing_budget:
            break

    return x, y, e


def plot_median_error_per_finger_printing_method(results: dict) -> None:
    fig, ax1 = plt.subplots(1, 1)

    x, y, e = get_error_bars(results["answers"], 1000)
    ax1.plot(x, y, label="frontend fingerprint")

    x, y, e = get_error_bars(results["subnet"], 1000)
    ax1.plot(x, y, label="subnet fingerprint")

    x, y, e = get_error_bars(results["bgp_prefix"], 1000)
    ax1.plot(x, y, label="bgp prefix fingerprint")

    x, y, e = get_error_bars(results["pop_id"], 1000)
    ax1.plot(x, y, label="pop fingerprint")

    plt.xlabel("Probing Budget [nb pings]")
    plt.ylabel("Median Error [km]")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()
    plt.yscale("log")
    plt.title(f"Geolocation Error Function of Probing Budget", fontsize=13)
    plt.savefig(path_settings.FIGURE_PATH / "mapping_scores_evaluation.pdf")
    plt.savefig(path_settings.FIGURE_PATH / "mapping_scores_evaluation.png")
    plt.show()
