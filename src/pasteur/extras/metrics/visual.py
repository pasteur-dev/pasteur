from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ...metadata import ColumnMeta, Metadata
from ...metric import ColumnMetric, RefColumnMetric
from ...utils.mlflow import load_matplotlib_style, mlflow_log_hists

A = TypeVar("A")


def _percent_formatter(x, pos):
    return f"{100*x:.1f}%"


def _gen_hist(
    meta: ColumnMeta,
    title: str,
    bins: np.ndarray,
    heights: dict[str, np.ndarray],
    xticks_x=None,
    xticks_label=None,
):
    fig, ax = plt.subplots()
    x = bins[:-1]
    w = (x[1] - x[0]) / len(heights)

    is_log = meta.metrics.y_log == True
    for i, (name, h) in enumerate(heights.items()):
        ax.bar(x + w * i, h / h.sum(), width=w, label=name, log=is_log)

    ax.legend()
    ax.set_title(title)
    ax.yaxis.set_major_formatter(_percent_formatter)

    if xticks_x is not None:
        ax.set_xticks(xticks_x, xticks_label)

    plt.tight_layout()
    return fig


def _gen_bar(
    meta: ColumnMeta, title: str, cols: list[str], counts: dict[str, np.ndarray]
):
    fig, ax = plt.subplots()

    x = np.array(range(len(cols)))
    w = 0.9 / len(counts)

    is_log = meta.metrics.y_log == True
    for i, (name, c) in enumerate(counts.items()):
        h = c / c.sum()
        ax.bar(
            x - 0.45 + w * i,
            h,
            width=w,
            align="edge",
            label=name,
            log=is_log,
        )

    plt.xticks(x, cols)
    rot = min(3 * len(cols), 90)
    if rot > 10:
        plt.setp(ax.get_xticklabels(), rotation=rot, horizontalalignment="right")

    ax.legend()
    ax.set_title(title)
    ax.yaxis.set_major_formatter(_percent_formatter)

    plt.tight_layout()
    return fig


class NumericalHist(ColumnMetric[np.ndarray]):
    name = "numerical"

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        self.meta = meta
        self.table = table
        self.col = col
        args = meta.args
        metrics = meta.metrics

        # Get maximums
        if metrics.x_min is not None:
            x_min = metrics.x_min
        else:
            x_min = args.get("min", data.min())
        if metrics.x_max is not None:
            x_max = metrics.x_max
        else:
            x_max = args.get("max", data.max())

        main_param = args.get("main_param", None)
        if main_param and (isinstance(main_param, int)):
            self.bin_n = main_param
        else:
            self.bin_n = args.get("bins", 20)

        self.bins = np.histogram_bin_edges(data, bins=self.bin_n, range=(x_min, x_max))

    def process(self, data: pd.Series):
        return np.histogram(data, self.bins, density=True)[0]

    def visualise(
        self,
        data: dict[str, np.ndarray],
        comparison: bool = False,
        wrk_set: str = "wrk",
        ref_set: str = "ref",
    ):

        load_matplotlib_style()
        v = _gen_hist(
            self.meta,
            self.col.capitalize(),
            self.bins,
            data,
        )

        mlflow_log_hists(self.table, self.col, v)
