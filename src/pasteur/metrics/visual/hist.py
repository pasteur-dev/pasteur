from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ...metadata import ColumnMeta
from ...utils import find_subclasses

A = TypeVar("A")


class BaseHist(ABC, Generic[A]):
    name = None

    def __init__(self, col: str, meta: ColumnMeta) -> None:
        self.col = col
        self.meta = meta

    @abstractmethod
    def fit(self, data: pd.Series):
        pass

    @abstractmethod
    def process(self, data: pd.Series) -> A:
        pass

    @abstractmethod
    def visualise(self, data: dict[str, A]) -> list[Figure] | Figure:
        pass


class BaseRefHist(BaseHist):
    @abstractmethod
    def fit(self, data: pd.Series, ref: pd.Series):
        pass

    @abstractmethod
    def process(self, data: pd.Series, ref: pd.Series) -> A:
        pass


class NumericalHist(BaseHist["NumericalHist.NumericalData"]):
    name = "numerical"

    class NumericalData(NamedTuple):
        bins: np.ndarray = None

    def fit(self, data: pd.Series):
        args = self.meta.args
        metrics = self.meta.metrics

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

    def process(self, data: pd.Series) -> NumericalData:
        return self.NumericalData(np.histogram(data, self.bins)[0])

    def visualise(self, data: dict[str, NumericalData]) -> Figure:
        fig, ax = plt.subplots()
        x = self.bins[:-1]
        w = (x[1] - x[0]) / len(data)

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            h = d.bins / d.bins.sum()
            ax.bar(x + w * i, h, width=w, label=name, log=is_log)
        ax.legend()
        ax.set_title(self.col.capitalize())
        plt.tight_layout()
        return fig


class CategoricalHist(BaseHist["CategoricalHist.CategoricalData"]):
    name = "categorical"

    class CategoricalData(NamedTuple):
        counts: pd.Series = None

    def fit(self, data: pd.Series):
        self.cols = data.value_counts().sort_values(ascending=False).index

    def process(self, data: pd.Series) -> CategoricalData:
        return self.CategoricalData(data.value_counts())

    def visualise(self, data: dict[str, CategoricalData]) -> Figure:
        fig, ax = plt.subplots()

        x = np.array(range(len(self.cols)))
        w = 1 / len(data)

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            ax.bar(
                x - 0.5 + w * i,
                d.counts[self.cols].to_numpy(),
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        if self.name == "categorical":
            plt.xticks(x, self.cols.to_numpy())
            rot = min(3 * len(self.cols), 90)
            rot = rot if rot > 10 else 0
            plt.setp(ax.get_xticklabels(), rotation=rot, horizontalalignment="right")

        ax.legend()
        ax.set_title(self.col.capitalize())
        plt.tight_layout()
        return fig


class OrdinalHist(CategoricalHist):
    name = "ordinal"

    def fit(self, data: pd.Series):
        self.cols = pd.Index(np.sort(data.unique()))


def get_hists():
    return find_subclasses(BaseHist)
