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

    def __init__(self, meta: ColumnMeta) -> None:
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


class NumericalHist(BaseHist[str]):
    name = "numerical"

    class NumericalData(NamedTuple):
        bins: np.ndarray = None

    def fit(self, data: pd.Series):
        args = self.meta.args
        self.min = args.get("min", data.min())
        self.max = args.get("max", data.max())

        main_param = args.get("main_param", None)
        if main_param and (isinstance(main_param, int)):
            self.bin_n = main_param
        else:
            self.bin_n = args.get("bins", 20)

        self.bins = np.histogram_bin_edges(
            data, bins=self.bin_n, range=(self.min, self.max)
        )

    def process(self, data: pd.Series) -> NumericalData:
        return self.NumericalData(np.histogram(data, self.bins)[0])

    def visualise(self, data: dict[str, NumericalData]) -> Figure:
        fig, ax = plt.subplots()
        x = self.bins[:-1]
        w = (x[1] - x[0]) / len(data)

        for i, (name, d) in enumerate(data.items()):
            h = d.bins / d.bins.sum()
            ax.bar(x + w * i, h, width=w, label=name)
        ax.legend()
        return fig


def get_hists():
    return find_subclasses(BaseHist)
