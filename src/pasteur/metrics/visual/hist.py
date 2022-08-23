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
    def visualise(self, data: dict[str, A]) -> dict[str, Figure] | Figure:
        pass


class BaseRefHist(BaseHist[A], Generic[A]):
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
        w = 0.9 / len(data)

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            ax.bar(
                x - 0.45 + w * i,
                d.counts[self.cols].to_numpy(),
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        plt.xticks(x, self.cols.to_numpy())
        rot = min(3 * len(self.cols), 90)
        if rot > 10:
            plt.setp(ax.get_xticklabels(), rotation=rot, horizontalalignment="right")

        ax.legend()
        ax.set_title(self.col.capitalize())
        plt.tight_layout()
        return fig


class OrdinalHist(CategoricalHist):
    name = "ordinal"

    def fit(self, data: pd.Series):
        self.cols = pd.Index(np.sort(data.unique()))


class FixedHist(BaseHist[None]):
    name = "fixed"

    def fit(self, data: pd.Series):
        pass

    def process(self, data: pd.Series) -> None:
        pass

    def visualise(self, data: dict[str, None]) -> None:
        pass


class DateHist(BaseRefHist["DateHist.DateData"]):
    name = "date"

    class DateData(NamedTuple):
        pass

    def fit(self, data: pd.Series, ref: pd.Series):
        pass

    def process(self, data: pd.Series, ref: pd.Series) -> DateData:
        return self.DateData()

    def visualise(self, data: dict[str, DateData]) -> dict[str, Figure] | Figure:
        return {}


class TimeHist(BaseHist["TimeHist.TimeData"]):
    name = "time"

    class TimeData(NamedTuple):
        counts: pd.Series = None

    def fit(self, data: pd.Series):
        if "main_param" in self.meta.args:
            self.span = self.meta.args["main_param"].split(".")[-1]
        elif "span" in self.meta.args:
            self.span = self.meta.args["span"].split(".")[-1]
        else:
            self.span = "halfhour"

    def process(self, data: pd.Series) -> TimeData:
        hours = data.dt.hour
        if self.span == "hour":
            segments = hours
        else:
            half_hours = data.dt.minute > 29
            segments = 2 * hours + half_hours
        vc = segments.value_counts()
        vc /= vc.sum()
        return self.TimeData(vc)

    def visualise(self, data: dict[str, TimeData]) -> Figure:
        fig, ax = plt.subplots()

        if self.span == "hour":
            seg_len = 24
            mult = 1
        else:
            seg_len = 48
            mult = 2

        x = np.array(range(seg_len))
        w = 1 / len(data)

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            segments = d.counts.reindex(range(seg_len), fill_value=0).to_numpy()
            ax.bar(
                x - 0.5 + w * i,
                segments,
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
        tick_x = mult * np.array(hours)
        tick_label = [f"{hour:02d}:00" for hour in hours]
        plt.xticks(tick_x, tick_label)

        ax.legend()
        ax.set_title(self.col.capitalize())
        plt.tight_layout()
        return fig


class DatetimeHist(BaseRefHist["DatetimeHist.DatetimeData"]):
    name = "datetime"

    class DatetimeData(NamedTuple):
        date: DateHist.DateData | None = None
        time: TimeHist.TimeData | None = None

    def __init__(self, col: str, meta: ColumnMeta) -> None:
        super().__init__(col, meta)
        self.date = DateHist(col, meta)
        self.time = TimeHist(col, meta)

    def fit(self, data: pd.Series, ref: pd.Series):
        self.date.fit(data, ref)
        self.time.fit(data)

    def process(self, data: pd.Series, ref: pd.Series) -> A:
        date = self.date.process(data, ref)
        time = self.time.process(data)
        return self.DatetimeData(date, time)

    def visualise(self, data: dict[str, DatetimeData]) -> dict[str, Figure] | Figure:
        date_fig = self.date.visualise({n: c.date for n, c in data.items()})
        time_fig = self.time.visualise({n: c.time for n, c in data.items()})

        return {**date_fig, "time": time_fig}


def get_hists():
    return find_subclasses(BaseHist)
