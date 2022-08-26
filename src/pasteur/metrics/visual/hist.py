from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ...metadata import ColumnMeta
from ...utils import find_subclasses

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
    def fit(self, data: pd.Series, ref: pd.Series | None = None):
        pass

    @abstractmethod
    def process(self, data: pd.Series, ref: pd.Series | None = None) -> A:
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
        return self.NumericalData(np.histogram(data, self.bins, density=True)[0])

    def visualise(self, data: dict[str, NumericalData]) -> Figure:
        return _gen_hist(
            self.meta,
            self.col.capitalize(),
            self.bins,
            {n: d.bins for n, d in data.items()},
        )


class CategoricalHist(BaseHist["CategoricalHist.CategoricalData"]):
    name = "categorical"

    class CategoricalData(NamedTuple):
        counts: np.ndarray = None

    def fit(self, data: pd.Series):
        self.cols = data.value_counts().sort_values(ascending=False).index

    def process(self, data: pd.Series) -> CategoricalData:
        return self.CategoricalData(data.value_counts()[self.cols].to_numpy())

    def visualise(self, data: dict[str, CategoricalData]) -> Figure:
        return _gen_bar(
            self.meta,
            self.col.capitalize(),
            self.cols,
            {n: c.counts for n, c in data.items()},
        )


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
        span: np.ndarray | None = None
        weeks: np.ndarray | None = None
        days: np.ndarray | None = None

    def fit(self, data: pd.Series, ref: pd.Series | None = None):
        if "main_param" in self.meta.args:
            self.span = self.meta.args["main_param"].split(".")[0]
        elif "span" in self.meta.args:
            self.span = self.meta.args["span"].split(".")[0]
        else:
            self.span = "year"

        self.max_len = self.meta.args.get("max_len", None)
        self.bin_n = self.meta.args.get("bins", 20)

        if ref is None:
            self.ref = data.min()
        else:
            self.ref = None

        # Find histogram bin edges
        if self.ref is None:
            assert ref is not None
            mask = ~pd.isna(data) & ~pd.isna(ref)
            data = data[mask]
            rf_dt = ref[mask].dt
        else:
            data = data[~pd.isna(data)]
            rf_dt = self.ref

        match self.span:
            case "year":
                segs = data.dt.year - rf_dt.year
            case "week":
                segs = (
                    (data.dt.normalize() - rf_dt.normalize()).dt.days
                    + rf_dt.day_of_week
                ) // 7
            case "day":
                segs = (
                    data.dt.normalize() - rf_dt.normalize()
                ).dt.days + rf_dt.day_of_week
            case other:
                assert False, f"Span {self.span} not supported by DateHist"

        segs = segs.astype("int16")
        if self.max_len is None:
            self.max_len = np.percentile(segs, 90)
        if self.max_len < self.bin_n:
            self.bin_n = int(self.max_len - 1)
        self.bins = np.histogram_bin_edges(
            segs, bins=self.bin_n, range=(0, self.max_len)
        )

    def process(self, data: pd.Series, ref: pd.Series | None = None) -> DateData:
        assert self.ref is not None or ref is not None

        # Based on date transformer
        if self.ref is None:
            assert ref is not None
            mask = ~pd.isna(data) & ~pd.isna(ref)
            data = data[mask]
            rf_dt = ref[mask].dt
        else:
            data = data[~pd.isna(data)]
            rf_dt = self.ref

        match self.span:
            case "year":
                years = data.dt.year - rf_dt.year
                span = np.histogram(years, bins=self.bins, density=True)[0]
            case "week":
                weeks = (
                    (data.dt.normalize() - rf_dt.normalize()).dt.days
                    + rf_dt.day_of_week
                ) // 7
                span = np.histogram(weeks, bins=self.bins, density=True)[0]
            case "day":
                days = (
                    data.dt.normalize() - rf_dt.normalize()
                ).dt.days + rf_dt.day_of_week
                span = np.histogram(days, bins=self.bins, density=True)[0]
            case other:
                assert False, f"Span {self.span} not supported by DateHist"

        weeks = data.dt.week.astype("int16")
        days = data.dt.day_of_week.astype("int16")
        weeks = weeks.value_counts().reindex(range(53), fill_value=0).to_numpy()
        days = days.value_counts().reindex(range(7), fill_value=0).to_numpy()
        return self.DateData(span, weeks, days)

    def _viz_days(self, data: dict[str, DateData]):
        return _gen_bar(
            meta=self.meta,
            title=f"{self.col.capitalize()} Weekday",
            cols=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            counts={n: d.days for n, d in data.items()},
        )

    def _viz_weeks(self, data: dict[str, DateData]):
        bins = np.array(range(54)) - 0.5
        return _gen_hist(
            meta=self.meta,
            title=f"{self.col.capitalize()} Season",
            bins=bins,
            heights={n: d.weeks for n, d in data.items()},
            xticks_x=[2, 15, 28, 41],
            xticks_label=["Winter", "Spring", "Summer", "Autumn"],
        )

    def _viz_binned(self, data: dict[str, DateData]):
        return _gen_hist(
            self.meta,
            f"{self.col.capitalize()} {self.span.capitalize()}s",
            self.bins,
            {n: d.span for n, d in data.items()},
        )

    def visualise(self, data: dict[str, DateData]) -> dict[str, Figure] | Figure:
        s = self.span
        return {
            f"n{s}s": self._viz_binned(data),
            "weeks": self._viz_weeks(data),
            "days": self._viz_days(data),
        }


class TimeHist(BaseHist["TimeHist.TimeData"]):
    name = "time"

    class TimeData(NamedTuple):
        counts: np.ndarray = None

    def fit(self, data: pd.Series):
        if "main_param" in self.meta.args:
            self.span = self.meta.args["main_param"].split(".")[-1]
        elif "span" in self.meta.args:
            self.span = self.meta.args["span"].split(".")[-1]
        else:
            self.span = "halfhour"

    def process(self, data: pd.Series) -> TimeData:
        data = data[~pd.isna(data)]
        hours = data.dt.hour

        if self.span == "hour":
            seg_len = 24
            segments = hours
        else:
            seg_len = 48
            half_hours = data.dt.minute > 29
            segments = 2 * hours + half_hours

        return self.TimeData(
            segments.value_counts().reindex(range(seg_len), fill_value=0).to_numpy()
        )

    def visualise(self, data: dict[str, TimeData]) -> Figure:
        if self.span == "hour":
            seg_len = 24
            mult = 1
        else:
            seg_len = 48
            mult = 2

        bins = np.array(range(seg_len + 1)) - 0.5
        hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
        tick_x = mult * np.array(hours)
        tick_label = [f"{hour:02d}:00" for hour in hours]

        return _gen_hist(
            meta=self.meta,
            title=f"{self.col.capitalize()} Time",
            bins=bins,
            heights={n: d.counts for n, d in data.items()},
            xticks_x=tick_x,
            xticks_label=tick_label,
        )


class DatetimeHist(BaseRefHist["DatetimeHist.DatetimeData"]):
    name = "datetime"

    class DatetimeData(NamedTuple):
        date: DateHist.DateData | None = None
        time: TimeHist.TimeData | None = None

    def __init__(self, col: str, meta: ColumnMeta) -> None:
        super().__init__(col, meta)
        self.date = DateHist(col, meta)
        self.time = TimeHist(col, meta)

    def fit(self, data: pd.Series, ref: pd.Series | None = None):
        self.date.fit(data, ref)
        self.time.fit(data)

    def process(self, data: pd.Series, ref: pd.Series | None = None) -> A:
        date = self.date.process(data, ref)
        time = self.time.process(data)
        return self.DatetimeData(date, time)

    def visualise(self, data: dict[str, DatetimeData]) -> dict[str, Figure] | Figure:
        date_fig = self.date.visualise({n: c.date for n, c in data.items()})
        time_fig = self.time.visualise({n: c.time for n, c in data.items()})

        return {**date_fig, "time": time_fig}


def get_hists():
    return find_subclasses(BaseHist)
