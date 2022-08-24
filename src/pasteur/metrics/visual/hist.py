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
        return self.NumericalData(np.histogram(data, self.bins, density=True)[0])

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
        dt = data.value_counts()
        dt /= dt.sum()
        return self.CategoricalData(dt)

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
        years: pd.Series | None = None
        weeks: pd.Series | None = None
        days: pd.Series | None = None

    def fit(self, data: pd.Series, ref: pd.Series):
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
        mask = ~pd.isna(data) & ~pd.isna(ref)
        data = data[mask]
        ref = ref[mask]
        rf_dt = self.ref if self.ref is not None else ref.dt

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

    def process(self, data: pd.Series, ref: pd.Series) -> DateData:
        assert self.ref is not None or ref is not None

        # Based on date transformer
        mask = ~pd.isna(data) & ~pd.isna(ref)
        data = data[mask]
        ref = ref[mask]
        rf_dt = self.ref if self.ref is not None else ref.dt

        match self.span:
            case "year":
                years = data.dt.year - rf_dt.year
                years = np.histogram(years, bins=self.bins, density=True)[0]

                weeks = data.dt.week.astype("int16")
                days = data.dt.day_of_week.astype("int16")

                weeks = weeks.value_counts()
                weeks /= weeks.sum()
                days = days.value_counts()
                days /= days.sum()
            case "week":
                years = None

                weeks = (
                    (data.dt.normalize() - rf_dt.normalize()).dt.days
                    + rf_dt.day_of_week
                ) // 7
                weeks = np.histogram(weeks, bins=self.bins, density=True)[0]

                days = data.dt.day_of_week.astype("int16")
                days = days.value_counts()
                days /= days.sum()
            case "day":
                years = None
                weeks = None
                days = (
                    data.dt.normalize() - rf_dt.normalize()
                ).dt.days + rf_dt.day_of_week
                days = np.histogram(days, bins=self.bins, density=True)[0]
            case other:
                assert False, f"Span {self.span} not supported by DateHist"

        return self.DateData(years, weeks, days)

    def _viz_days(self, data: dict[str, DateData]):
        fig, ax = plt.subplots()

        x = np.array(range(7))
        w = 0.9 / len(data)
        ofs = 0.9 / 2

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            segments = d.days.reindex(range(7), fill_value=0).to_numpy()
            ax.bar(
                x - ofs + w * i,
                segments,
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        days_x = [0, 1, 2, 3, 4, 5, 6]
        days_label = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        plt.xticks(days_x, days_label)

        ax.legend()
        ax.set_title(f"{self.col.capitalize()} Weekday")
        plt.tight_layout()
        return fig

    def _viz_weeks(self, data: dict[str, DateData]):
        fig, ax = plt.subplots()

        x = np.array(range(53))
        w = 1 / len(data)
        ofs = 1 / 2

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            segments = d.weeks.reindex(range(53), fill_value=0).to_numpy()
            ax.bar(
                x - ofs + w * i,
                segments,
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        tick_x = [2, 15, 28, 41]
        tick_label = ["Winter", "Spring", "Summer", "Autumn"]
        plt.xticks(tick_x, tick_label)

        ax.legend()
        ax.set_title(self.col.capitalize())
        ax.set_title(f"{self.col.capitalize()} Season")
        plt.tight_layout()
        return fig

    def _viz_binned(self, data: dict[str, DateData], field: str, title: str):
        fig, ax = plt.subplots()

        x = self.bins[:-1]
        w = (x[1] - x[0]) / len(data)

        is_log = self.meta.metrics.y_log == True
        for i, (name, d) in enumerate(data.items()):
            ax.bar(
                x + w * i,
                getattr(d, field),
                width=w,
                align="edge",
                label=name,
                log=is_log,
            )

        ax.legend()
        ax.set_title(f"{self.col.capitalize()} {title}")
        plt.tight_layout()
        return fig

    def visualise(self, data: dict[str, DateData]) -> dict[str, Figure] | Figure:
        match self.span:
            case "year":
                return {
                    "years": self._viz_binned(data, "years", "Years"),
                    "weeks": self._viz_weeks(data),
                    "days": self._viz_days(data),
                }
            case "week":
                return {
                    "weeks": self._viz_binned(data, "weeks", "Weeks"),
                    "days": self._viz_days(data),
                }
            case "day":
                return {"days": self._viz_binned(data, "days", "Days")}
            case other:
                assert False, f"Span {self.span} not supported by DateHist"


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
        data = data[~pd.isna(data)]
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
