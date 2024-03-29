from typing import Any, NamedTuple, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from ...metadata import ColumnMeta, Metadata
from ...metric import (
    AbstractColumnMetric,
    ColumnMetric,
    RefColumnData,
    RefColumnMetric,
    Summaries,
)
from ...utils import list_unique
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
        h = c / c.sum() if c.sum() > 0 else c
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


class NumericalHist(ColumnMetric[Summaries[np.ndarray], Summaries[np.ndarray]]):
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
        elif "min" in args:
            x_min = args["min"]
        else:
            x_min = data.min()

        if metrics.x_max is not None:
            x_max = metrics.x_max
        elif "max" in args:
            x_max = args["max"]
        else:
            x_max = data.max()

        # In the case the column is NA, x_min and x_max will be NA
        # Disable visualiser
        self.disabled = (
            x_max is None or x_min is None or np.isnan(x_max) or np.isnan(x_min)
        )
        if self.disabled:
            return

        main_param = args.get("main_param", None)
        if main_param and (isinstance(main_param, int)):
            self.bin_n = main_param
        else:
            self.bin_n = args.get("bins", 20)

        self.x_min = x_min
        self.x_max = x_max
        self.bins = np.histogram_bin_edges(data, bins=self.bin_n, range=(x_min, x_max))

    def reduce(self, other: "NumericalHist"):
        self.x_min = min(self.x_min, other.x_min)
        self.x_max = max(self.x_max, other.x_max)
        self.bins = np.linspace(self.x_min, self.x_max, self.bin_n + 1)

    def _process(self, data: pd.Series):
        if self.disabled:
            return np.array([])
        return np.histogram(data.astype(np.float32), self.bins, density=True)[0]

    def preprocess(self, wrk: Series, ref: Series):
        return Summaries(self._process(wrk), self._process(ref))

    def process(
        self, wrk: Series, ref: Series, syn: Series, pre: Summaries[np.ndarray]
    ):
        return pre.replace(syn=self._process(syn))

    def combine(self, summaries: list[Summaries[ndarray]]) -> Summaries[ndarray]:
        return Summaries(
            wrk=np.sum([s.wrk for s in summaries], axis=0),
            ref=np.sum([s.ref for s in summaries], axis=0),
            syn=np.sum([s.syn for s in summaries if s.syn is not None], axis=0),
        )

    def visualise(self, data: dict[str, Summaries[np.ndarray]]):
        if self.disabled:
            return

        keys = list(data.keys())
        splits = {"wrk": data[keys[0]].wrk, "ref": data[keys[0]].ref}
        for name, split in data.items():
            assert split.syn is not None, f"Received null syn split for split {name}."
            splits[name] = split.syn

        load_matplotlib_style()
        v = _gen_hist(
            self.meta,
            self.col.capitalize(),
            self.bins,
            splits,
        )

        mlflow_log_hists(self.table, self.col, v)


class CategoricalHist(ColumnMetric[Summaries[np.ndarray], Summaries[np.ndarray]]):
    name = "categorical"

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        self.meta = meta
        self.table = table
        self.col = col
        self.cols = list(data.value_counts().sort_values(ascending=False).index)

    def reduce(self, other: "CategoricalHist"):
        self.cols = list_unique(self.cols, other.cols)

    def _process(self, data: pd.Series):
        return data.value_counts().reindex(self.cols, fill_value=0).to_numpy()

    def _combine(self, summaries: list[np.ndarray]) -> np.ndarray:
        return np.sum(summaries, axis=0)

    def preprocess(self, wrk: Series, ref: Series):
        return Summaries(self._process(wrk), self._process(ref))

    def process(
        self, wrk: Series, ref: Series, syn: Series, pre: Summaries[np.ndarray]
    ):
        return pre.replace(syn=self._process(syn))

    def combine(self, summaries: list[Summaries[ndarray]]) -> Summaries[ndarray]:
        return Summaries(
            wrk=self._combine([s.wrk for s in summaries]),
            ref=self._combine([s.ref for s in summaries]),
            syn=self._combine([s.syn for s in summaries if s.syn is not None]),
        )

    def visualise(self, data: dict[str, Summaries[np.ndarray]]):
        keys = list(data.keys())
        splits = {"wrk": data[keys[0]].wrk, "ref": data[keys[0]].ref}
        for name, split in data.items():
            assert split.syn is not None, f"Received null syn split for split {name}."
            splits[name] = split.syn

        load_matplotlib_style()
        v = _gen_bar(
            self.meta,
            self.col.capitalize(),
            self.cols,
            splits,
        )
        mlflow_log_hists(self.table, self.col, v)


class OrdinalHist(CategoricalHist):
    name = "ordinal"

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        super().fit(table, col, meta, data)
        self.cols = pd.Index(np.sort(data.unique()))


class FixedHist(ColumnMetric[Any, Any]):
    """Fixed values can not be visualised. Removes warning."""

    name = "fixed"

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        ...

    def reduce(self, other: AbstractColumnMetric):
        pass

    def process(
        self,
        wrk: Series | DataFrame,
        ref: Series | DataFrame,
        syn: Series | DataFrame,
        pre: Any,
    ) -> Any:
        return []

    def combine(self, summaries: list[Any]) -> Any:
        return []


class DateData(NamedTuple):
    span: np.ndarray | None = None
    weeks: np.ndarray | None = None
    days: np.ndarray | None = None
    na: np.ndarray | None = None


class DateHist(RefColumnMetric[Summaries[DateData], Summaries[DateData]]):
    name = "date"

    def fit(self, table: str, col: str | tuple[str], meta: ColumnMeta, data: RefColumnData):
        ref = data['ref']
        data = data['data']
        self.table = table
        self.col = col

        self.meta = meta
        if "main_param" in meta.args:
            self.span = meta.args["main_param"].split(".")[0]
        elif "span" in meta.args:
            self.span = meta.args["span"].split(".")[0]
        else:
            self.span = "year"

        self.weeks53 = self.span == "year53"
        if self.weeks53:
            self.span = "year"

        self.max_len = meta.args.get("max_len", None)
        self.bin_n = meta.args.get("bins", 20)

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
            case _:
                assert False, f"Span {self.span} not supported by DateHist"

        segs = segs.astype("int16")
        if self.max_len is None:
            self.max_len = float(np.percentile(segs, 90))
        if self.max_len < self.bin_n:
            self.bin_n = int(self.max_len - 1)

        self.bins = np.histogram_bin_edges(
            segs, bins=self.bin_n, range=(0, self.max_len)
        )
        self.nullable = meta.args.get("nullable", False)

    def reduce(self, other: "DateHist"):
        self.max_len = max(self.max_len, other.max_len)  # type: ignore
        self.bin_n = max(self.bin_n, other.bin_n)

        if self.max_len < self.bin_n:
            self.bin_n = int(self.max_len - 1)

        self.bins = np.linspace(0, self.max_len, self.bin_n + 1)

    def _process(self, data: pd.Series, ref: pd.Series | None = None) -> DateData:
        assert self.ref is not None or ref is not None

        # Based on date transformer
        if self.ref is None:
            assert ref is not None
            mask = ~pd.isna(data) & ~pd.isna(ref)
            data = data[mask]
            rf_dt = ref[mask].dt
        else:
            mask = ~pd.isna(data)
            data = data[mask]
            rf_dt = self.ref

        iso = data.dt.isocalendar()
        iso_rf = rf_dt.isocalendar()

        if self.ref is not None:
            rf_year = iso_rf.year
            rf_day = iso_rf.weekday
        else:
            rf_year = iso_rf["year"]
            rf_day = iso_rf["day"]

        weeks = iso["week"].astype("int16") - 1
        days = iso["day"].astype("int16") - 1

        # Push week 53 to next year
        if not self.weeks53:
            m = weeks == 52
            weeks[m] = 0

        match self.span:
            case "year":
                span = iso["year"] - rf_year
                if not self.weeks53:
                    span[m] += 1  # type: ignore
                span = np.histogram(span, bins=self.bins, density=True)[0]
            case "week":
                span = ((data.dt.normalize() - rf_dt.normalize()).dt.days + rf_day) // 7
                span = np.histogram(span, bins=self.bins, density=True)[0]
            case "day":
                span = (data.dt.normalize() - rf_dt.normalize()).dt.days + rf_day
                span = np.histogram(span, bins=self.bins, density=True)[0]
            case _:
                assert False, f"Span {self.span} not supported by DateHist"

        weeks = (
            weeks.value_counts()
            .reindex(range(53 if self.weeks53 else 52), fill_value=0)
            .to_numpy()
        )
        days = days.value_counts().reindex(range(7), fill_value=0).to_numpy()

        na = None
        if self.nullable:
            non_na_rate = np.sum(mask) / len(mask)  # type: ignore
            na = np.array([non_na_rate, 1 - non_na_rate])

        return DateData(span, weeks, days, na)

    def _combine(self, summaries: list[DateData]) -> DateData:
        return DateData(
            span=np.sum([s.span for s in summaries if s.span is not None], axis=0),
            weeks=np.sum([s.weeks for s in summaries if s.weeks is not None], axis=0),
            days=np.sum([s.days for s in summaries if s.days is not None], axis=0),
            na=np.sum([s.na for s in summaries if s.na is not None], axis=0),
        )

    def preprocess(
        self, wrk: RefColumnData, ref: RefColumnData
    ) -> Summaries[DateData] | None:
        return Summaries(
            self._process(wrk["data"], wrk["ref"]),  # type: ignore
            self._process(ref["data"], ref["ref"]),  # type: ignore
        )

    def process(
        self,
        wrk: RefColumnData,
        ref: RefColumnData,
        syn: RefColumnData,
        pre: Summaries[DateData],
    ) -> Summaries[DateData]:
        return pre.replace(syn=self._process(syn["data"], syn["ref"]))  # type: ignore

    def combine(self, summaries: list[Summaries[DateData]]) -> Summaries[DateData]:
        return Summaries(
            wrk=self._combine([s.wrk for s in summaries]),
            ref=self._combine([s.ref for s in summaries]),
            syn=self._combine([s.syn for s in summaries if s.syn is not None]),
        )

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
            counts={n: d.days for n, d in data.items() if d.days is not None},
        )

    def _viz_weeks(self, data: dict[str, DateData]):
        bins = np.array(range(54 if self.weeks53 else 53)) - 0.5
        return _gen_hist(
            meta=self.meta,
            title=f"{self.col.capitalize()} Season",
            bins=bins,
            heights={n: d.weeks for n, d in data.items() if d.weeks is not None},
            xticks_x=[2, 15, 28, 41],
            xticks_label=["Winter", "Spring", "Summer", "Autumn"],
        )

    def _viz_binned(self, data: dict[str, DateData]):
        return _gen_hist(
            self.meta,
            f"{self.col.capitalize()} {self.span.capitalize()}s",
            self.bins,
            {n: d.span for n, d in data.items() if d.span is not None},
        )

    def _viz_na(self, data: dict[str, DateData]):
        return _gen_bar(
            self.meta,
            f"{self.col.capitalize()} NA",
            ["Val", "NA"],
            {n: d.na for n, d in data.items() if d.na is not None},
        )

    def _visualise(self, data: dict[str, Summaries[DateData]]) -> dict[str, Figure]:
        keys = list(data.keys())
        splits = {"wrk": data[keys[0]].wrk, "ref": data[keys[0]].ref}
        for name, split in data.items():
            assert split.syn is not None, f"Received null syn split for split {name}."
            splits[name] = split.syn

        s = self.span
        charts = {
            f"n{s}s": self._viz_binned(splits),
            "weeks": self._viz_weeks(splits),
            "days": self._viz_days(splits),
        }
        if self.nullable:
            charts["na"] = self._viz_na(splits)
        return charts

    def visualise(self, data: dict[str, Summaries[DateData]]):
        load_matplotlib_style()
        v = self._visualise(data)
        mlflow_log_hists(self.table, self.col, v)


class TimeHist(ColumnMetric[Summaries[np.ndarray], Summaries[np.ndarray]]):
    name = "time"

    def fit(self, table: str, col: str, meta: ColumnMeta, data: pd.Series):
        self.meta = meta
        self.table = table
        self.col = col

        if "main_param" in meta.args:
            self.span = meta.args["main_param"].split(".")[-1]
        elif "span" in meta.args:
            self.span = meta.args["span"].split(".")[-1]
        else:
            self.span = "halfhour"

    def reduce(self, other: AbstractColumnMetric):
        pass

    def _process(self, data: pd.Series):
        data = data[~pd.isna(data)]
        hours = data.dt.hour

        if self.span == "hour":
            seg_len = 24
            segments = hours
        else:
            seg_len = 48
            half_hours = data.dt.minute > 29
            segments = 2 * hours + half_hours

        return segments.value_counts().reindex(range(seg_len), fill_value=0).to_numpy()

    def _combine(self, summaries: list[np.ndarray]) -> np.ndarray:
        return np.sum(summaries, axis=0)

    def preprocess(self, wrk: Series, ref: Series):
        return Summaries(self._process(wrk), self._process(ref))

    def process(
        self, wrk: Series, ref: Series, syn: Series, pre: Summaries[np.ndarray]
    ):
        return pre.replace(syn=self._process(syn))

    def combine(self, summaries: list[Summaries[ndarray]]) -> Summaries[ndarray]:
        return Summaries(
            wrk=np.sum([s.wrk for s in summaries], axis=0),
            ref=np.sum([s.ref for s in summaries], axis=0),
            syn=np.sum([s.syn for s in summaries if s.syn is not None], axis=0),
        )

    def _visualise(self, data: dict[str, Summaries[np.ndarray]]) -> Figure:
        keys = list(data.keys())
        splits = {"wrk": data[keys[0]].wrk, "ref": data[keys[0]].ref}
        for name, split in data.items():
            assert split.syn is not None, f"Received null syn split for split {name}."
            splits[name] = split.syn

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
            heights=splits,
            xticks_x=tick_x,
            xticks_label=tick_label,
        )

    def visualise(self, data: dict[str, Summaries[np.ndarray]]):
        load_matplotlib_style()
        v = self._visualise(data)
        mlflow_log_hists(self.table, self.col, v)


class DatetimeData(NamedTuple):
    date: DateData
    time: np.ndarray


class DatetimeHist(
    RefColumnMetric[
        tuple[Summaries[DateData], Summaries[ndarray]],
        tuple[Summaries[DateData], Summaries[ndarray]],
    ]
):
    name = "datetime"

    def __init__(self, *args, _from_factory: bool = False, **kwargs) -> None:
        super().__init__(*args, _from_factory=_from_factory, **kwargs)
        self.date = DateHist(*args, _from_factory=_from_factory, **kwargs)
        self.time = TimeHist(*args, _from_factory=_from_factory, **kwargs)

    def fit(
        self, table: str, col: str, meta: ColumnMeta, data: RefColumnData
    ):
        self.table = table
        self.col = col
        self.date.fit(table=table, col=col, meta=meta, data=data)
        self.time.fit(table=table, col=col, meta=meta, data=cast(pd.Series, data['data']))

    def preprocess(
        self, wrk: RefColumnData, ref: RefColumnData
    ) -> tuple[Summaries[DateData], Summaries[ndarray]] | None:
        return (
            self.date.preprocess(wrk, ref),
            self.time.preprocess(wrk["data"], ref["data"]),  # type: ignore
        )

    def process(
        self,
        wrk: RefColumnData,
        ref: RefColumnData,
        syn: RefColumnData,
        pre: tuple[Summaries[DateData], Summaries[ndarray]],
    ) -> tuple[Summaries[DateData], Summaries[ndarray]]:
        return (
            self.date.process(wrk, ref, syn, pre[0]),
            self.time.process(wrk["data"], ref["data"], syn["data"], pre[1]),  # type: ignore
        )
    
    def combine(
        self, summaries: list[tuple[Summaries[DateData], Summaries[ndarray]]]
    ) -> tuple[Summaries[DateData], Summaries[ndarray]]:
        return (
            self.date.combine([s[0] for s in summaries]),
            self.time.combine([s[1] for s in summaries]),
        )

    def visualise(self, data: dict[str, tuple[Summaries[DateData], Summaries[ndarray]]]):
        load_matplotlib_style()
        date_fig = self.date._visualise({n: c[0] for n, c in data.items()})
        time_fig = self.time._visualise({n: c[1] for n, c in data.items()})

        mlflow_log_hists(self.table, self.col, {**date_fig, "time": time_fig})
