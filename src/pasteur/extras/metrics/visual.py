from typing import TYPE_CHECKING, Any, NamedTuple, Sequence, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pasteur.attribute import SeqValue
from pasteur.metric import AbstractColumnMetric, SeqColumnData, Summaries

from ...metric import (
    AbstractColumnMetric,
    ColumnMetric,
    RefColumnData,
    RefColumnMetric,
    SeqColumnMetric,
    Summaries,
    name_style_fn,
    name_style_title,
)
from ...utils import list_unique
from ...utils.mlflow import load_matplotlib_style, mlflow_log_hists

if TYPE_CHECKING:
    from matplotlib.figure import Figure

A = TypeVar("A")

def _percent_formatter(x, pos):
    return f"{100*x:.1f}%"


def _gen_hist(
    y_log: bool,
    title: str,
    bins: np.ndarray | Sequence[float],
    heights: dict[str, np.ndarray],
    xticks_x=None,
    xticks_label=None,
):
    fig, ax = plt.subplots()
    x = np.array(bins)[:-1]
    w = (x[1] - x[0]) / len(heights)

    for i, (name, h) in enumerate(heights.items()):
        ax.bar(x + w * i, h / h.sum(), width=w, label=name, log=y_log)

    ax.legend()
    ax.set_title(title)
    ax.yaxis.set_major_formatter(_percent_formatter)

    if xticks_x is not None:
        ax.set_xticks(xticks_x, xticks_label)

    plt.tight_layout()
    return fig


def _gen_bar(y_log: bool, title: str, cols: list[str], counts: dict[str, np.ndarray]):
    fig, ax = plt.subplots()

    x = np.array(range(len(cols)))
    w = 0.9 / len(counts)

    for i, (name, c) in enumerate(counts.items()):
        h = c / c.sum() if c.sum() > 0 else c
        ax.bar(
            x - 0.45 + w * i,
            h,
            width=w,
            align="edge",
            label=name,
            log=y_log,
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

    def __init__(
        self,
        bins: Sequence[float] | int = 10,
        min: float | None = None,
        max: float | None = None,
        y_log: bool = False,
        _from_factory: bool = False,
        **_,
    ) -> None:
        super().__init__(_from_factory=_from_factory)

        self.y_log = y_log
        if isinstance(bins, Sequence):
            self.manual = True
            self.bins_arg = tuple(bins)
            self.bin_n = len(bins) - 1
            self.min_arg = bins[0]
            self.max_arg = bins[-1]

            assert (
                not min and not max
            ), "Either min,max or specific buckets can be provided."
        else:
            self.manual = False
            self.bins_arg = None
            self.bin_n = bins
            self.min_arg = min
            self.max_arg = max

    def fit(self, table: str, col: str, data: pd.Series):
        self.table = table
        self.col = col

        # Get maximums
        self.min = self.min_arg if self.min_arg is not None else data.min()
        self.max = self.max_arg if self.max_arg is not None else data.max()

        # In the case the column is NA, x_min and x_max will be NA
        # Disable visualiser
        self.disabled = (
            self.max is None
            or self.min is None
            or np.isnan(self.max)
            or np.isnan(self.min)
        )

        if self.bins_arg is None:
            self.bins = np.linspace(self.min, self.max, self.bin_n + 1)
        else:
            self.bins = self.bins_arg

        if self.disabled:
            return

    def reduce(self, other: "NumericalHist"):
        if self.manual:
            return
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.bins = np.linspace(self.min, self.max, self.bin_n + 1)

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
            self.y_log,
            self.col.capitalize(),
            self.bins,
            splits,
        )

        mlflow_log_hists(self.table, self.col, v)


class CategoricalHist(ColumnMetric[Summaries[np.ndarray], Summaries[np.ndarray]]):
    name = "categorical"

    def __init__(self, y_log: bool = False, _from_factory: bool = False, **_) -> None:
        super().__init__(_from_factory=_from_factory)
        self.y_log = y_log

    def fit(self, table: str, col: str, data: pd.Series):
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
            self.y_log,
            self.col.capitalize(),
            self.cols,
            splits,
        )
        mlflow_log_hists(self.table, self.col, v)


class OrdinalHist(CategoricalHist):
    name = "ordinal"

    def fit(self, table: str, col: str, data: pd.Series):
        super().fit(table, col, data)
        self.cols = pd.Index(np.sort(data.unique()))


class FixedHist(ColumnMetric[Any, Any]):
    """Fixed values can not be visualised. Removes warning."""

    name = "fixed"

    def fit(self, table: str, col: str, data: pd.Series): ...

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

    def __init__(
        self,
        span: str = "year",
        y_log: bool = False,
        nullable: bool = False,
        bins: int = 20,
        max_len: int | None = None,
        **_,
    ) -> None:
        self.span = span.split(".")[0]
        self.y_log = y_log
        self.nullable = nullable
        self.bin_n = bins
        self.max_len_arg = max_len

    def fit(self, table: str, col: str | tuple[str, ...], data: RefColumnData):
        ddata = cast(pd.Series, data["data"])
        dref = cast(pd.Series | None, data.get("ref", None))

        self.table = table
        self.col = col

        self.weeks53 = self.span == "year53"
        if self.weeks53:
            self.span = "year"

        if dref is None:
            self.ref = ddata.min()
        else:
            self.ref = None

        # Find histogram bin edges
        if self.ref is None:
            assert dref is not None
            mask = ~pd.isna(ddata) & ~pd.isna(dref)
            ddata = ddata[mask]
            rf_dt = dref[mask].dt
        else:
            ddata = ddata[~pd.isna(ddata)]
            rf_dt = self.ref

        match self.span:
            case "year":
                segs = ddata.dt.year - rf_dt.year
            case "week":
                segs = (
                    (ddata.dt.normalize() - rf_dt.normalize()).dt.days
                    + rf_dt.day_of_week
                ) // 7
            case "day":
                segs = (
                    ddata.dt.normalize() - rf_dt.normalize()
                ).dt.days + rf_dt.day_of_week
            case _:
                assert False, f"Span {self.span} not supported by DateHist"

        segs = segs.astype("int16")
        if self.max_len_arg is None:
            self.max_len = float(np.percentile(segs, 90))
        else:
            self.max_len = self.max_len_arg

        if self.max_len < self.bin_n:
            self.bin_n = int(self.max_len - 1)

        self.bins = np.linspace(0, self.max_len, self.bin_n + 1)

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
            y_log=self.y_log,
            title=name_style_title(self.col, "Weekday"),
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
            y_log=self.y_log,
            title=name_style_title(self.col, "Season"),
            bins=bins,
            heights={n: d.weeks for n, d in data.items() if d.weeks is not None},
            xticks_x=[2, 15, 28, 41],
            xticks_label=["Winter", "Spring", "Summer", "Autumn"],
        )

    def _viz_binned(self, data: dict[str, DateData]):
        return _gen_hist(
            self.y_log,
            name_style_title(self.col, f"{self.span.capitalize()}s"),
            self.bins,
            {n: d.span for n, d in data.items() if d.span is not None},
        )

    def _viz_na(self, data: dict[str, DateData]):
        return _gen_bar(
            self.y_log,
            name_style_title(self.col, "NA"),
            ["Val", "NA"],
            {n: d.na for n, d in data.items() if d.na is not None},
        )

    def _visualise(self, data: dict[str, Summaries[DateData]]) -> dict[str, "Figure"]:
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
        mlflow_log_hists(self.table, name_style_fn(self.col), v)


class TimeHist(ColumnMetric[Summaries[np.ndarray], Summaries[np.ndarray]]):
    name = "time"

    def __init__(
        self,
        span: str = "halfhour",
        y_log: bool = False,
        _from_factory: bool = False,
        **_,
    ) -> None:
        self.span = span.split(".")[-1]
        self.y_log = y_log
        super().__init__(_from_factory=_from_factory)

    def fit(self, table: str, col: str, data: pd.Series):
        self.table = table
        self.col = col

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

    def _visualise(self, data: dict[str, Summaries[np.ndarray]]) -> "Figure":
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
            y_log=self.y_log,
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

    def fit(self, table: str, col: str, data: RefColumnData):
        self.table = table
        self.col = col
        self.date.fit(table=table, col=col, data=data)
        self.time.fit(table=table, col=col, data=cast(pd.Series, data["data"]))

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

    def visualise(
        self, data: dict[str, tuple[Summaries[DateData], Summaries[ndarray]]]
    ):
        load_matplotlib_style()
        date_fig = self.date._visualise({n: c[0] for n, c in data.items()})
        time_fig = self.time._visualise({n: c[1] for n, c in data.items()})

        mlflow_log_hists(self.table, self.col, {**date_fig, "time": time_fig})


class SeqHist(
    SeqColumnMetric[
        Summaries[np.ndarray],
        Summaries[np.ndarray],
    ]
):
    name = "seq"

    def __init__(
        self, y_log=False, max_len: int | None = None, _from_factory: bool = False, **_
    ) -> None:
        super().__init__(_from_factory=_from_factory, **_)
        self.y_log = y_log
        self.max_len_arg = max_len

    def fit(
        self,
        table: str,
        col: str | tuple[str, ...],
        seq_val: SeqValue | None,
        data: SeqColumnData,
    ):
        self.max_len = self.max_len_arg or int(data["seq"].max() + 1)
        self.table = table
        self.col = col

        assert seq_val is not None
        self.parent = seq_val.table

    def reduce(self, other: "SeqHist"):
        self.max_len = max(self.max_len, other.max_len)

    def _process(self, data: SeqColumnData):
        return (
            data["seq"]
            .groupby(data["ids"][self.parent])
            .max()
            .value_counts()
            .reindex(range(self.max_len), fill_value=0)
            .to_numpy()
        )

    def preprocess(self, wrk: SeqColumnData, ref: SeqColumnData) -> Summaries[ndarray]:
        return Summaries(self._process(wrk), self._process(ref))

    def combine(self, summaries: list[Summaries[ndarray]]) -> Summaries[ndarray]:
        return Summaries(
            wrk=np.sum([s.wrk for s in summaries], axis=0),
            ref=np.sum([s.ref for s in summaries], axis=0),
            syn=np.sum([s.syn for s in summaries if s.syn is not None], axis=0),
        )

    def process(
        self,
        wrk: SeqColumnData,
        ref: SeqColumnData,
        syn: SeqColumnData,
        pre: Summaries[ndarray],
    ) -> Summaries[ndarray]:
        return pre.replace(syn=self._process(syn))

    def visualise(self, data: dict[str, Summaries[ndarray]]):
        load_matplotlib_style()

        keys = list(data.keys())
        splits = {"wrk": data[keys[0]].wrk, "ref": data[keys[0]].ref}
        for name, split in data.items():
            splits[name] = split.syn

        f = _gen_hist(
            self.y_log,
            f"N-1 with parent '{self.parent}'",
            np.arange(self.max_len + 1) - 0.5,
            splits,
        )

        mlflow_log_hists(self.table, f"_n_per_{self.parent}", f)
