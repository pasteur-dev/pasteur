import logging
from typing import Any

import pandas as pd

from ...metadata import Metadata
from matplotlib.figure import Figure
from .hist import BaseHist, BaseRefHist, get_hists

logger = logging.getLogger(__name__)

VizData = dict[str, Any]


class HistHolder:
    """Class that holds the fitted histograms of a table.

    User feeds in dataset metadata and tables. The class then
    fits a set of histograms to the data based on the metadata.

    Those histograms can then be used to dump intermediary data about
    a dataset split into a picklable dictionary (by calling `process()`).

    A set of those picklable dictionaries can then be fed into `visualise()`
    (name, dict pairs) which will produce visualizations for each column
    that compare the splits with each other."""

    def __init__(self, table: str, meta: Metadata) -> None:
        self.table = table
        self.meta = meta
        self._bootstraped = False

    def _bootstrap(self):
        hist_cls = get_hists()

        self.hists: dict[str, BaseHist | BaseRefHist] = {}
        for name, col in self.meta[self.table].cols.items():
            if col.is_id():
                continue
            if col.type not in hist_cls:
                logger.warning(
                    f"Type {col.type} of {self.table}.{name} does not support visualisation."
                )
                continue

            self.hists[name] = hist_cls[col.type](name, col)

        self._bootstraped = True

    def fit(
        self,
        tables: dict[str, pd.DataFrame],
        ids: pd.DataFrame | None = None,
    ):
        """Fits all classes of type `BaseHist` into the table columns."""
        if not self._bootstraped:
            self._bootstrap()

        for name, hist in self.hists.items():
            ref = self.meta[self.table, name].ref
            if isinstance(hist, BaseRefHist) and ref is not None:
                if ref.table is not None:
                    ref_col = ids.join(tables[ref.table][ref.col], on=ref.table)[
                        ref.col
                    ]
                else:
                    ref_col = tables[self.table][ref.col]
                hist.fit(tables[self.table][name], ref_col)
            else:
                hist.fit(tables[self.table][name])

    def process(
        self, tables: dict[str, pd.DataFrame], ids: dict[str, pd.DataFrame]
    ) -> VizData:
        """Captures metadata about the provided dataset split, that can be used
        to create visualisations about it by calling `visualize()`"""
        data = {}
        for name, hist in self.hists.items():
            ref = self.meta[self.table, name].ref
            if isinstance(hist, BaseRefHist) and ref is not None:
                if ref.table is not None:
                    ref_col = ids.join(tables[ref.table][ref.col], on=ref.table)[
                        ref.col
                    ]
                else:
                    ref_col = tables[self.table][ref.col]
                data[name] = hist.process(tables[self.table][name], ref_col)
            else:
                data[name] = hist.process(tables[self.table][name])

        return data

    def visualise(
        self, data: dict[str, VizData]
    ) -> dict[str, dict[str, Figure] | Figure]:
        """Takes in a dictionary of (split_name, metadata) and returns visualizations
        that compare the provided runs. `split_name` is used for the legends."""
        viz = {}
        cols = next(iter(data.values())).keys()
        for name in cols:
            assert name in self.hists
            hist_data = {split: data[split][name] for split in data.keys()}

            v = self.hists[name].visualise(hist_data)
            if v is not None:
                viz[name] = v

        return viz


def _separate_ids_tables(data: dict[str, pd.DataFrame]):
    """Separates tables and ids from incoming kedro data. ID Dataframes should
    start with `ids.` and table dataframes with `tbl.`"""
    ids = {}
    tables = {}

    for in_name, d in data.items():
        split, name = in_name.split(".")
        match split:
            case "ids":
                ids[name] = d
            case "tbl":
                tables[name] = d

    return tables, ids


def create_fitted_hist_holder(
    name: str, meta: Metadata, ids: pd.DataFrame, **tables: pd.DataFrame
):
    holder = HistHolder(name, meta)
    holder.fit(tables, ids)
    return holder


def project_hists_for_view(
    holder: HistHolder, ids: pd.DataFrame, **tables: pd.DataFrame
):
    return holder.process(tables, ids)
