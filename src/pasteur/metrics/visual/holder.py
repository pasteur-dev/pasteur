import logging
from typing import Any

import pandas as pd

from ...transform.table import TableTransformer

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

            self.hists[name] = hist_cls[col.type](col)

        self._bootstraped = True

    def fit(self, tables: dict[str, pd.DataFrame]):
        """Fits all classes of type `BaseHist` into the table columns."""
        if not self._bootstraped:
            self._bootstrap()

        for name, hist in self.hists.items():
            if isinstance(hist, BaseRefHist):
                assert False, "Ref Hists not supported yet"

            hist.fit(tables[self.table][name])

    def process(self, tables: dict[str, pd.DataFrame]) -> VizData:
        """Captures metadata about the provided dataset split, that can be used
        to create visualisations about it by calling `visualize()`"""
        data = {}
        for name, hist in self.hists.items():
            if isinstance(hist, BaseRefHist):
                assert False, "Ref Hists not supported yet"

            data[name] = hist.process(tables[self.table][name])

        return data

    def visualise(self, data: dict[str, VizData]) -> dict[str, list[Figure] | Figure]:
        """Takes in a dictionary of (split_name, metadata) and returns visualizations
        that compare the provided runs. `split_name` is used for the legends."""
        viz = {}
        cols = next(iter(data.values())).keys()
        for name in cols:
            assert name in self.hists
            hist_data = {split: data[split][name] for split in data.keys()}

            viz[name] = self.hists[name].visualise(hist_data)

        return viz


def create_fitted_hist_holder(table: str, meta: Metadata, tables: pd.DataFrame):
    holder = HistHolder(table, meta)
    holder.fit(tables)
    return holder


def create_fitted_hist_holder_closure(table: str):
    def closure(transformer: TableTransformer, **tables):
        return create_fitted_hist_holder(table, transformer.meta, tables)

    closure.__name__ = f"fit_hists_for_{table}"
    return closure


def project_hists_for_view(holder: HistHolder, **tables: pd.DataFrame):
    return holder.process(tables)
