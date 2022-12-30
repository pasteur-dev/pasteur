from __future__ import annotations

from typing import TYPE_CHECKING

from .module import ModuleClass, ModuleFactory
from .table import TransformHolder
from .utils import LazyFrame
from functools import partial, wraps


if TYPE_CHECKING:
    from .attribute import Attributes
    from .metadata import Metadata
    import pandas as pd

import logging

logger = logging.getLogger(__name__)


def make_deterministic(obj_func, /, *, noise_kw: str | None = None):
    """Takes an object function (with self), and if the object has a seed attribute
    it fixes the np.random.seed attribute to it and prints a random number at the end.

    If the algorithm sampled the same amount of numbers at the same order, then the
    numbers should be the same."""

    if isinstance(obj_func, str):
        return partial(make_deterministic, noise_kw=obj_func)

    import random

    import numpy as np

    @wraps(obj_func)
    def wrapped(self, *args, **kwargs):
        if self.seed is not None:
            seed = self.seed

            if noise_kw is not None:
                seed += kwargs[noise_kw]

            np.random.seed(seed)
            random.seed(seed)

        a = obj_func(self, *args, **kwargs)

        if self.seed is not None:
            logger.info(
                f"Deterministic check: random number after "
                + f"{f'{type(self).__name__}.{obj_func.__name__}':>22s}(): "
                + f"<np.random> {np.random.random():7.5f} <random> {random.random():7.5f}"
            )
        return a

    return wrapped


class SynthFactory(ModuleFactory["Synth"]):
    def __init__(self, cls: type[Synth], *args, name: str | None = None, **_) -> None:
        super().__init__(cls, *args, name=name, **_)
        self.type = cls.type
        self.tabular = cls.tabular
        self.multimodal = cls.multimodal
        self.timeseries = cls.timeseries
        self.gpu = cls.gpu
        self.parallel = cls.parallel


class Synth(ModuleClass):
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    partitions = 1
    gpu = False
    parallel = False

    _factory = SynthFactory

    def preprocess(
        self,
        attrs: dict[str, Attributes],
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        """Runs any preprocessing required, such as domain reduction."""
        raise NotImplementedError()

    def bake(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        """Bakes the model based on the data provided (such as creating and
        modeling a bayesian network on the data).

        Attributes provide context about the data columns, including hierarchical
        relationships, na vals, etc."""
        raise NotImplementedError()

    def fit(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        """Fits the model based on the provided data.

        Data and Ids are dictionaries containing the dataframes with the data."""
        raise NotImplementedError()

    def sample(
        self, *, n: int | None = None, i: int = 0
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Returns id, table dict dataframes in the same format they were provided.

        Optional `n` parameter sets how many rows should be sampled. Otherwise,
        the initial size of the dataset is sampled.
        Warning: not setting `n` technically violates DP for DP-aware algorithms.

        `i` is the partition number that can be used for modifying the random state
        sampling, since deterministic sampling will always return the same data.
        """
        raise NotImplementedError()


def synth_fit(
    factory: SynthFactory,
    metadata: Metadata,
    ids: dict[str, LazyFrame],
    tables: dict[str, LazyFrame],
    trns: dict[str, TransformHolder],
):
    from .utils.perf import PerformanceTracker

    tracker = PerformanceTracker.get("synth")

    tracker.ensemble("total", "preprocess", "bake", "fit", "sample")

    meta = metadata
    args = {**meta.algs.get(factory.name, {}), **meta.alg_override}

    attrs = {n: t[factory.type].get_attributes() for n, t in trns.items()}
    model = factory.build(**args, seed=meta.seed)

    if factory.gpu:
        tracker.use_gpu()

    tracker.start("preprocess")
    model.preprocess(attrs, ids, tables)
    tracker.stop("preprocess")

    tracker.start("bake")
    model.bake(ids, tables)
    tracker.stop("bake")

    tracker.start("fit")
    model.fit(ids, tables)
    tracker.stop("fit")
    return model


def _synth_sample_part(i: int, n: int | None, model: Synth):
    ids, tables = model.sample(n=n, i=i)

    return {
        "tables": {name: {f"{i:04d}": table} for name, table in tables.items()},
        "ids": {name: {f"{i:04d}": table} for name, table in ids.items()},
    }


def synth_sample(model: Synth, n: int | None = None, partitions: int | None = None):
    # TODO: Track synth speed
    return {
        partial(_synth_sample_part, i, n, model)
        for i in range(partitions or model.partitions)
    }


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = "ident_idx"
    type = "idx"

    tabular = True
    multimodal = True
    timeseries = True
    partitions = 1

    def preprocess(
        self,
        attrs: dict[str, Attributes],
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        pass

    def bake(self, ids: dict[str, LazyFrame], tables: dict[str, LazyFrame]):
        pass

    def fit(self, ids: dict[str, LazyFrame], tables: dict[str, LazyFrame]):
        self._ids = {name: table.sample() for name, table in ids.items()}
        self._tables = {name: table.sample() for name, table in tables.items()}
        self.partitions = len(tables[next(iter(tables))])

    def sample(self, n: int | None = None, i: int = 0):
        return self._ids, self._tables


__all__ = ["Synth", "synth_fit", "synth_sample"]
