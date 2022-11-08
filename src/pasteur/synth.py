from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from .attribute import Attributes
    from .transform import TableTransformer
    from .metadata import Metadata

import logging

logger = logging.getLogger(__name__)


def make_deterministic(obj_func):
    """Takes an object function (with self), and if the object has a seed attribute
    it fixes the np.random.seed attribute to it and prints a random number at the end.

    If the algorithm sampled the same amount of numbers at the same order, then the
    numbers should be the same."""

    import random

    import numpy as np

    def wrapped(self, *args, **kwargs):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        a = obj_func(self, *args, **kwargs)

        if self.seed is not None:
            logger.info(
                f"Deterministic check: random number after "
                + f"{f'{type(self).__name__}.{obj_func.__name__}':>22s}(): "
                + f"<np.random> {np.random.random():7.5f} <random> {random.random():7.5f}"
            )
        return a

    return wrapped


class Synth(ABC):
    name = None
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    gpu = False
    parallel = False

    def __init__(self, **_) -> None:
        pass

    @abstractmethod
    def preprocess(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Runs any preprocessing required, such as domain reduction."""
        pass

    @abstractmethod
    def bake(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Bakes the model based on the data provided (such as creating and
        modeling a bayesian network on the data).

        Attributes provide context about the data columns, including hierarchical
        relationships, na vals, etc."""
        pass

    @abstractmethod
    def fit(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Fits the model based on the provided data.

        Data and Ids are dictionaries containing the dataframes with the data."""
        pass

    @abstractmethod
    def sample(
        self, n: int | None = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Returns data, ids dict dataframes in the same format they were provided.

        Optional `n` parameter sets how many rows should be sampled. Otherwise,
        the initial size of the dataset is sampled.
        Warning: not setting `n` technically violates DP for DP-aware algorithms."""
        pass


def synth_fit(
    cls: type[Synth], metadata: Metadata, **kwargs: pd.DataFrame | TableTransformer
):
    from .utils import PerformanceTracker

    tracker = PerformanceTracker.get("synth")

    tracker.ensemble("total", "preprocess", "bake", "fit", "sample")

    ids = {n[4:]: i for n, i in kwargs.items() if "ids_" in n}
    data = {n[4:]: d for n, d in kwargs.items() if "enc_" in n}
    trns = {n[4:]: t for n, t in kwargs.items() if "trn_" in n}

    meta = metadata
    args = {**meta.algs.get(cls.name, {}), **meta.alg_override}

    attrs = {n: t[cls.type].get_attributes() for n, t in trns.items()}
    model = cls(**args, seed=meta.seed)

    if cls.gpu:
        tracker.use_gpu()

    tracker.start("preprocess")
    model.preprocess(attrs, data, ids)
    tracker.stop("preprocess")

    tracker.start("bake")
    model.bake(data, ids)
    tracker.stop("bake")

    tracker.start("fit")
    model.fit(data, ids)
    tracker.stop("fit")
    return model


def synth_sample(model: Synth):
    from .utils import PerformanceTracker

    tracker = PerformanceTracker.get("synth")

    tracker.start("sample")
    data, ids = model.sample()
    tracker.stop("sample")

    return {
        **{f"enc_{n}": d for n, d in data.items()},
        **{f"ids_{n}": d for n, d in ids.items()},
    }


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = "ident_idx"
    type = "idx"
    tabular = True
    multimodal = True
    timeseries = True

    def preprocess(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        pass

    def bake(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        pass

    def fit(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        self._data = data
        self._ids = ids

    def sample(self):
        return self._data, self._ids


class NumIdentSynth(IdentSynth):
    name = "ident_num"
    type = "num"


__all__ = ["Synth", "synth_fit", "synth_sample"]