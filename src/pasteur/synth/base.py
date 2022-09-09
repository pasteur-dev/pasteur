from abc import ABC, abstractmethod
import logging
import random

import numpy as np
import pandas as pd

from ..transform import Attribute, Attributes, TableTransformer
from ..metadata import Metadata

logger = logging.getLogger(__name__)


def make_deterministic(obj_func):
    """Takes an object function (with self), and if the object has a seed attribute
    it fixes the np.random.seed attribute to it and prints a random number at the end.

    If the algorithm sampled the same amount of numbers at the same order, then the
    numbers should be the same."""

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

    def __init__(self, **_) -> None:
        pass

    @abstractmethod
    def bake(
        self,
        attrs: dict[str, dict[str, Attribute]],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Bakes the transformer based on the data provided (such as creating a
        modeling a bayesian network on the data). Does not fit the transformer
        to the data. Optional"""
        pass

    @abstractmethod
    def fit(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Bakes and fits the model based on the provided data.

        Transformers provide the Synthetic algorithm with access to the
        Metadata of the dataset and the hierarchical attributes.

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
    ids = {n[4:]: i for n, i in kwargs.items() if "ids_" in n}
    data = {n[4:]: d for n, d in kwargs.items() if "enc_" in n}
    trns = {n[4:]: t for n, t in kwargs.items() if "trn_" in n}

    meta = metadata
    args = {**meta.algs.get(cls.name, {}), **meta.alg_override}

    attrs = {n: t[cls.type].get_attributes() for n, t in trns.items()}
    model = cls(**args, seed=meta.seed)
    model.bake(attrs, data, ids)
    model.fit(data, ids)
    return model


def synth_sample(model: Synth):
    data, ids = model.sample()

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

    def bake(
        self,
        attrs: dict[str, Attributes],
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
