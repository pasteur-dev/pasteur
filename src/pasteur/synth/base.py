import logging

import numpy as np
import pandas as pd
import random

from tqdm.contrib.concurrent import thread_map, process_map
from os import cpu_count

from ..transform import TableTransformer

logger = logging.getLogger(__name__)


def calc_worker(args):
    fun, base_args, chunk = args
    out = []
    for op in chunk:
        args = {**base_args, **op} if base_args else op
        out.append(fun(**args))

    return out


def process_in_parallel(
    fun: callable,
    per_call_args: list[dict],
    base_args: dict[str, any] | None = None,
    desc: str | None = None,
):
    """Processes arguments in parallel using python's multiprocessing and prints progress bar.

    Task is split into chunks based on CPU cores and each process handles a chunk of
    calls before exiting."""
    chunk_n = cpu_count() * 5
    per_call_n = len(per_call_args) // chunk_n

    chunks = np.array_split(per_call_args, chunk_n)

    args = []
    for chunk in chunks:
        args.append((fun, base_args, chunk))

    res = process_map(
        calc_worker,
        args,
        desc=f"{desc}, {per_call_n}/{len(per_call_args)} per it",
        leave=False,
    )
    out = []
    for sub_arr in res:
        out.extend(sub_arr)

    return out


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


class Synth:
    name = None
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False

    def __init__(self, **_) -> None:
        pass

    def bake(
        self,
        transformers: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Bakes the transformer based on the data provided (such as creating a
        modeling a bayesian network on the data). Does not fit the transformer
        to the data. Optional"""
        pass

    def fit(
        self,
        transformers: dict[str, pd.DataFrame],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        """Bakes and fits the model based on the provided data.

        Transformers provide the Synthetic algorithm with access to the
        Metadata of the dataset and the hierarchical attributes.

        Data and Ids are dictionaries containing the dataframes with the data."""
        assert False, "Not implemented"

    def sample(
        self, n: int = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Returns data, ids dict dataframes in the same format they were provided.

        Optional `n` parameter sets how many rows should be sampled. Otherwise,
        the initial size of the dataset is sampled.
        Warning: not setting `n` technically violates DP for DP-aware algorithms."""
        assert False, "Not implemented"


def synth_fit_closure(cls):
    def fit(**kwargs: pd.DataFrame | TableTransformer):
        transformers = {n[4:]: t for n, t in kwargs.items() if "trn_" in n}
        ids = {n[4:]: i for n, i in kwargs.items() if "ids_" in n}
        data = {n[4:]: d for n, d in kwargs.items() if "enc_" in n}

        meta = next(iter(transformers.values())).meta
        model = (
            cls(**meta.algs[cls.name], seed=meta.seed)
            if cls.name in meta.algs
            else cls(seed=meta.seed)
        )
        model.bake(transformers, data, ids)
        model.fit(transformers, data, ids)
        return model

    return fit


def synth_sample(model: Synth):
    data, ids = model.sample()

    return {
        **{f"enc_{n}": d for n, d in data.items()},
        **{f"ids_{n}": d for n, d in ids.items()},
    }


class IdentSynth(Synth):
    """Returns the data it was provided."""

    name = None
    type = "idx"
    tabular = True
    multimodal = True
    timeseries = True

    def fit(
        self,
        transformers: dict[str, pd.DataFrame],
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


class BinIdentSynth(IdentSynth):
    name = "ident_bin"
    type = "bin"


class BhrIdentSynth(IdentSynth):
    name = "ident_bhr"
    type = "bhr"
