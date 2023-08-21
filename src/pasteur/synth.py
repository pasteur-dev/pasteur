""" Contains the base definition for Synth(esizer modules).

In addition, a test Synthesizer (IdentSynth) is provided, which returns
the data it was provided as is. """

from __future__ import annotations

from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pasteur.utils import LazyDataset

from .encode import ViewEncoder
from .metadata import Metadata
from .module import ModuleClass, ModuleFactory
from .utils import LazyDataset, LazyFrame

META = TypeVar("META")

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
            noise_info = f" ('{noise_kw}': {kwargs[noise_kw]:3d})" if noise_kw else ""
            logger.info(
                f"Deterministic check: random number after "
                + f"{f'{type(self).__name__}.{obj_func.__name__}':>22s}(): "
                + f"<np.random> {np.random.random():7.5f} <random> {random.random():7.5f}"
                + noise_info
            )
        return a

    return wrapped


class SynthFactory(ModuleFactory["Synth"]):
    def __init__(self, cls: type[Synth], *args, name: str | None = None, **_) -> None:
        super().__init__(cls, *args, name=name, **_)
        self.type = cls.type


class Synth(ModuleClass, Generic[META]):
    type = "idx"
    _factory = SynthFactory

    # Fill in for `sample` function to work
    _n: int | None = None
    # Fill in for `sample` function to work
    _partitions: int | None = None

    def preprocess(
        self,
        meta: META,
        data: dict[str, LazyDataset],
    ):
        """Runs any preprocessing required, such as domain reduction."""
        raise NotImplementedError()

    def bake(
        self,
        data: dict[str, LazyDataset],
    ):
        """Bakes the model based on the data provided (such as creating and
        modeling a bayesian network on the data).

        Attributes provide context about the data columns, including hierarchical
        relationships, na vals, etc."""
        raise NotImplementedError()

    def fit(
        self,
        data: dict[str, LazyDataset],
    ):
        """Fits the model based on the provided data.

        Data and Ids are dictionaries containing the dataframes with the data."""
        raise NotImplementedError()

    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        """Returns synthetic data in the same format they were provided.

        `n` sets how many rows should be sampled. Otherwise,
        Warning: not setting `n` technically violates DP for DP-aware algorithms.

        `i` is the partition number that can be used for modifying the random state
        sampling, since deterministic sampling will always return the same data.
        """
        raise NotImplementedError()

    def sample(self, *, n: int | None = None, partitions: int | None = None):
        """Samples `n` samples across `partitions` partitions.

        The return value should be finalized to `dict[str, Any]`, which
        matches the format of `data` provided to the fitting function.
        Since this

        A default implementation is provided, that packages `sample_partition()`
        in such a way that pasteur can sample and save partitions in parallel."""
        n = n or self._n
        partitions = partitions or self._partitions

        assert (
            n and partitions
        ), "Either `n` or `partitions` was not provided.\nFill in `_n` and `_partitions` based on `fit` data."

        n_chunk = n // partitions

        return {
            partial(self.sample_partition, i=i, n=n_chunk) for i in range(partitions)
        }


def synth_fit(
    factory: SynthFactory,
    metadata: Metadata,
    encoder: ViewEncoder,
    data: dict[str, LazyDataset],
):
    from .utils.perf import PerformanceTracker

    tracker = PerformanceTracker.get("synth")

    tracker.ensemble("total", "preprocess", "bake", "fit")

    meta = encoder.get_metadata()
    args = {**metadata.algs.get(factory.name, {}), **metadata.alg_override}
    model = factory.build(**args, seed=metadata.seed)

    # if factory.gpu:
    #     tracker.use_gpu()

    tracker.start("preprocess")
    model.preprocess(meta, data)
    tracker.stop("preprocess")

    tracker.start("bake")
    model.bake(data)
    tracker.stop("bake")

    tracker.start("fit")
    model.fit(data)
    tracker.stop("fit")

    return model


def synth_sample(s: Synth):
    return s.sample()


class IdentSynth(Synth):
    """Samples the data it was provided."""

    name = "ident_idx"
    type = "idx"
    partitions = 1

    def preprocess(self, meta: Any, data: dict[str, LazyDataset]):
        pass

    def bake(self, data: dict[str, LazyDataset]):
        pass

    def fit(self, data: dict[str, LazyDataset]):
        self.data = data

    def sample(self, n: int | None = None):
        return self.data


__all__ = ["Synth", "SynthFactory", "IdentSynth", "make_deterministic"]
