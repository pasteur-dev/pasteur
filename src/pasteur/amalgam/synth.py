from typing import Any

from pasteur.synth import Synth
from pasteur.utils import LazyDataset, gen_closure
from pasteur.mare.synth import MareModel

import logging
from collections import defaultdict
from typing import Any, Type, cast

import pandas as pd
import numpy as np
from pasteur.marginal import MarginalOracle, counts_preprocess

from pasteur.attribute import (
    Attributes,
    CatValue,
    DatasetAttributes,
    SeqAttributes,
    get_dtype,
)


def _repack(pid, ids, data):
    return {
        "ids": {pid: ids()},
        "data": {pid: data()},
    }


class AmalgamSynth(Synth):
    name = "amalgam"
    in_types = ["json", "idx"]
    in_sample = True
    type = "json"
    partitions = 1

    def __init__(
        self,
        model_cls: Type[MareModel],
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.model_cls = model_cls

    def preprocess(self, meta: Any, data: dict[str, dict[str, LazyDataset]]):
        with MarginalOracle(
            data["idx"],  # type: ignore
            meta["idx"],  # type: ignore
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
            preprocess=counts_preprocess,
        ) as o:
            self.counts = o.get_counts(desc="Calculating counts for column rebalancing")
        self.meta = meta

    def bake(self, data: dict[str, dict[str, LazyDataset]]): ...

    def fit(self, data: dict[str, dict[str, LazyDataset]]):
        top_table = self.meta["json"]["top_table"]
        with MarginalOracle(
            {top_table: data["idx"][top_table]},  # type: ignore
            {None: self.meta["idx"][top_table]},  # type: ignore
            mode=self.marginal_mode,
            max_worker_mult=self.marginal_worker_mult,
            min_chunk_size=self.marginal_min_chunk,
        ) as o:
            kwargs = dict(self.kwargs)

            model = self.model_cls(**kwargs)
            model.fit(
                data["idx"][top_table].shape[0],
                top_table,
                {None: self.meta["idx"][top_table]},
                o,
            )
            self.model = model

    def sample(self, n: int | None = None, data=None):
        return {
            gen_closure(_repack, pid, i, d)
            for pid, (i, d) in LazyDataset.zip(data["json"]["ids"], data["json"]["data"]).items()  # type: ignore
        }
