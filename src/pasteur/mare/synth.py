import logging
from typing import Any, Type

import pandas as pd

from .chains import TablePartition, TableVersion

from ..attribute import Attributes, DatasetAttributes
from ..marginal import MarginalOracle
from ..marginal.numpy import TableSelector
from ..synth import Synth, make_deterministic
from ..utils import LazyDataset
from .unroll import ModelVersion, calculate_model_versions

logger = logging.getLogger(__name__)


class MareModel:
    def fit(self, n: int, attrs: DatasetAttributes, oracle: MarginalOracle):
        ...

    def sample(self, hist: dict[TableSelector, pd.DataFrame]):
        ...


class MareSynth(Synth):
    name = "mare"
    type = "idx"

    def __init__(
        self,
        model_cls: Type[MareModel],
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        max_vers: int = 20,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.max_vers = max_vers

        self.model_cls = model_cls

    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyDataset]):
        logger.info(
            f"Calculating required model versions for dataset (max versions per model: {self.max_vers})..."
        )
        self.versions = calculate_model_versions(meta, data, self.max_vers)
        logger.info(f"Calculated {len(self.versions)} model versions.")

    def bake(self, data: dict[str, LazyDataset]):
        ...

    @make_deterministic
    def fit(self, data: dict[str, LazyDataset]):
        self.models = {}
        for i, (ver, (attrs, load)) in enumerate(self.versions.items()):
            logger.info(
                f"Fitting {i + 1:2d}/{len(self.versions)} '{'context' if ver.ctx else 'series'}' model for table '{ver.ver.name}'"
            )
            with MarginalOracle(data, attrs, load, mode=self.marginal_mode) as o:
                model = self.model_cls(**self.kwargs)
                model.fit(ver.ver.rows, attrs, o)
                self.models[ver] = model

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0):
        todo = list(self.models)

        while todo:
            # Find synthesizable candidate
            ver = None
            for candidate in todo:
                if meets_requirements(candidate, todo):
                    ver = candidate
            assert ver is not None


def get_parents(version: TableVersion):
    out = set()
    for p in version.parents:
        if isinstance(p, TablePartition):
            out.add(p.table.name)
            out.update(get_parents(p.table))
        else:
            out.add(p.name)
            out.update(get_parents(p))
    return out


def meets_requirements(ver: ModelVersion, todo: list[ModelVersion]):
    parents = get_parents(ver.ver)

    # Find if a version that is not currently done is required
    for other in todo:

        # If the other version is a parent
        if other.ver.name in parents:
            return False

        # If this is a series model and a context model of this table is not synthesized
        if not ver.ctx and other.ver.name == ver.ver.name and other.ctx:
            return False

    return True
