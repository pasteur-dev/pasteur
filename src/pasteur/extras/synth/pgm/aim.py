from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ....synth import Synth, make_deterministic
from ....utils import LazyFrame

if TYPE_CHECKING:
    from ....attribute import Attributes
    from ....marginal import MarginalOracle


logger = logging.getLogger(__name__)

class AIM(Synth):
    name = "aim"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

    def __init__(
        self,
        e: float = 1.0,
        delta: float = 1e-9,
        num_marginals: int = 0,
        degree: int = 2,
        max_cells: int = 2**16,
        rounds: int = 50,
        marginal_mode: "MarginalOracle.MODES" = "out_of_core",
        seed: int | None = None,
        n: int | None = None,
        partitions: int | None = None,
        **kwargs,
    ) -> None:
        self.e = e
        self.delta = delta
        self.num_marginals = num_marginals
        self.degree = degree
        self.max_cells = max_cells
        self.rounds = rounds
        self.seed = seed
        self.n = n
        self.partitions = partitions
        self.marginal_mode: "MarginalOracle.MODES" = marginal_mode
        self.kwargs = kwargs


    @make_deterministic
    def preprocess(
        self,
        attrs: dict[str, Attributes],
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        self.table = next(iter(attrs))
        self.attrs = attrs
        pass

    @make_deterministic
    def bake(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        pass

    @make_deterministic
    def fit(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        import itertools

        import numpy as np
        from aim import AIM as AIMimpl

        from ....marginal import MarginalOracle
        from .common import OracleDataset

        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)

        with MarginalOracle(
            self.attrs[self.table], tables[self.table], mode=self.marginal_mode
        ) as o:
            data = OracleDataset(o)

            workload = list(itertools.combinations(data.domain, self.degree))
            workload = [cl for cl in workload if data.domain.size(cl) <= self.max_cells]
            if self.num_marginals > 0:
                workload = [
                    workload[i]
                    for i in np.random.choice(
                        len(workload), self.num_marginals, replace=False
                    )
                ]

            workload = [(cl, 1.0) for cl in workload]
            mech = AIMimpl(self.e, self.delta, rounds=self.rounds)

            self.model = mech.run(data, workload)

    @make_deterministic("i")
    def sample(
        self, *, n: int | None = None, i: int = 0
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        data = self.model.synthetic_data(n or self.n)
        return {self.table: pd.DataFrame()}, {self.table: data.df}
