from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ....synth import Synth, make_deterministic
from ....utils import LazyFrame

if TYPE_CHECKING:
    from ....attribute import Attributes
    from ....marginal import MarginalOracle

def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, sigma, proj in measurements:
        col = proj[0]
        sup = y >= 3*sigma
        supports[col] = sup
        if supports[col].sum() == y.size:
            new_measurements.append( (Q, y, sigma, proj) )
        else: # need to re-express measurement over the new domain
            y2 = np.append(y[sup], y[~sup].sum())
            I2 = np.ones(y2.size)
            I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
            y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
            I2 = sparse.diags(I2)
            new_measurements.append( (I2, y2, sigma, proj) )
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn

def MST_out_of_core(data, epsilon, delta):
    from mst import cdp_rho, measure, select, FactoredInference
    import numpy as np

    rho = cdp_rho(epsilon, delta)
    sigma = np.sqrt(3/(2*rho))
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques, sigma)
    data, log1, undo_compress_fn = compress_domain(data, log1)

    cliques = select(data, rho/3.0, log1)
    log2 = measure(data, cliques, sigma)
    engine = FactoredInference(data.domain, iters=5000)
    est = engine.estimate(log1+log2)
    return est, supports

logger = logging.getLogger(__name__)

class MST(Synth):
    name = "mst"
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
        from mst import MST as MSTimpl

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
            self.model = MSTimpl(data, self.e, self.delta)

    @make_deterministic("i")
    def sample(
        self, *, n: int | None = None, i: int = 0
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        data = self.model.synthetic_data(n or self.n)
        return {self.table: pd.DataFrame()}, {self.table: data.df}
