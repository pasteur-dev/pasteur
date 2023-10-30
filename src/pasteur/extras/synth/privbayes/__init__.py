from __future__ import annotations

import logging
from math import ceil
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from ....attribute import Attributes, DatasetAttributes
from ....mare.synth import MareModel
from ....marginal import MarginalOracle
from ....marginal.numpy import TableSelector
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data

logger = logging.getLogger(__name__)


class PrivBayesMare(MareModel):
    def __init__(
        self,
        ep: float | None = None,
        e1: float = 0.3,
        e2: float = 0.7,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        rebalance: bool = False,
        unbounded_dp: bool = False,
        random_init: bool = False,
        skip_zero_counts: bool = False,
        **kwargs,
    ) -> None:
        self.ep = ep
        self.e1 = e1
        self.e2 = e2
        self.theta = theta
        self.use_r = use_r
        self.seed = seed
        self.random_init = random_init
        self.unbounded_dp = unbounded_dp
        self.rebalance = rebalance
        self.skip_zero_counts = skip_zero_counts
        self.kwargs = kwargs

    @make_deterministic
    def fit(self, n: int, attrs: DatasetAttributes, oracle: MarginalOracle):
        from .implementation import MAX_EPSILON, calc_noisy_marginals, greedy_bayes

        # Fit network
        nodes, t = greedy_bayes(
            oracle,
            attrs,
            n,
            self.e1,
            self.e2,
            self.theta,
            self.use_r,
            self.unbounded_dp,
            self.random_init,
        )

        # Nodes are a tuple of a x attribute
        self.t = t
        self.nodes = nodes
        self.attrs = attrs
        logger.info(self)

        d = 0
        for attr in cast(Attributes, attrs[None]).values():
            d += len(attr.vals)

        noise = (1 if self.unbounded_dp else 2) * d / self.e2 / n
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0

        self.marginals = calc_noisy_marginals(
            oracle, self.nodes, noise, self.skip_zero_counts
        )

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame:
        import pandas as pd

        from .implementation import sample_rows

        return sample_rows(index, self.attrs, hist, self.nodes, self.marginals)

    def __str__(self) -> str:
        from .implementation import print_tree

        return print_tree(
            cast(Attributes, self.attrs[None]),
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )


class PrivBayesSynth(Synth):
    name = "privbayes"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

    def __init__(
        self,
        ep: float | None = None,
        e1: float = 0.3,
        e2: float = 0.7,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        rebalance: bool = False,
        unbounded_dp: bool = False,
        random_init: bool = False,
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        skip_zero_counts: bool = False,
        **kwargs,
    ) -> None:
        self.ep = ep
        self.e1 = e1
        self.e2 = e2
        self.theta = theta
        self.use_r = use_r
        self.seed = seed
        self.random_init = random_init
        self.unbounded_dp = unbounded_dp
        self.rebalance = rebalance
        self.marginal_mode: MarginalOracle.MODES = marginal_mode
        self.marginal_min_chunk = marginal_min_chunk
        self.marginal_worker_mult = marginal_worker_mult
        self.skip_zero_counts = skip_zero_counts
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(
        self, meta: dict[str | None, Attributes], data: dict[str, LazyFrame]
    ):
        from ....hierarchy import rebalance_attributes

        attrs = meta
        _, tables = data_to_tables(data)
        table_name = next(iter(tables.keys()))
        table = tables[table_name]
        table_attrs = attrs[table_name]
        self.table_attrs = table_attrs

        self._n = table.shape[0]
        self._partitions = len(table)
        self.attrs = attrs

        # if self.rebalance:
        #     with MarginalOracle(
        #         data,  # type: ignore
        #         table_attrs,
        #         mode=self.marginal_mode,
        #         min_chunk_size=self.marginal_min_chunk,
        #         max_worker_mult=self.marginal_worker_mult,
        #     ) as o:
        #         counts = o.get_counts(desc="Calculating counts for column rebalancing")

        #     self.attrs = rebalance_attributes(
        #         counts[None],
        #         table_attrs,
        #         self.ep,
        #         unbounded_dp=self.unbounded_dp,
        #         **self.kwargs,
        #     )
        # else:
        #     self.attrs = table_attrs

    @make_deterministic
    def bake(self, data: dict[str, LazyFrame]):
        from .implementation import greedy_bayes

        _, tables = data_to_tables(data)

        assert len(tables) == 1, "Only tabular data supported for now"

        table_name = next(iter(tables.keys()))
        table = tables[table_name]

        with MarginalOracle(
            data,  # type: ignore
            self.attrs[table_name],
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            self.n, self.d = table.shape
            # Fit network
            nodes, t = greedy_bayes(
                oracle,
                {None: self.table_attrs},
                table.shape[0],
                self.e1,
                self.e2,
                self.theta,
                self.use_r,
                self.unbounded_dp,
                self.random_init,
            )

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.t = t
        self.nodes = nodes
        logger.info(self)

    @make_deterministic
    def fit(self, data: dict[str, LazyFrame]):
        from .implementation import MAX_EPSILON, calc_noisy_marginals

        _, tables = data_to_tables(data)
        table = tables[self.table_name]
        self.partitions = len(table)
        self.n = ceil(table.shape[0] / self.partitions)

        noise = (1 if self.unbounded_dp else 2) * self.d / self.e2 / self.n
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0

        with MarginalOracle(
            data,  # type: ignore
            self.table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as o:
            self.marginals = calc_noisy_marginals(
                o, self.nodes, noise, self.skip_zero_counts
            )

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        import pandas as pd

        from .implementation import sample_rows

        if n is None:
            n = self.n
        tables = {
            self.table_name: sample_rows(
                pd.RangeIndex(n), {None: self.attrs[self.table_name]}, {}, self.nodes, self.marginals
            )
        }
        ids = {self.table_name: pd.DataFrame()}

        return tables_to_data(ids, tables)

    def __str__(self) -> str:
        from .implementation import print_tree

        return print_tree(
            self.attrs[self.table_name],
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )
