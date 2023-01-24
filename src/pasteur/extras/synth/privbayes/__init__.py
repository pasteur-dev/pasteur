from __future__ import annotations

import logging
from math import ceil
from typing import TYPE_CHECKING

from ....marginal import MarginalOracle
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame

if TYPE_CHECKING:
    import pandas as pd

    from ....attribute import Attributes

logger = logging.getLogger(__name__)


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
        self,
        attrs: dict[str, Attributes],
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        from ....hierarchy import rebalance_attributes

        table_name = next(iter(tables.keys()))
        table = tables[table_name]
        table_attrs = attrs[table_name]

        if self.rebalance:
            with MarginalOracle(
                table_attrs,
                table,
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
            ) as o:
                counts = o.get_counts(desc="Calculating counts for column rebalancing")

            self.attrs = rebalance_attributes(
                counts,
                table_attrs,
                self.ep,
                unbounded_dp=self.unbounded_dp,
                **self.kwargs,
            )
        else:
            self.attrs = table_attrs

    @make_deterministic
    def bake(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        from .implementation import greedy_bayes

        assert len(tables) == 1, "Only tabular data supported for now"

        table_name = next(iter(tables.keys()))
        table = tables[table_name]

        with MarginalOracle(
            self.attrs,
            table,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            self.n, self.d = oracle.get_shape()
            # Fit network
            nodes, t = greedy_bayes(
                oracle,
                self.attrs,
                self.e1,
                self.e2,
                self.theta,
                self.use_r,
                self.unbounded_dp,
                self.random_init,
                self.skip_zero_counts,
            )

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.t = t
        self.nodes = nodes
        logger.info(self)

    @make_deterministic
    def fit(
        self,
        ids: dict[str, LazyFrame],
        tables: dict[str, LazyFrame],
    ):
        from .implementation import MAX_EPSILON, calc_noisy_marginals

        table = tables[self.table_name]
        self.partitions = len(table)
        self.n = ceil(table.shape[0] / self.partitions)

        noise = (1 if self.unbounded_dp else 2) * self.d / self.e2 / self.n
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0

        with MarginalOracle(
            self.attrs,
            table,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as o:
            self.marginals = calc_noisy_marginals(
                o, self.attrs, self.nodes, noise, self.skip_zero_counts
            )

    @make_deterministic("i")
    def sample(
        self, *, n: int | None = None, i: int = 0
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        import pandas as pd

        from .implementation import sample_rows

        data = {
            self.table_name: sample_rows(
                self.attrs, self.nodes, self.marginals, self.n if n is None else n  # type: ignore
            )
        }
        ids = {self.table_name: pd.DataFrame()}

        return ids, data

    def __str__(self) -> str:
        from .implementation import print_tree

        return print_tree(
            self.attrs,
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )
