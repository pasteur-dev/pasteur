from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Synth, make_deterministic

if TYPE_CHECKING:
    import pandas as pd

    from ..transform import Attributes

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
        ep: float = None,
        e1: float = 0.3,
        e2: float = 0.7,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        rebalance_columns: bool = False,
        unbounded_dp: bool = False,
        random_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ep = ep
        self.e1 = e1
        self.e2 = e2
        self.theta = theta
        self.use_r = use_r
        self.seed = seed
        self.random_init = random_init
        self.unbounded_dp = unbounded_dp
        self.rebalance = rebalance_columns
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(
        self,
        attrs: dict[str, Attributes],
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        from .hierarchy import rebalance_attributes

        table_name = next(iter(data.keys()))
        table = data[table_name]
        table_attrs = attrs[table_name]

        if self.rebalance:
            self.attrs = rebalance_attributes(
                table,
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
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        from .privbayes import greedy_bayes

        assert len(data) == 1, "Only tabular data supported for now"

        table_name = next(iter(data.keys()))
        table = data[table_name]

        # Fit network
        nodes, t = greedy_bayes(
            table,
            self.attrs,
            self.e1,
            self.e2,
            self.theta,
            self.use_r,
            self.unbounded_dp,
            self.random_init,
        )

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.d = len(table.keys())
        self.t = t
        self.nodes = nodes
        logger.info(self)

    @make_deterministic
    def fit(
        self,
        data: dict[str, pd.DataFrame],
        ids: dict[str, pd.DataFrame],
    ):
        from .privbayes import MAX_EPSILON, calc_noisy_marginals

        table = data[self.table_name]
        self.n = len(table)
        noise = (1 if self.unbounded_dp else 2) * self.d / self.e2
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0
        self.marginals = calc_noisy_marginals(self.attrs, table, self.nodes, noise)

    @make_deterministic
    def sample(
        self, n: int = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        import pandas as pd

        from .privbayes import sample_rows

        data = {
            self.table_name: sample_rows(
                self.attrs, self.nodes, self.marginals, self.n if n is None else n
            )
        }
        ids = {self.table_name: pd.DataFrame()}

        return data, ids

    def __str__(self) -> str:
        from .privbayes import print_tree

        return print_tree(
            self.attrs,
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
        )
