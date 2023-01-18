from __future__ import annotations

import logging
from typing import cast

from ....marginal import AttrSelector, MarginalOracle, MarginalRequest
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame
from mbi import Dataset, Domain
from aim import AIM as AIMimpl
from ....attribute import Attributes, IdxValue
import pandas as pd


logger = logging.getLogger(__name__)


class OracleDataset(Dataset):
    def __init__(
        self,
        o: MarginalOracle,
        domain: Domain | None = None,
        force_cache: bool = True,
        cache: dict = {},
    ):
        self.o = o
        self.attrs = o.attrs
        self.cache = cache
        self.force_cache = force_cache

        if domain is not None:
            self.domain = domain
        else:
            names = []
            domains = []

            for attr in self.attrs.values():
                for val in attr.vals.values():
                    names.append(val.name)
                    domains.append(cast(IdxValue, val).get_domain(0))

            self.domain = Domain(names, domains)

    def project(self, cols):
        """project dataset onto a subset of columns"""
        if type(cols) in [str, int]:
            cols = [cols]
        domain = self.domain.project(cols)
        return OracleDataset(
            self.o, domain, force_cache=self.force_cache, cache=self.cache
        )

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    @property
    def records(self):
        return self.o.get_shape()[0]

    def datavector(self, flatten=True):
        """return the database in vector-of-counts form"""
        assert flatten

        cols = tuple(self.domain.attrs)

        if cols in self.cache:
            return self.cache[cols]
        else:
            assert not self.force_cache

        return self.o.process(
            [
                MarginalRequest(
                    None,
                    {col: AttrSelector(col, 0, {col: 0}) for col in self.domain.attrs},
                    False,
                    0,
                    False,
                )
            ]
        )[0]

    def cache_marginals(self, requests: list[tuple[str]]):
        non_cached_req = [req for req in requests if req not in self.cache]
        marginals = self.o.process(
            [
                MarginalRequest(
                    None,
                    {col: AttrSelector(col, 0, {col: 0}) for col in cols},
                    False,
                    0,
                    False,
                )
                for cols in non_cached_req
            ]
        )

        for req, mar in zip(non_cached_req, marginals):
            self.cache[req] = mar


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
        seed: int | None = None,
        **kwargs,
    ) -> None:
        self.e = e
        self.delta = delta
        self.num_marginals = num_marginals
        self.degree = degree
        self.max_cells = max_cells
        self.rounds = rounds
        self.seed = seed
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

        table = tables[self.table]
        self.partitions = len(table)
        self.n = table.shape[0] // self.partitions

        with MarginalOracle(
            self.attrs[self.table], tables[self.table], batched=False
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
