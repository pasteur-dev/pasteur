"""MST: Marginal Synthesis via Tree.

Generalization of the winning mechanism from the 2018 NIST DP Synthetic Data
Competition. Selects 2-way marginals via an exponential-mechanism MST, then
fits a PGM using our mirror descent + BP engine.

Reference: private-pgm-ref/mechanisms/mst.py
"""

from __future__ import annotations

import itertools
import logging
from math import sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ....attribute import Attributes, DatasetAttributes
from ....marginal import MarginalOracle
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data
from .common import cdp_rho, measure, fit_pgm, exponential_mechanism, get_attr_names, _col_to_attr_sel

if TYPE_CHECKING:
    pass

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
        marginal_mode: "MarginalOracle.MODES" = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        seed: int | None = None,
        n: int | None = None,
        partitions: int | None = None,
        mirror_descent: dict | None = None,
        **kwargs,
    ) -> None:
        self.e = e
        self.delta = delta
        self.marginal_mode = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.seed = seed
        self.n = n
        self.partitions = partitions
        self.md_params = mirror_descent or {}
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyFrame]):
        self.table = next(iter(meta))
        self.attrs = meta
        self._n = data[self.table].shape[0]
        self._partitions = len(data[self.table])

    @make_deterministic
    def bake(self, data: dict[str, LazyFrame]):
        pass

    @make_deterministic
    def fit(self, data: dict[str, LazyFrame]):
        from scipy.cluster.hierarchy import DisjointSet

        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = self.n
        table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        rho = cdp_rho(self.e, self.delta)
        sigma = sqrt(3 / (2 * rho))

        all_attrs = get_attr_names(table_attrs)

        with MarginalOracle(
            data,
            table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            # Phase 1: Measure all 1-way marginals
            oneway = [(a,) for a in all_attrs]
            logger.info(
                f"MST Phase 1: Measuring {len(oneway)} 1-way marginals (sigma={sigma:.2f})"
            )
            log1 = measure(oracle, table_attrs, oneway, sigma)

            # Phase 2: Select 2-way cliques via exponential mechanism
            logger.info("MST Phase 2: Selecting 2-way cliques")

            # Fit 1-way model to estimate errors
            est = fit_pgm(
                table_attrs, log1, n,
                {**self.md_params},
            )

            # Compute L1 error on all 2-way candidates (single batched call)
            candidates = list(itertools.combinations(all_attrs, 2))
            requests = []
            for a, b in candidates:
                req = {}
                for col_name in (a, b):
                    attr_name, sel = _col_to_attr_sel(col_name, table_attrs)
                    if attr_name in req:
                        req[attr_name].update(sel)
                    else:
                        req[attr_name] = dict(sel)
                requests.append(list(req.items()))
            results = oracle.process(requests, postprocess=None)
            weights = {}
            for (a, b), x_raw in zip(candidates, results):
                x = x_raw.ravel().astype(np.float64)
                xhat = est.project((a, b), table_attrs).ravel()
                weights[a, b] = np.linalg.norm(x - xhat, 1)

            # Build MST using exponential mechanism
            T = set()
            ds = DisjointSet(all_attrs)
            r = len(all_attrs)
            eps_select = sqrt(8 * rho / 3 / (r - 1))

            for _ in range(r - 1):
                cands = [e for e in candidates if not ds.connected(*e)]
                wgts = np.array([weights[e] for e in cands])
                idx = exponential_mechanism(wgts, eps_select)
                e = cands[idx]
                T.add(e)
                ds.merge(*e)

            selected = list(T)
            logger.info(f"MST Phase 2: Selected {len(selected)} 2-way cliques")

            # Phase 3: Measure selected 2-way cliques
            logger.info("MST Phase 3: Measuring selected cliques")
            log2 = measure(oracle, table_attrs, selected, sigma)

            # Phase 4: Final model estimation
            logger.info("MST Phase 4: Final mirror descent")
            self.model = fit_pgm(
                table_attrs, log1 + log2, n,
                {**self.md_params, "max_iters": 10000},
            )
            self.table_attrs = table_attrs

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        n = n or self.n
        df = self.model.synthetic_data(n, self.table_attrs)
        return tables_to_data(
            {self.table: pd.DataFrame()}, {self.table: df},
            partition=i if self.partitions > 1 else None,
        )
