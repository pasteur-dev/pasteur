"""AIM: Adaptive and Iterative Mechanism for DP Synthetic Data.

Iteratively selects worst-approximated marginals via exponential mechanism,
measures them, and refits the PGM model. Uses our mirror descent + BP engine.

Reference: private-pgm-ref/mechanisms/aim.py
"""

from __future__ import annotations

import itertools
import logging
from math import sqrt
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from ....attribute import Attributes, DatasetAttributes
from ....marginal import MarginalOracle
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data
from .common import cdp_rho, measure, fit_pgm, exponential_mechanism, _build_request

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _compile_workload(
    workload: list[tuple[tuple[str, ...], float]],
) -> dict[tuple[str, ...], float]:
    """Compute downward closure with scores."""
    weights = {cl: wt for cl, wt in workload}

    def powerset(s):
        return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(1, len(s) + 1)
        )

    all_cliques = set()
    for cl, _ in workload:
        all_cliques.update(powerset(cl))

    return {
        cl: sum(wt * len(set(cl) & set(wcl)) for wcl, wt in workload)
        for cl in sorted(all_cliques, key=len)
    }


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
        rounds: int = 50,
        max_model_size: float = 80,
        degree: int = 2,
        max_cells: int = 10000,
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
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.degree = degree
        self.max_cells = max_cells
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
        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = self.n
        table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        rho = cdp_rho(self.e, self.delta)
        all_attrs = list(cast(Attributes, table_attrs[None]).keys())
        num_attrs = len(all_attrs)
        rounds = self.rounds or 16 * num_attrs

        # Build workload
        workload_cliques = list(itertools.combinations(all_attrs, self.degree))
        # Filter by domain size
        from .common import clique_domain_size

        workload_cliques = [
            cl
            for cl in workload_cliques
            if clique_domain_size(cl, table_attrs) <= self.max_cells
        ]
        workload = [(cl, 1.0) for cl in workload_cliques]
        candidates = _compile_workload(workload)

        # Budget allocation: 90% measurements, 10% selection
        sigma = sqrt(rounds / (2 * 0.9 * rho))
        epsilon = sqrt(8 * 0.1 * rho / rounds)

        with MarginalOracle(
            data,
            table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            # Precompute true answers for all candidates (single batched call)
            candidate_list = list(candidates.keys())
            logger.info(
                f"AIM: Computing answers for {len(candidate_list)} candidates"
            )
            requests = [_build_request(cl, table_attrs) for cl in candidate_list]
            results = oracle.process(requests, postprocess=None)
            answers = {
                cl: r.ravel().astype(np.float64)
                for cl, r in zip(candidate_list, results)
            }

            # Phase 1: Measure all 1-way marginals
            oneway = [cl for cl in candidates if len(cl) == 1]
            logger.info(
                f"AIM: Measuring {len(oneway)} 1-way marginals (sigma={sigma:.2f})"
            )
            measurements = measure(oracle, table_attrs, oneway, sigma)
            rho_used = len(oneway) * 0.5 / sigma**2

            # Initial model
            model = fit_pgm(
                table_attrs,
                measurements,
                n,
                {**self.md_params, "max_iters": 1000},
            )

            # Adaptive loop
            t = 0
            terminate = False
            while not terminate:
                t += 1
                remaining = rho - rho_used
                if remaining <= 0:
                    break
                if remaining < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                    # Final round: use remaining budget
                    sigma = sqrt(1 / (2 * 0.9 * remaining))
                    epsilon = sqrt(8 * 0.1 * remaining)
                    terminate = True

                rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
                logger.info(f"AIM round {t}: budget {rho_used:.4f}/{rho:.4f}")

                # Filter candidates by model size
                size_limit = self.max_model_size * rho_used / rho
                fitted_cliques = [m[0] for m in measurements]
                # Downward closure of fitted cliques (free)
                free = set()
                for cl in fitted_cliques:
                    for r in range(1, len(cl) + 1):
                        free.update(itertools.combinations(cl, r))

                small_candidates = {}
                for cl, score in candidates.items():
                    if cl in free:
                        small_candidates[cl] = score
                        continue
                    new_size = sum(
                        clique_domain_size(c, table_attrs) * 8 / 1e6
                        for c in fitted_cliques + [cl]
                    )
                    if new_size <= size_limit:
                        small_candidates[cl] = score

                if not small_candidates:
                    logger.warning("AIM: No viable candidates, terminating")
                    break

                # Select worst-approximated clique
                errors = {}
                for cl in small_candidates:
                    wgt = small_candidates[cl]
                    x = answers[cl]
                    bias = sqrt(2 / np.pi) * sigma * len(x)
                    xest = model.project(cl, table_attrs).ravel()
                    xest_scaled = (
                        xest / xest.sum() * x.sum() if xest.sum() > 0 else xest
                    )
                    errors[cl] = wgt * (
                        np.linalg.norm(x - xest_scaled, 1) - bias
                    )

                max_sens = max(abs(candidates[cl]) for cl in small_candidates)
                cl = exponential_mechanism(errors, epsilon, max_sens)
                logger.info(f"AIM: Selected {cl}")

                # Measure
                new_meas = measure(oracle, table_attrs, [cl], sigma)
                measurements.extend(new_meas)

                # Refit with warm start from previous model
                model = fit_pgm(
                    table_attrs,
                    measurements,
                    n,
                    {**self.md_params, "max_iters": 1000},
                    prev_model=model,
                )

                # Adaptive sigma reduction
                if not terminate:
                    w = model.project(cl, table_attrs).ravel()
                    dom_size = len(answers[cl])
                    if (
                        np.linalg.norm(w - answers[cl], 1)
                        <= sigma * sqrt(2 / np.pi) * dom_size
                    ):
                        logger.info(
                            f"AIM: Reducing sigma {sigma:.2f} -> {sigma / 2:.2f}"
                        )
                        sigma /= 2
                        epsilon *= 2

            # Final model
            logger.info(f"AIM: Final fit with {len(measurements)} measurements")
            self.model = fit_pgm(
                table_attrs,
                measurements,
                n,
                {**self.md_params, "max_iters": 10000},
            )
            self.table_attrs = table_attrs

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        n = n or self.n
        df = self.model.synthetic_data(n, self.table_attrs)
        return tables_to_data(
            {self.table: pd.DataFrame()}, {self.table: df}
        )
