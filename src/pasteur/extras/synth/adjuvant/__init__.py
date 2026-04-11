"""Adjuvant: DP structure learning + mirror descent synthetic data generation.

Combines PrivMRF-style greedy edge addition (scored by noisy pairwise TVD)
with PrivBayes-style height-chain nodes and exponential mechanism selection.
Fits clique potentials via mirror descent and samples from the junction tree.

Budget allocation: e1 (scaling), e2 (structure learning), e3 (measurement + fitting).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ....attribute import Attributes, DatasetAttributes
from ....hierarchy import rebalance_attributes
from ....marginal import MarginalOracle
from ....marginal.oracle import counts_preprocess
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AdjuvantSynth(Synth):
    name = "adjuvant"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

    def __init__(
        self,
        e: float = 2.0,
        delta: float = 1e-9,
        e1_frac: float = 0.10,
        e2_frac: float = 0.20,
        e3_frac: float = 0.70,
        e2_tvd_frac: float = 0.50,
        max_clique_size: float = 1e7,
        size_penalty: float = 1e-8,
        max_no_improve: int = 3,
        rebalance: bool | dict = True,
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
        self.e1_frac = e1_frac
        self.e2_frac = e2_frac
        self.e3_frac = e3_frac
        self.e2_tvd_frac = e2_tvd_frac
        self.max_clique_size = max_clique_size
        self.size_penalty = size_penalty
        self.max_no_improve = max_no_improve
        self.rebalance = rebalance
        self.marginal_mode = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.seed = seed
        self.n = n
        self.partitions = partitions
        self.md_params = mirror_descent if mirror_descent and mirror_descent != True else {}
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyFrame]):
        self.table = next(iter(meta))
        self._n = data[self.table].shape[0]
        self._partitions = len(data[self.table])

        if self.rebalance:
            rebalance_kwargs = self.rebalance if isinstance(self.rebalance, dict) else {}
            with MarginalOracle(
                data,
                meta,
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
                preprocess=counts_preprocess,
            ) as o:
                counts = o.get_counts(desc="Calculating counts for column rebalancing")

            self.attrs = {
                k: rebalance_attributes(counts[k], v, **rebalance_kwargs)
                for k, v in meta.items()
            }
        else:
            self.attrs = meta

    @make_deterministic
    def bake(self, data: dict[str, LazyFrame]):
        pass

    @make_deterministic
    def fit(self, data: dict[str, LazyFrame]):
        from .implementation import (
            compute_all_marginals,
            add_noise_1way,
            compute_noisy_tvd,
            build_height_chain_graph,
            structure_learn,
            select_cliques_to_measure,
            measure_cliques,
            build_1way_observations,
        )
        from ..sota.common import cdp_rho, get_attr_names
        from ....graph.mirror_descent import (
            MIRROR_DESCENT_DEFAULT,
            build_junction_tree,
            mirror_descent,
            fit_model,
        )
        from ....graph.hugin import to_moral, find_elim_order, get_junction_tree

        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = self.n
        table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        # Privacy budget
        rho = cdp_rho(self.e, self.delta)
        rho1 = self.e1_frac * rho
        rho2 = self.e2_frac * rho
        rho3 = self.e3_frac * rho
        rho2_tvd = self.e2_tvd_frac * rho2
        rho2_exp = (1 - self.e2_tvd_frac) * rho2

        all_attrs = get_attr_names(table_attrs)

        with MarginalOracle(
            data,
            table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            # ==================================================
            # Step 0: Single data pass — cache all marginals
            # ==================================================
            logger.info(
                f"Adjuvant Step 0: Computing all 1-way and 2-way marginals "
                f"({len(all_attrs)} attrs)"
            )
            cached = compute_all_marginals(oracle, table_attrs, all_attrs)

            # ==================================================
            # Step 1: Noisy 1-way marginals (budget e1)
            # ==================================================
            logger.info(f"Adjuvant Step 1: Noisy 1-way marginals (rho1={rho1:.4f})")
            noisy_1way, sigma1 = add_noise_1way(cached, rho1)

            # ==================================================
            # Step 2: Structure learning (budget e2)
            # ==================================================
            logger.info(
                f"Adjuvant Step 2: Structure learning "
                f"(rho2_tvd={rho2_tvd:.4f}, rho2_exp={rho2_exp:.4f})"
            )
            tvd = compute_noisy_tvd(cached, n, rho2_tvd)
            directed_graph = build_height_chain_graph(table_attrs)
            logger.info(
                f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
                f"nodes, {directed_graph.number_of_edges()} chain edges"
            )

            moral, structure_edges, rho2_remaining = structure_learn(
                directed_graph,
                table_attrs,
                tvd,
                self.max_clique_size,
                self.size_penalty,
                self.max_no_improve,
                rho2_exp,
            )
            rho3 += rho2_remaining
            logger.info(
                f"Adjuvant: rho2 leftover {rho2_remaining:.4f} -> rho3 now {rho3:.4f}"
            )

            # ==================================================
            # Step 2e: Finalize graph — proper triangulation + junction tree
            # ==================================================
            logger.info("Adjuvant Step 2e: Triangulation and junction tree")
            _, triangulated, cost = find_elim_order(moral, table_attrs)
            logger.info(f"Adjuvant: triangulation cost={cost:_}")
            junction = get_junction_tree(triangulated, table_attrs)
            cliques = list(junction.nodes())
            logger.info(f"Adjuvant: {len(cliques)} cliques in junction tree")

            # ==================================================
            # Step 3: Measurement + fitting (budget e3)
            # ==================================================
            # Select which cliques to measure (must have structure edges)
            cliques_to_measure = select_cliques_to_measure(
                triangulated, table_attrs, structure_edges
            )
            logger.info(
                f"Adjuvant Step 3: Measuring {len(cliques_to_measure)}/{len(cliques)} "
                f"cliques (rho3={rho3:.4f})"
            )

            clique_obs, sigma3 = measure_cliques(
                oracle, cliques_to_measure, table_attrs, n, rho3
            )
            oneway_obs = build_1way_observations(noisy_1way, table_attrs, n, sigma1)
            all_obs = clique_obs + oneway_obs
            logger.info(
                f"Adjuvant: {len(clique_obs)} clique obs + {len(oneway_obs)} 1-way obs, "
                f"sigma3={sigma3:.2f}, sigma1={sigma1:.2f}"
            )

            # ==================================================
            # Mirror descent fitting
            # ==================================================
            md = {**MIRROR_DESCENT_DEFAULT, **self.md_params}
            md.pop("compress", None)
            md.pop("sample", None)
            md.pop("tree", None)
            device = md.pop("device", "auto")
            device = None if device == "auto" else device

            from ....graph.hugin import get_message_passing_order
            from ....graph.beliefs import create_messages

            generations = get_message_passing_order(junction)
            messages = create_messages(generations, table_attrs)

            logger.info(
                f"Adjuvant: Running mirror descent "
                f"(max_iters={md.get('max_iters', 1000)})"
            )
            potentials, loss_fn, raw_theta = mirror_descent(
                cliques, messages, all_obs, table_attrs,
                device=device, **md,
            )

            self.potentials = potentials
            self.junction = junction
            self.cliques = cliques
            self.table_attrs = table_attrs

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        from ....graph.sample import create_sampler_meta, sample_junction_tree

        n = n or self.n

        meta = create_sampler_meta(self.junction, self.cliques, self.table_attrs)
        columns = sample_junction_tree(self.potentials, meta, n, self.table_attrs)
        df = pd.DataFrame(columns)

        return tables_to_data(
            {self.table: pd.DataFrame()},
            {self.table: df},
            partition=i if self.partitions > 1 else None,
        )
