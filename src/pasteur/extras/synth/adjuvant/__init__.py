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
        e1_frac: float = 0.2,
        e2_frac: float = 0.1,
        e3_frac: float = 0.7,
        size_penalty: float = 0.1,
        min_tvd: float = 0.05,
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
        self.size_penalty = size_penalty
        self.min_tvd = min_tvd
        self.rebalance = rebalance
        self.marginal_mode = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.seed = seed
        self.n = n
        self.partitions = partitions
        self.md_params = (
            mirror_descent if mirror_descent and mirror_descent != True else {}
        )
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyFrame]):
        self.table = next(iter(meta))
        self._n = data[self.table].shape[0]
        self._partitions = len(data[self.table])

        if self.rebalance:
            rebalance_kwargs = (
                self.rebalance if isinstance(self.rebalance, dict) else {}
            )
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
            compute_tvd,
            build_height_chain_graph,
            structure_learn,
            measure_edges,
            build_1way_observations,
            cdp_rho,
            get_col_names,
        )

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

        all_cols = get_col_names(table_attrs)
        d = len(all_cols)

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
                f"({d} cols)"
            )
            cached = compute_all_marginals(oracle, table_attrs, all_cols)

            # ==================================================
            # Step 1: Noisy 1-way marginals (budget e1)
            # ==================================================
            logger.info(f"Adjuvant Step 1: Noisy 1-way marginals (rho1={rho1:.4f})")
            noisy_1way, sigma1 = add_noise_1way(cached, rho1)

            # ==================================================
            # Step 2: Structure learning (budget e2)
            # ==================================================
            logger.info(f"Adjuvant Step 2: Structure learning (rho2={rho2:.4f})")
            tvd = compute_tvd(cached)
            directed_graph = build_height_chain_graph(table_attrs)
            logger.info(
                f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
                f"nodes, {directed_graph.number_of_edges()} chain edges"
            )

            moral, structure_edges, rho2_remaining = structure_learn(
                directed_graph,
                table_attrs,
                tvd,
                n,
                self.size_penalty,
                rho2,
                self.min_tvd,
            )
            rho3 += rho2_remaining
            logger.info(
                f"Adjuvant: rho2 leftover {rho2_remaining:.4f} -> rho3 now {rho3:.4f}"
            )

            # ==================================================
            # Step 3: Measure 2-way marginals for structure-learning edges
            # ==================================================
            logger.info(
                f"Adjuvant Step 3: Measuring {len(structure_edges)} edge marginals "
                f"(rho3={rho3:.4f})"
            )

            edge_obs, sigma3 = measure_edges(
                oracle, structure_edges, moral, table_attrs, n, rho3
            )
            oneway_obs = build_1way_observations(noisy_1way, table_attrs, n, sigma1)
            self.all_obs = edge_obs + oneway_obs
            self.moral = moral
            self.table_attrs = table_attrs
            logger.info(
                f"Adjuvant: {len(edge_obs)} edge obs + {len(oneway_obs)} 1-way obs, "
                f"sigma3={sigma3:.2f}, sigma1={sigma1:.2f}"
            )

        self._build_jt()
        self._run_md()

    def _build_jt(self):
        # ==================================================
        # Build junction tree
        # ==================================================
        from ....graph.hugin import get_clique_domain
        from ....graph.mirror_descent import build_junction_tree
        from ....graph.mirror_descent import MIRROR_DESCENT_DEFAULT

        md = {**MIRROR_DESCENT_DEFAULT, **self.md_params}
        md.pop("compress", None)
        md.pop("sample", None)
        tree_mode = md.pop("tree", "hugin")
        elim_factor_cost = md.pop("elim_factor_cost", 1)

        logger.info(f"Adjuvant: building junction tree (mode={tree_mode})")
        mg = self.moral if tree_mode != "maximal" else None
        self.junction, self.cliques, self.messages = build_junction_tree(
            self.all_obs,
            self.table_attrs,
            tree_mode=tree_mode,
            moral_graph=mg,
            elim_factor_cost=elim_factor_cost,
        )
        total_params = sum(
            get_clique_domain(cl, self.table_attrs) for cl in self.cliques
        )
        logger.info(
            f"Adjuvant: junction tree has {len(self.cliques)} cliques, "
            f"{total_params:_} parameters"
        )

    def _run_md(self):
        # ==================================================
        # Mirror descent fitting
        # ==================================================
        from ....graph.mirror_descent import MIRROR_DESCENT_DEFAULT, mirror_descent

        md = {**MIRROR_DESCENT_DEFAULT, **self.md_params}
        device = md.pop("device", "auto")
        device = None if device == "auto" else device

        logger.info(
            f"Adjuvant: Running mirror descent "
            f"(max_iters={md.get('max_iters', 1000)})"
        )
        potentials, loss_fn, raw_theta = mirror_descent(
            self.cliques,
            self.messages,
            self.all_obs,
            self.table_attrs,
            device=device,
            **md,
        )

        self.potentials = potentials
        self.table_attrs = self.table_attrs

    def refresh(self, **kwargs):
        if "mirror_descent" in kwargs and isinstance(kwargs["mirror_descent"], dict):
            self.md_params = kwargs["mirror_descent"]
        if "mirror_descent" in kwargs:
            self._build_jt()
            self._run_md()
        elif "run_md" in kwargs:
            self._run_md()

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
