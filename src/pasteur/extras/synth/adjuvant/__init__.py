"""Adjuvant: DP structure learning + mirror descent synthetic data generation.

Combines PrivMRF-style greedy edge addition (scored by noisy pairwise TVD)
with PrivBayes-style height-chain nodes and exponential mechanism selection.
Fits clique potentials via mirror descent and samples from the junction tree.

Budget allocation: theta_1w (1-way marginals), theta_2w + em_z (structure learning + measurement).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ....attribute import Attributes, DatasetAttributes
from ....hierarchy import rebalance_attributes
from ....mare.synth import MareModel
from ....marginal import MarginalOracle
from ....marginal.numpy import TableSelector
from ....marginal.oracle import counts_preprocess
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AdjuvantMare(MareModel):
    """MARE-compatible wrapper for the Adjuvant algorithm.

    Runs the full Adjuvant pipeline (structure learning + mirror descent)
    within a single MARE model version, including hist/parent columns
    as evidence for conditioned junction tree sampling.
    """

    def __init__(
        self,
        *,
        etotal: float | None = None,
        e: float = 2.0,
        delta: float = 1e-9,
        theta_1w: float = 20,
        theta_2w: float = 4,
        em_z: float = 2.0,
        ew_ratio: float = 0.8,
        size_penalty: float = 0.1,
        min_tvd: float = 0.05,
        sigma_floor: float = 1.0,
        mirror_descent: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        if etotal is not None:
            self.e = etotal
        else:
            self.e = e
        self.delta = delta
        self.theta_1w = theta_1w
        self.theta_2w = theta_2w
        self.em_z = em_z
        self.ew_ratio = ew_ratio
        self.size_penalty = size_penalty
        self.min_tvd = min_tvd
        self.sigma_floor = sigma_floor
        self.md_params = mirror_descent if mirror_descent else {}
        self.seed = seed
        self.kwargs = kwargs

    @make_deterministic
    def fit(
        self,
        n: int,
        table: str | None,
        attrs: DatasetAttributes,
        oracle: MarginalOracle,
    ):
        from .implementation import (
            compute_all_marginals,
            compute_1way_budget,
            add_noise_1way,
            compute_tvd,
            build_height_chain_graph,
            structure_learn,
            measure_edges,
            build_1way_observations,
            cdp_rho,
            get_col_names,
            get_hist_cols,
        )

        self.table_attrs = attrs

        # Privacy budget (e <= 0 means no DP)
        rho = cdp_rho(self.e, self.delta) if self.e > 0 else 0.0

        all_cols = get_col_names(attrs)
        hist_cols = get_hist_cols(all_cols)
        d = len(all_cols)
        h = len(hist_cols)

        # ==================================================
        # Step 0: Compute all 1-way and 2-way marginals
        # ==================================================
        logger.info(
            f"AdjuvantMare Step 0: Computing marginals "
            f"({d} cols, {h} hist)"
        )
        cached = compute_all_marginals(oracle, attrs, all_cols)

        # ==================================================
        # Step 1: Noisy 1-way marginals (budget from theta_1w)
        # ==================================================
        rho1_max = self.ew_ratio * rho if rho > 0 else None
        sigmas_1w, rho1, eff_theta_1w = compute_1way_budget(
            cached, n, self.theta_1w, self.sigma_floor, rho1_max
        )
        logger.info(
            f"AdjuvantMare Step 1: Noisy 1-way marginals "
            f"(theta_1w={eff_theta_1w:.1f}, rho1={rho1:.6f})"
        )
        noisy_1way = add_noise_1way(cached, sigmas_1w)

        # ==================================================
        # Step 2: Structure learning + measurement (remaining budget)
        # ==================================================
        rho_avail = rho - rho1
        logger.info(
            f"AdjuvantMare Step 2: Structure learning "
            f"(rho_avail={rho_avail:.6f}, em_z={self.em_z}, theta_2w={self.theta_2w})"
        )
        tvd = compute_tvd(cached)
        directed_graph = build_height_chain_graph(attrs)
        logger.info(
            f"AdjuvantMare: height-chain graph has "
            f"{directed_graph.number_of_nodes()} nodes, "
            f"{directed_graph.number_of_edges()} chain edges"
        )

        # Identify frozen (hist) nodes in the graph — block hist-hist edges
        frozen_nodes: set[str] = set()
        for node, data in directed_graph.nodes(data=True):
            if data.get("table") is not None:
                frozen_nodes.add(node)

        moral, structure_edges, rho_remaining = structure_learn(
            directed_graph,
            attrs,
            tvd,
            n,
            self.size_penalty,
            rho_avail,
            self.min_tvd,
            em_z=self.em_z,
            theta_2w=self.theta_2w,
            sigma_floor=self.sigma_floor,
            frozen_nodes=frozen_nodes,
            n_hist_cols=h,
        )

        # ==================================================
        # Step 3: Measure edge marginals (per-edge sigma from theta_2w)
        # ==================================================
        logger.info(
            f"AdjuvantMare Step 3: Measuring {len(structure_edges)} edge "
            f"marginals (theta_2w={self.theta_2w}, rho_remaining={rho_remaining:.6f})"
        )
        edge_obs, max_sigma = measure_edges(
            oracle, structure_edges, moral, attrs, n, self.theta_2w, self.sigma_floor,
            rho_extra=rho_remaining,
        )
        oneway_obs = build_1way_observations(noisy_1way, attrs, n, sigmas_1w, self.sigma_floor)
        all_obs = edge_obs + oneway_obs

        # ==================================================
        # Build junction tree and run mirror descent
        # ==================================================
        from ....graph.mirror_descent import (
            MIRROR_DESCENT_DEFAULT,
            build_junction_tree,
            mirror_descent,
        )
        from ....graph.hugin import get_clique_domain

        md = {**MIRROR_DESCENT_DEFAULT, **self.md_params}
        md.pop("compress", None)
        md.pop("sample", None)
        tree_mode = md.pop("tree", "hugin")
        elim_factor_cost = md.pop("elim_factor_cost", 1)
        elim_max_attempts = md.pop("elim_max_attempts", 5000)
        device = md.pop("device", "auto")
        device = None if device == "auto" else device

        mg = moral if tree_mode != "maximal" else None
        self.junction, self.cliques, messages = build_junction_tree(
            all_obs,
            attrs,
            tree_mode=tree_mode,
            moral_graph=mg,
            elim_factor_cost=elim_factor_cost,
            elim_max_attempts=elim_max_attempts,
        )
        total_params = sum(
            get_clique_domain(cl, attrs) for cl in self.cliques
        )
        logger.info(
            f"AdjuvantMare: junction tree has {len(self.cliques)} cliques, "
            f"{total_params:_} parameters"
        )

        self.potentials, _, _ = mirror_descent(
            self.cliques,
            messages,
            all_obs,
            attrs,
            device=device,
            **md,
        )

        # Pre-compute which clique dims correspond to hist columns
        # for evidence injection during sampling
        self._hist_evidence_meta = self._build_hist_evidence_meta(attrs)

    def _build_hist_evidence_meta(
        self, attrs: DatasetAttributes
    ) -> list[tuple[int, int, TableSelector, str, str]]:
        """Identify clique dims that correspond to hist columns.

        Returns list of (clique_idx, dim_idx, table_selector, attr_name, val_name)
        for each hist dim found in the junction tree.
        """
        from ....attribute import CatValue
        from ....graph.beliefs import convert_sel
        from ....graph.hugin import get_attrs

        meta = []
        for ci, clique in enumerate(self.cliques):
            for di, a_meta in enumerate(clique):
                if a_meta.table is None:
                    continue  # main table dim, not hist
                sel = convert_sel(a_meta.sel)
                if isinstance(sel, int):
                    continue  # common-only dim, skip
                # This is a hist dim — record how to look up values
                table_sel: TableSelector = (
                    (a_meta.table, a_meta.order)
                    if a_meta.order is not None
                    else a_meta.table
                )
                for val_name, h in sel.items():
                    if h == 0:
                        meta.append((ci, di, table_sel, a_meta.attr, val_name))
        return meta

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame:
        from ....attribute import CatValue
        from ....graph.beliefs import convert_sel
        from ....graph.hugin import get_attrs
        from ....graph.sample import create_sampler_meta, sample_junction_tree

        n = len(index)
        sampler_meta = create_sampler_meta(
            self.junction, self.cliques, self.table_attrs
        )

        # Build evidence: map hist column values to compressed clique dim indices
        evidence: dict[tuple[int, int], np.ndarray] = {}
        for ci, clique in enumerate(self.cliques):
            for di, a_meta in enumerate(clique):
                if a_meta.table is None:
                    continue

                sel = convert_sel(a_meta.sel)
                if isinstance(sel, int):
                    # Common-only dim: derive from a sibling value in hist
                    attr = get_attrs(self.table_attrs, a_meta.table, a_meta.order)[
                        a_meta.attr
                    ]
                    cmn = attr.common
                    assert cmn is not None
                    table_sel: TableSelector = (
                        (a_meta.table, a_meta.order)
                        if a_meta.order is not None
                        else a_meta.table
                    )
                    if table_sel not in hist:
                        continue

                    hist_df = hist[table_sel]
                    # Find a sibling value column in hist to derive common
                    mapping = None
                    for sib_name, sib_val in attr.vals.items():
                        if (
                            sib_name in hist_df.columns
                            and isinstance(sib_val, CatValue)
                        ):
                            raw_vals = hist_df[sib_name].to_numpy()
                            mapping = np.array(
                                cmn.get_mapping(sel), dtype=np.int64
                            )
                            sib_mapping = np.array(
                                sib_val.get_mapping(sib_val.height - 1),
                                dtype=np.int64,
                            )
                            # raw -> sib leaf -> cmn leaf -> cmn compressed
                            evidence[(ci, di)] = mapping[sib_mapping[raw_vals]]
                            break
                    continue

                # Multi-value or single-value hist dim
                table_sel = (
                    (a_meta.table, a_meta.order)
                    if a_meta.order is not None
                    else a_meta.table
                )
                if table_sel not in hist:
                    continue

                hist_df = hist[table_sel]
                attr = get_attrs(self.table_attrs, a_meta.table, a_meta.order)[
                    a_meta.attr
                ]

                if len(sel) == 1:
                    # Single-value dim: raw value -> compressed index
                    val_name, h = next(iter(sel.items()))
                    if val_name not in hist_df.columns:
                        continue
                    raw_vals = hist_df[val_name].to_numpy()
                    val_meta = attr[val_name]
                    assert isinstance(val_meta, CatValue)
                    mapping = np.array(val_meta.get_mapping(h), dtype=np.int64)
                    evidence[(ci, di)] = mapping[raw_vals]
                else:
                    # Multi-value dim: combine per-value mappings
                    comp_mapping = attr.get_mapping(sel)
                    # Build flat index from raw values
                    flat_idx = np.zeros(n, dtype=np.int64)
                    mul = 1
                    for val_name, h in sel.items():
                        if val_name not in hist_df.columns:
                            break
                        raw_vals = hist_df[val_name].to_numpy()
                        val_meta = attr[val_name]
                        assert isinstance(val_meta, CatValue)
                        flat_idx += raw_vals.astype(np.int64) * mul
                        mul *= val_meta.get_domain(0)
                    else:
                        # Map raw flat index -> compressed index
                        evidence[(ci, di)] = comp_mapping[flat_idx]

        columns = sample_junction_tree(
            self.potentials,
            sampler_meta,
            n,
            self.table_attrs,
            evidence=evidence if evidence else None,
        )
        return pd.DataFrame(columns, index=index)


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
        etotal: float | None = None,
        delta: float = 1e-9,
        theta_1w: float = 50,
        theta_2w: float = 4,
        em_z: float = 2.0,
        ew_ratio: float = 0.8,
        size_penalty: float = 0,
        min_tvd: float = 0.05,
        sigma_floor: float = 1.0,
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
        if etotal is not None:
            self.e = etotal
        else:
            self.e = e
        self.delta = delta
        self.theta_1w = theta_1w
        self.theta_2w = theta_2w
        self.em_z = em_z
        self.ew_ratio = ew_ratio
        self.size_penalty = size_penalty
        self.min_tvd = min_tvd
        self.sigma_floor = sigma_floor
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
            compute_1way_budget,
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

        # Privacy budget (e <= 0 means no DP)
        rho = cdp_rho(self.e, self.delta) if self.e > 0 else 0.0

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
            # Step 1: Noisy 1-way marginals (budget from theta_1w)
            # ==================================================
            rho1_max = self.ew_ratio * rho if rho > 0 else None
            sigmas_1w, rho1, eff_theta_1w = compute_1way_budget(
                cached, n, self.theta_1w, self.sigma_floor, rho1_max
            )
            logger.info(
                f"Adjuvant Step 1: Noisy 1-way marginals "
                f"(theta_1w={eff_theta_1w:.1f}, rho1={rho1:.6f})"
            )
            noisy_1way = add_noise_1way(cached, sigmas_1w)

            # ==================================================
            # Step 2: Structure learning + measurement (remaining budget)
            # ==================================================
            rho_avail = rho - rho1
            logger.info(
                f"Adjuvant Step 2: Structure learning "
                f"(rho_avail={rho_avail:.6f}, em_z={self.em_z}, theta_2w={self.theta_2w})"
            )
            tvd = compute_tvd(cached)
            directed_graph = build_height_chain_graph(table_attrs)
            logger.info(
                f"Adjuvant: height-chain graph has {directed_graph.number_of_nodes()} "
                f"nodes, {directed_graph.number_of_edges()} chain edges"
            )

            moral, structure_edges, rho_remaining = structure_learn(
                directed_graph,
                table_attrs,
                tvd,
                n,
                self.size_penalty,
                rho_avail,
                self.min_tvd,
                em_z=self.em_z,
                theta_2w=self.theta_2w,
                sigma_floor=self.sigma_floor,
            )

            # ==================================================
            # Step 3: Measure edge marginals (per-edge sigma from theta_2w)
            # ==================================================
            logger.info(
                f"Adjuvant Step 3: Measuring {len(structure_edges)} edge marginals "
                f"(theta_2w={self.theta_2w}, rho_remaining={rho_remaining:.6f})"
            )

            edge_obs, max_sigma = measure_edges(
                oracle, structure_edges, moral, table_attrs, n, self.theta_2w, self.sigma_floor
            )
            oneway_obs = build_1way_observations(noisy_1way, table_attrs, n, sigmas_1w, self.sigma_floor)
            self.all_obs = edge_obs + oneway_obs
            self.moral = moral
            self.table_attrs = table_attrs
            logger.info(
                f"Adjuvant: {len(edge_obs)} edge obs + {len(oneway_obs)} 1-way obs, "
                f"max_sigma_2w={max_sigma:.2f}"
            )

        self._run_md()

    def _run_md(self):
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
        elim_max_attempts = md.pop("elim_max_attempts", 5000)

        logger.info(f"Adjuvant: building junction tree (mode={tree_mode})")
        mg = self.moral if tree_mode != "maximal" else None
        self.junction, self.cliques, messages = build_junction_tree(
            self.all_obs,
            self.table_attrs,
            tree_mode=tree_mode,
            moral_graph=mg,
            elim_factor_cost=elim_factor_cost,
            elim_max_attempts=elim_max_attempts,
        )
        total_params = sum(
            get_clique_domain(cl, self.table_attrs) for cl in self.cliques
        )
        logger.info(
            f"Adjuvant: junction tree has {len(self.cliques)} cliques, "
            f"{total_params:_} parameters"
        )

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
        self.potentials, *_ = mirror_descent(
            self.cliques,
            messages,
            self.all_obs,
            self.table_attrs,
            device=device,
            **md,
        )

    def refresh(self, **kwargs):
        if "mirror_descent" not in kwargs:
            return

        if isinstance(kwargs["mirror_descent"], dict):
            self.md_params = kwargs["mirror_descent"]

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
