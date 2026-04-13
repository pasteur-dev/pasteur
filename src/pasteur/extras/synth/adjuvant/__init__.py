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
        ew_ratio: float = 0.7,
        theta_1w: float = 50,
        theta_2w: float = 2,
        em_z: float = 4.0,
        size_penalty: float = 0,
        min_tvd: float = 0.05,
        sigma_floor: float = 1.0,
        max_clique_size: float = 1e5,
        rescale: bool = True,
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
        self.max_clique_size = max_clique_size
        self.rescale = rescale
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
    ) -> float:
        from .implementation import (
            adjuvant_fit,
            adjuvant_run_md,
            get_hist_cols,
            get_col_names,
        )

        self.table_attrs = attrs

        # Identify frozen (hist) nodes for blocking hist-hist edges
        from .implementation import build_height_chain_graph

        all_cols = get_col_names(attrs)
        hist_cols = get_hist_cols(all_cols)
        directed_graph = build_height_chain_graph(attrs)
        frozen_nodes: set[str] = set()
        for node, data in directed_graph.nodes(data=True):
            if data.get("table") is not None:
                frozen_nodes.add(node)

        all_obs, moral, rho_remaining = adjuvant_fit(
            oracle,
            attrs,
            n,
            e=self.e,
            delta=self.delta,
            theta_1w=self.theta_1w,
            theta_2w=self.theta_2w,
            em_z=self.em_z,
            ew_ratio=self.ew_ratio,
            size_penalty=self.size_penalty,
            min_tvd=self.min_tvd,
            sigma_floor=self.sigma_floor,
            frozen_nodes=frozen_nodes,
            n_hist_cols=len(hist_cols),
            max_clique_size=self.max_clique_size,
            rescale=self.rescale,
        )

        self.rho_remaining = rho_remaining
        self.junction, self.cliques, self.potentials = adjuvant_run_md(
            all_obs, attrs, moral, self.md_params
        )

        # Pre-compute which clique dims correspond to hist columns
        # for evidence injection during sampling
        self._hist_evidence_meta = self._build_hist_evidence_meta(attrs)

        return rho_remaining

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
        ew_ratio: float = 0.7,
        theta_1w: float = 50,
        theta_2w: float = 2,
        em_z: float = 4.0,
        size_penalty: float = 0,
        min_tvd: float = 0.05,
        sigma_floor: float = 5.0,
        max_clique_size: float = 1e5,
        rescale: bool = True,
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
        self.max_clique_size = max_clique_size
        self.rescale = rescale
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
        from .implementation import adjuvant_fit, adjuvant_run_md

        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = table.shape[0]
        self.table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        with MarginalOracle(
            data,
            self.table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            self.all_obs, self.moral, self.rho_remaining = adjuvant_fit(
                oracle,
                self.table_attrs,
                n,
                e=self.e,
                delta=self.delta,
                theta_1w=self.theta_1w,
                theta_2w=self.theta_2w,
                em_z=self.em_z,
                ew_ratio=self.ew_ratio,
                size_penalty=self.size_penalty,
                min_tvd=self.min_tvd,
                sigma_floor=self.sigma_floor,
                max_clique_size=self.max_clique_size,
                rescale=self.rescale,
            )

        self._run_md()

    def _run_md(self):
        from .implementation import adjuvant_run_md

        self.junction, self.cliques, self.potentials = adjuvant_run_md(
            self.all_obs, self.table_attrs, self.moral, self.md_params
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
