"""Adjuvant: DP structure learning + mirror descent synthetic data generation.

Combines PrivMRF-style greedy edge addition (scored by noisy pairwise TVD)
with PrivBayes-style height-chain nodes and exponential mechanism selection.
Fits clique potentials via mirror descent and samples from the junction tree.

Budget allocation: theta_1w (1-way marginals), theta_2w + sel_z (structure learning + measurement).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

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

DEFAULT_E_W1_MAX_RATIO = 0.7
DEFAULT_E_W1_MIN_RATIO = 0.25
DEFAULT_E_SEL_MAX_RATIO = 0.05
DEFAULT_E_SEL_MIN_RATIO = 0.0003
DEFAULT_THETA_1W = 40
DEFAULT_THETA_2W = 4
DEFAULT_SIZE_PENALTY = 0
DEFAULT_SEL_MAX = 50.0
DEFAULT_SEL_SAFETY_FACTOR = 3.0
DEFAULT_SEL_Z = 2.0
DEFAULT_MIN_TVD = ("auto", 0)
DEFAULT_MIN_MI = 0.005
DEFAULT_MAX_CLIQUE_SIZE = 5e5
DEFAULT_MAX_ROOT_CLIQUE_SIZE = 5e6
DEFAULT_RESCALE = True
DEFAULT_RAKE = False
DEFAULT_SCORING: Literal["mi", "tvd", "tvd_n"] = "tvd_n"


class AdjuvantMare(MareModel):
    """MARE-compatible wrapper for the Adjuvant algorithm.

    Runs the full Adjuvant pipeline (structure learning + mirror descent)
    within a single MARE model version, including hist/parent columns
    as evidence for conditioned junction tree sampling.

    dp_type controls the privacy mechanism:
      "dp"  -> Laplace noise, epsilon budget (linear composition)
      "cdp" -> Gaussian noise, rho budget (zCDP composition)
    """

    dp_type: Literal["dp", "cdp"] = "dp"
    total_params: int | None = None

    def __init__(
        self,
        *,
        rho: float = 0.0,
        etotal: float | None = None,
        e_w1_max_ratio: float = DEFAULT_E_W1_MAX_RATIO,
        e_w1_min_ratio: float = DEFAULT_E_W1_MIN_RATIO,
        e_sel_max_ratio: float | None = DEFAULT_E_SEL_MAX_RATIO,
        e_sel_min_ratio: float | None = DEFAULT_E_SEL_MIN_RATIO,
        sel_max: float = DEFAULT_SEL_MAX,
        theta_1w: float = DEFAULT_THETA_1W,
        theta_2w: float = DEFAULT_THETA_2W,
        sel_z: float = DEFAULT_SEL_Z,
        size_penalty: float = DEFAULT_SIZE_PENALTY,
        min_tvd: float | tuple(Literal["auto"], float) = DEFAULT_MIN_TVD,
        min_mi: float = DEFAULT_MIN_MI,
        sel_safety_factor: float = DEFAULT_SEL_SAFETY_FACTOR,
        max_clique_size: float = DEFAULT_MAX_CLIQUE_SIZE,
        max_root_clique_size: float = DEFAULT_MAX_ROOT_CLIQUE_SIZE,
        rescale: bool = DEFAULT_RESCALE,
        rake: bool = DEFAULT_RAKE,
        scoring: Literal["mi", "tvd", "tvd_n"] = DEFAULT_SCORING,
        max_order: int | None = 1,
        accountant: bool = True,
        mirror_descent: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        # MARE passes rho= for CDP, etotal= for DP; accept either
        self.budget = etotal if etotal is not None else rho
        self.theta_1w = theta_1w
        self.theta_2w = theta_2w
        self.sel_z = sel_z
        self.e_w1_max_ratio = e_w1_max_ratio
        self.e_w1_min_ratio = e_w1_min_ratio
        self.e_sel_max_ratio = e_sel_max_ratio
        self.e_sel_min_ratio = e_sel_min_ratio
        self.sel_max = sel_max
        self.size_penalty = size_penalty
        self.min_tvd = min_tvd
        self.min_mi = min_mi
        self.sel_safety_factor = sel_safety_factor
        self.max_clique_size = max_clique_size
        self.max_root_clique_size = max_root_clique_size
        self.rescale = rescale if accountant else False
        self.rake = rake
        self.scoring = scoring
        self.max_order = max_order
        self.md_params = mirror_descent if mirror_descent and mirror_descent != True else {}
        self.seed = seed
        self.kwargs = kwargs
    
    @classmethod
    def visualise(cls, dir: str, models: dict):
        """Render the relational chain of submodels.

        Writes ``graph.svg`` (moral graph view) and ``junction.svg``
        (junction-tree view).

        ``graph.svg`` uses nested clusters following the relational
        schema (parent table → child table) and dedupes evidence vars
        by aliasing each ctx-style hist node to the matching main node
        in its source-table submodel; cross-table structure edges then
        cross cluster boundaries naturally.  Lagged evidence
        sub-clusters use ``[-1]``, ``[-2]`` indexing, and hist nodes
        with no structure/evidence edges are hidden.
        """
        import os
        from collections import defaultdict

        import pydot

        from ....graph.hugin import get_message_passing_order
        from ....graph.utils import (
            _build_induced_graph_into,
            _build_junction_tree_into,
            display_pydot,
        )

        os.makedirs(dir, exist_ok=True)

        items = list(models.items())
        N = len(items)
        prefixes = [f"m{i}/" for i in range(N)]

        # Map table -> seq submodel index (the canonical "owner").  Seq
        # holds main rows of the table; ctx holds reduced/wider context.
        seq_owner: dict[str, int] = {}
        for i, (ver, _) in enumerate(items):
            if not ver.ctx:
                seq_owner[ver.ver.name] = i
        for i, (ver, _) in enumerate(items):  # fall back to ctx for tables w/o seq
            seq_owner.setdefault(ver.ver.name, i)

        def parent_table(ver) -> str | None:
            for p in (ver.ver.parents or ()):
                pname = getattr(p, "name", None) or getattr(
                    getattr(p, "table", None), "name", None
                )
                if pname:
                    return pname
            return None

        # Catalog of main-var rendered ids in the seq submodel of each
        # table — used for aliasing ctx-style evidence in consumer models.
        # We need a dry build to collect val_names; do a throwaway call.
        seq_val_names: dict[str, dict] = {}
        for table, src_idx in seq_owner.items():
            ver, model = items[src_idx]
            tmp = pydot.Dot(graph_type="graph")
            info = _build_induced_graph_into(
                tmp, model.moral, attrs=None, prefix=prefixes[src_idx],
            )
            seq_val_names[table] = info["val_names"]

        # Per-submodel: alias map (hist node -> external rendered id)
        # and skip set (hist nodes with no useful edges).
        alias_per_submodel: list[dict] = []
        skip_per_submodel: list[set] = []
        for i, (ver, model) in enumerate(items):
            alias: dict[str, str] = {}
            skip: set[str] = set()
            # Aliasing: ctx-style hist nodes (order=None) whose source
            # table has a seq submodel get aliased to the source's main
            # node with the matching (attr, value).
            for n, d in model.moral.nodes(data=True):
                t = d.get("table")
                if t is None:
                    continue
                if d.get("order") is not None:
                    continue
                src_idx = seq_owner.get(t)
                if src_idx is None or src_idx == i:
                    continue
                key = (None, None, d["attr"], d["value"])
                src_id = seq_val_names.get(t, {}).get(key)
                if src_id is not None:
                    alias[n] = src_id

            # Hide hist nodes that do not participate in any
            # structure/evidence edge — they are not "used in the main
            # graph".
            useful: set[str] = set()
            for u, v, ed in model.moral.edges(data=True):
                if ed.get("structure") or ed.get("evidence"):
                    useful.add(u)
                    useful.add(v)
            for n, d in model.moral.nodes(data=True):
                if d.get("table") is None:
                    continue  # always keep main vars
                if n in alias:
                    continue  # aliased -> already represented
                if n not in useful:
                    skip.add(n)
            alias_per_submodel.append(alias)
            skip_per_submodel.append(skip)

        # Build a tree of table containers based on parent_table().
        # Every table gets one outer "table_cluster"; the table cluster
        # of a child is nested inside the table cluster of its parent.
        children_by_parent: dict[str | None, list[str]] = defaultdict(list)
        seen_tables: set[str] = set()
        for ver, _ in items:
            t = ver.ver.name
            if t in seen_tables:
                continue
            seen_tables.add(t)
            children_by_parent[parent_table(ver)].append(t)

        submodels_by_table: dict[str, list[int]] = defaultdict(list)
        for i, (ver, _) in enumerate(items):
            submodels_by_table[ver.ver.name].append(i)

        # ---------- graph.svg ----------
        o = pydot.Dot(
            graph_type="digraph",
            compound="true",
            rankdir="TB",
            nodesep="0.12",
            ranksep="0.45",
            margin="0.05",
            pad="0.1",
            splines="true",
            overlap="false",
        )
        o.set_node_defaults(fontsize="9", margin="0.03,0.015")
        o.set_edge_defaults(fontsize="9", penwidth="0.8", arrowsize="0.6")

        order_label = lambda ord_: f"-{ord_ + 1}"

        def render_table(table: str, parent_container) -> None:
            tc_name = f"t_{table}"
            tc = pydot.Cluster(
                tc_name, label=table, style="rounded,filled",
                fillcolor="#fafafa", penwidth="2.5", margin="14",
                labeljust="l", fontname="bold",
            )
            parent_container.add_subgraph(tc)

            for sub_idx in submodels_by_table[table]:
                ver, model = items[sub_idx]
                sub_label = f"{ver.ver.name} ({'ctx' if ver.ctx else 'seq'})"
                sc = pydot.Cluster(
                    f"sm{sub_idx}", label=sub_label, style="rounded",
                    penwidth="1.5", margin="10", labeljust="l",
                )
                tc.add_subgraph(sc)
                _build_induced_graph_into(
                    sc, model.moral,
                    attrs=model.table_attrs,
                    prefix=prefixes[sub_idx],
                    skip_nodes=skip_per_submodel[sub_idx],
                    node_alias=alias_per_submodel[sub_idx],
                    order_label=order_label,
                )

            for child in children_by_parent.get(table, []):
                render_table(child, tc)

        for root in children_by_parent.get(None, []):
            render_table(root, o)

        display_pydot(
            o,
            edges={"labeldistance": 1.0, "labelfontsize": 8, "arrowsize": 0.6},
            out=os.path.join(dir, "graph.svg"),
        )

        # ---------- junction.html ----------
        # Render one dot graph per table (submodels of the same table
        # share a per-table figure with rank=same so ctx/seq sit side
        # by side).  Tables are stacked vertically in the output HTML
        # in parent->child dependency order, which avoids fighting
        # dot's layout to enforce cross-cluster vertical ranking.

        # Topological order over parent_table relationships, falling
        # back to insertion order for tables with no recorded parent.
        ordered_tables: list[str] = []
        seen_t: set[str] = set()
        def _emit(table: str) -> None:
            if table in seen_t or table not in submodels_by_table:
                return
            seen_t.add(table)
            ordered_tables.append(table)
            for child in children_by_parent.get(table, []):
                _emit(child)
        for root in children_by_parent.get(None, []):
            _emit(root)
        for t in submodels_by_table:  # any leftover (cycles / orphans)
            _emit(t)

        svgs: list[tuple[str, bytes]] = []
        for table in ordered_tables:
            jt = pydot.Dot(
                graph_type="digraph",
                compound="true",
                newrank="true",
                rankdir="TB",
                nodesep="0.35",
                ranksep="0.6",
                margin="0.05",
                pad="0.1",
                splines="true",
                overlap="false",
            )
            jt.set_node_defaults(fontsize="9", margin="0")
            jt.set_edge_defaults(fontsize="9", penwidth="0.8", arrowsize="0.6")

            sub_anchors: list[str] = []
            for sub_idx in submodels_by_table[table]:
                ver, model = items[sub_idx]
                sub_label = f"{ver.ver.name} ({'ctx' if ver.ctx else 'seq'})"
                sc = pydot.Cluster(
                    f"sj{sub_idx}", label=sub_label, style="rounded",
                    penwidth="1.5", margin="10", labeljust="l",
                )
                jt.add_subgraph(sc)
                messages = (
                    get_message_passing_order(model.junction)
                    if model.junction.number_of_edges() > 0
                    else None
                )
                node_id = _build_junction_tree_into(
                    sc, model.junction,
                    attrs=model.table_attrs,
                    messages=messages,
                    prefix=prefixes[sub_idx],
                )
                for cl in model.junction.nodes():
                    sub_anchors.append(node_id(cl))
                    break

            # Force submodels of this table side by side.
            if len(sub_anchors) >= 2:
                rs = pydot.Subgraph(f"row_{table}", rank="same")
                for a in sub_anchors:
                    rs.add_node(pydot.Node(a))
                jt.add_subgraph(rs)

            svg = jt.create(
                format="svg",
                prog=["dot", "-Elabeldistance=1.2", "-Elabelangle=20"],
            )
            svgs.append((table, svg))

        # Stitch the per-table SVGs into a single HTML file, each
        # under a table-name heading, stacked vertically.
        def _strip_svg(svg: bytes) -> str:
            s = svg.decode("utf-8") if isinstance(svg, bytes) else svg
            s = s.lstrip()
            if s.startswith("<?xml"):
                s = s.split("?>", 1)[1].lstrip()
            while s.startswith("<!DOCTYPE") or s.startswith("<!doctype"):
                s = s.split(">", 1)[1].lstrip()
            return s

        body_parts = [
            (
                f'<section style="margin:18px 12px">'
                f'<h2 style="font:600 14px/1.2 sans-serif;margin:0 0 8px">'
                f'{table}</h2>{_strip_svg(svg)}</section>'
            )
            for table, svg in svgs
        ]
        html = (
            '<!doctype html><html><head><meta charset="utf-8">'
            '<title>junction</title>'
            '<style>html,body{margin:0;padding:0;background:#fff;'
            'overflow:auto}svg{display:block}</style>'
            '</head><body>' + "".join(body_parts) + '</body></html>'
        )
        with open(os.path.join(dir, "junction.html"), "w") as f:
            f.write(html)

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

        all_obs, moral, bdg_remaining, tvd_diag = adjuvant_fit(
            oracle,
            attrs,
            n,
            rho=self.budget,
            theta_1w=self.theta_1w,
            theta_2w=self.theta_2w,
            sel_z=self.sel_z,
            e_w1_max_ratio=self.e_w1_max_ratio,
            e_w1_min_ratio=self.e_w1_min_ratio,
            e_sel_max_ratio=self.e_sel_max_ratio,
            e_sel_min_ratio=self.e_sel_min_ratio,
            sel_max=self.sel_max,
            size_penalty=self.size_penalty,
            min_tvd=self.min_tvd,
            min_mi=self.min_mi,
            sel_safety_factor=self.sel_safety_factor,
            frozen_nodes=frozen_nodes,
            n_hist_cols=len(hist_cols),
            max_clique_size=self.max_clique_size,
            max_root_clique_size=self.max_root_clique_size,
            rescale=self.rescale,
            rake=self.rake,
            scoring=self.scoring,
            max_order=self.max_order,
            dp_type=self.dp_type,
        )

        self.bdg_remaining = bdg_remaining
        self.all_obs = all_obs
        self.moral = moral
        self.tvd_diag = tvd_diag
        # Only evidence (frozen) vars that structure_learn actually pulled
        # into the model (via a `structure=True` edge to a main var) need
        # to share the root clique — those are the ones that already have
        # all-pairs edges among them in `moral`.  Passing the full
        # frozen_nodes set would tell build_junction_tree to add all-pairs
        # edges among unconnected hist vars, blowing up triangulation.
        selected_evidence: set[str] = set()
        for u, v, data in moral.edges(data=True):
            if not data.get("structure"):
                continue
            if u in frozen_nodes:
                selected_evidence.add(u)
            if v in frozen_nodes:
                selected_evidence.add(v)

        self.junction, self.cliques, self.potentials = adjuvant_run_md(
            all_obs,
            attrs,
            moral,
            self.md_params,
            evidence_vars=selected_evidence,
        )
        self.total_params = sum(int(p.size) for p in self.potentials)

        # Pre-compute which clique dims correspond to hist columns
        # for evidence injection during sampling
        self._hist_evidence_meta = self._build_hist_evidence_meta(attrs)

        return bdg_remaining

    def __str__(self) -> str:
        from .implementation import print_adjuvant

        return print_adjuvant(
            self.table_attrs,
            self.moral,
            rho=self.budget,
            rho_remaining=self.bdg_remaining,
            theta_1w=self.theta_1w,
            theta_2w=self.theta_2w,
            sel_z=self.sel_z,
            n_obs=len(self.all_obs),
            dp_type=self.dp_type,
            tvd_diag=self.tvd_diag,
        )

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
                    # Per-value mapping: attr.get_mapping(sel) is defined over the
                    # full combined attribute domain (product of all val domains),
                    # but raw_vals are per-value leaf indices, so it must be the
                    # per-value mapping at the requested height.
                    val_meta = attr.vals[val_name]
                    assert isinstance(val_meta, CatValue)
                    mapping = np.array(val_meta.get_mapping(h), dtype=np.int64)
                    evidence[(ci, di)] = mapping[raw_vals]
                else:
                    # Multi-value dim: build a lookup from per-value indices
                    # to compressed bin using _decompose_dim's inverse.
                    from ....graph.sample import _decompose_dim

                    comp_dom = attr.get_domain(sel)
                    all_comp = np.arange(comp_dom)
                    decomposed = _decompose_dim(attr, sel, all_comp)

                    # Build reverse lookup: per-value tuple -> comp bin
                    # Use a flat product of per-value domains as key
                    val_names_sorted = list(sel.keys())
                    val_doms = [
                        cast(CatValue, attr.vals[vn]).get_domain(sel[vn])
                        for vn in val_names_sorted
                    ]

                    # Build lookup array: flat_key -> comp_bin
                    flat_dom = 1
                    for d in val_doms:
                        flat_dom *= d
                    lookup = np.full(flat_dom, 0, dtype=np.int64)
                    for c in range(comp_dom):
                        flat_key = 0
                        mul_k = 1
                        for vi, vn in enumerate(val_names_sorted):
                            flat_key += int(decomposed[vn][c]) * mul_k
                            mul_k *= val_doms[vi]
                        if flat_key < flat_dom:
                            lookup[flat_key] = c

                    # Encode each row's per-value data into flat key
                    all_found = True
                    flat_idx = np.zeros(n, dtype=np.int64)
                    mul = 1
                    for vi, (val_name, h) in enumerate(sel.items()):
                        if val_name not in hist_df.columns:
                            all_found = False
                            break
                        raw_vals = hist_df[val_name].to_numpy()
                        val_meta = attr.vals[val_name]
                        assert isinstance(val_meta, CatValue)
                        # Map raw leaf indices through per-value mapping
                        val_mapping = np.array(
                            val_meta.get_mapping(h), dtype=np.int64
                        )
                        flat_idx += val_mapping[raw_vals] * mul
                        mul *= val_doms[vi]

                    if all_found:
                        np.clip(flat_idx, 0, flat_dom - 1, out=flat_idx)
                        evidence[(ci, di)] = lookup[flat_idx]

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
    dp_type: Literal["dp", "cdp"] = "cdp"
    total_params: int | None = None

    def __init__(
        self,
        e: float = 2.0,
        etotal: float | None = None,
        delta: float | Literal["tenth"] = "tenth",
        e_w1_max_ratio: float = DEFAULT_E_W1_MAX_RATIO,
        e_w1_min_ratio: float = DEFAULT_E_W1_MIN_RATIO,
        e_sel_max_ratio: float | None = DEFAULT_E_SEL_MAX_RATIO,
        e_sel_min_ratio: float | None = DEFAULT_E_SEL_MIN_RATIO,
        sel_max: float = DEFAULT_SEL_MAX,
        theta_1w: float = DEFAULT_THETA_1W,
        theta_2w: float = DEFAULT_THETA_2W,
        sel_z: float = DEFAULT_SEL_Z,
        size_penalty: float = DEFAULT_SIZE_PENALTY,
        min_tvd: float | Literal["auto"] = DEFAULT_MIN_TVD,
        min_mi: float = DEFAULT_MIN_MI,
        sel_safety_factor: float = DEFAULT_SEL_SAFETY_FACTOR,
        max_clique_size: float = DEFAULT_MAX_CLIQUE_SIZE,
        rescale: bool = DEFAULT_RESCALE,
        rake: bool = DEFAULT_RAKE,
        scoring: Literal["mi", "tvd", "tvd_n"] = DEFAULT_SCORING,
        rebalance: bool | dict = True,
        marginal_mode: "MarginalOracle.MODES" = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        seed: int | None = None,
        n: int | None = None,
        partitions: int | None = None,
        mirror_descent: dict | None = None,
        ablation: Literal[
            None, "1-way", "no-compression", "no-confidence", "no-cost-penalty"
        ] = None,
        **kwargs,
    ) -> None:
        if etotal is not None:
            self.e = etotal
        else:
            self.e = e
        self.delta = delta
        self.ablation = ablation
        self.theta_1w = theta_1w
        self.theta_2w = theta_2w
        self.sel_z = sel_z
        self.e_w1_max_ratio = e_w1_max_ratio
        self.e_w1_min_ratio = e_w1_min_ratio
        self.e_sel_max_ratio = e_sel_max_ratio
        self.e_sel_min_ratio = e_sel_min_ratio
        self.sel_max = sel_max
        self.size_penalty = size_penalty
        self.min_tvd = min_tvd
        self.min_mi = min_mi
        self.sel_safety_factor = sel_safety_factor
        self.max_clique_size = max_clique_size
        self.rescale = rescale
        self.rake = rake
        self.scoring = scoring
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

        if self.rebalance and self.ablation != "no-compression":
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
        from .implementation import adjuvant_fit, adjuvant_run_md, cdp_rho

        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = table.shape[0]
        self.table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        if self.dp_type == "cdp":
            if self.delta == "tenth":
                self.delta = 1.0 / (10 * n)
                logger.info(f"Resolved delta='tenth' to delta={self.delta:.2e} (n={n})")
            budget = cdp_rho(self.e, self.delta) if self.e > 0 else 0.0
        else:
            budget = self.e

        with MarginalOracle(
            data,
            self.table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            self.all_obs, self.moral, self.bdg_remaining, self.tvd_diag = adjuvant_fit(
                oracle,
                self.table_attrs,
                n,
                rho=budget,
                theta_1w=self.theta_1w,
                theta_2w=self.theta_2w,
                sel_z=self.sel_z,
                e_w1_max_ratio=self.e_w1_max_ratio,
                e_w1_min_ratio=self.e_w1_min_ratio,
                e_sel_max_ratio=self.e_sel_max_ratio,
                e_sel_min_ratio=self.e_sel_min_ratio,
                sel_max=self.sel_max,
                size_penalty=self.size_penalty,
                min_tvd=self.min_tvd,
                min_mi=self.min_mi,
                sel_safety_factor=self.sel_safety_factor,
                max_clique_size=self.max_clique_size,
                rescale=self.rescale,
                rake=self.rake,
                scoring=self.scoring,
                dp_type=self.dp_type,
                skip_structure=self.ablation == "1-way",
                no_confidence=self.ablation == "no-confidence",
                cost_penalty=self.ablation != "no-cost-penalty",
            )

        self._run_md()

    def _run_md(self):
        from .implementation import adjuvant_run_md

        self.junction, self.cliques, self.potentials = adjuvant_run_md(
            self.all_obs, self.table_attrs, self.moral, self.md_params
        )
        self.total_params = sum(int(p.size) for p in self.potentials)

    def refresh(self, **kwargs):
        if "mirror_descent" not in kwargs:
            return

        if isinstance(kwargs["mirror_descent"], dict):
            self.md_params = kwargs["mirror_descent"]

        self._run_md()

    def __str__(self) -> str:
        from .implementation import print_adjuvant, cdp_rho

        if self.dp_type == "cdp":
            budget = cdp_rho(self.e, self.delta) if self.e > 0 else 0.0
        else:
            budget = self.e
        return print_adjuvant(
            self.table_attrs,
            self.moral,
            rho=budget,
            rho_remaining=self.bdg_remaining,
            theta_1w=self.theta_1w,
            theta_2w=self.theta_2w,
            sel_z=self.sel_z,
            n_obs=len(self.all_obs),
            dp_type=self.dp_type,
            tvd_diag=self.tvd_diag,
        )

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

    def visualise(self, dir: str):
        import os

        from ....graph.hugin import get_message_passing_order
        from ....graph.utils import display_induced_graph, display_junction_tree

        os.makedirs(dir, exist_ok=True)

        display_induced_graph(
            self.moral, attrs=self.table_attrs,
            out=os.path.join(dir, "graph.svg"),
        )
        display_junction_tree(
            self.junction, self.moral,
            messages=get_message_passing_order(self.junction),
            attrs=self.table_attrs,
            out=os.path.join(dir, "junction.svg"),
        )


class AdjuvantMareEdp(AdjuvantMare):
    """AdjuvantMare using e-DP (laplace)."""

    dp_type: Literal["dp", "cdp"] = "dp"


class AdjuvantSynthEdp(AdjuvantSynth):
    """AdjuvantSynth using zCDP (Gaussian noise, rho budget)."""

    name = "adjuvant_edp"
    dp_type: Literal["dp", "cdp"] = "dp"
