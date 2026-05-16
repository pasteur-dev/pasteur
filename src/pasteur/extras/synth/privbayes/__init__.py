from __future__ import annotations
from itertools import chain

import logging
from math import ceil
from typing import Any, Sequence, cast

import pandas as pd
import numpy as np

from ....attribute import Attributes, CatValue, DatasetAttributes, SeqAttributes
from ....hierarchy import rebalance_attributes
from ....mare.synth import MareModel
from ....marginal import MarginalOracle, PostprocessFun, PreprocessFun
from ....marginal.numpy import TableSelector
from ....marginal.oracle import counts_preprocess
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data
from .implementation import (
    MAX_EPSILON,
    Node,
    calc_noisy_marginals,
    greedy_bayes,
    print_tree,
    sample_rows,
)

logger = logging.getLogger(__name__)

from ....graph.mirror_descent import MirrorDescentParams, MIRROR_DESCENT_DEFAULT


def _noisy_1way_counts(
    oracle: MarginalOracle,
    ep: float,
    unbounded_dp: bool,
    desc: str = "Calculating 1-way counts for DP rebalancing",
) -> dict[TableSelector, dict[str, np.ndarray]]:
    """Compute Laplace-noised 1-way counts via the oracle.

    Returns a `{table_sel: {val_name: counts}}` dict matching
    `MarginalOracle.get_counts`. Total DP cost is `ep`, split evenly
    across all columns in `attrs[None]` only — hist/parent columns are
    returned without added noise (they were already noised under their
    own preprocess) but we still pass them through so the rebalance
    machinery can use them as common-value lookups.

    Sensitivity is 1 (unbounded_dp) or 2 (bounded_dp). With sequential
    composition for `d` per-column Laplace queries each at scale `b`,
    total epsilon = sens * d / b ⇒ pick `b = sens * d / ep`.
    """
    raw = oracle.get_counts(desc=desc)
    main = raw.get(None, {})
    out: dict[TableSelector, dict[str, np.ndarray]] = {k: v for k, v in raw.items()}
    if not main or ep <= 0:
        return out

    sens = 1 if unbounded_dp else 2
    d = max(len(main), 1)
    scale = sens * d / ep
    noisy_main: dict[str, np.ndarray] = {}
    for name, count in main.items():
        noise = np.random.laplace(scale=scale, size=count.shape)
        noisy_main[name] = count + noise
    out[None] = noisy_main
    return out


def _rebalance_with_noisy_counts(
    table_attrs: Attributes,
    noisy_counts: dict[str, np.ndarray],
    rebalance_kwargs: dict,
) -> Attributes:
    """Run `rebalance_attributes` against pre-noised 1-way counts.

    Skips synthetic attrs (those containing a `GenerationValue`) — those
    are MARE-injected generation count attributes and have no
    meaningful counts to rebalance against.
    """
    from ....attribute import GenerationValue

    for attr in table_attrs.values():
        for val in attr.vals.values():
            if isinstance(val, GenerationValue):
                return table_attrs
    if not noisy_counts:
        return table_attrs
    return rebalance_attributes(noisy_counts, table_attrs, **rebalance_kwargs)


def _fit_mirror_descent(
    mirror_descent: MirrorDescentParams | bool,
    nodes: Sequence[Node],
    attrs: DatasetAttributes,
    marginals: Sequence[np.ndarray],
    n: int,
    d: int,
    e2: float,
    unbounded_dp: bool,
):
    from ....graph.hugin import to_moral
    from ....graph.mirror_descent import fit_model

    md = mirror_descent if isinstance(mirror_descent, dict) else {}
    params = {**MIRROR_DESCENT_DEFAULT, **md}
    compress = params.pop("compress", True)
    md_sample = params.pop("sample", False)
    tree_mode = params.pop("tree", "hugin")
    device = None if params["device"] == "auto" else params["device"]
    params.pop("device", None)

    # Build observations (privbayes-specific: nodes → LinearObservation)
    noise_scale = (1 if unbounded_dp else 2) * d / e2
    obs = derive_obs_from_model(nodes, attrs, marginals, n, noise_scale)

    # Build moral graph for hugin modes (privbayes-specific: nodes → graph)
    mg = None
    if tree_mode != "maximal":
        g = derive_graph_from_nodes(nodes, attrs, prune=True)
        mg = to_moral(g)

    # Generic pipeline: build junction tree + fit via mirror descent
    potentials, junction, cliques, messages, loss_fn, _ = fit_model(
        obs, attrs,
        tree_mode=tree_mode,
        compress=compress,
        moral_graph=mg,
        device=device,
        **params,
    )

    # Project clique potentials directly to node marginals.
    # Each node's attributes are a subset of some clique's attributes.
    # We marginalize (sum out) extra clique dims and remap indices
    # to the node's native shape. No lossy compress/decompress roundtrip.
    from ....graph.hugin import AttrMeta, get_attrs as _get_attrs, get_clique_domain
    from ....graph.loss import get_smallest_parent, get_parent_meta, get_parents
    from ....graph.beliefs import convert_sel

    md_marginals = list(marginals)
    for idx, node in enumerate(nodes):
        orig_marg = marginals[idx]
        o = obs[idx]

        # Reuse observation's source (same AttrMeta as derive_obs_from_model)
        source = o.source

        # Rebuild orig order for transpose (node's native dim ordering)
        orig = []
        used_parent_table = None
        used_parent_order = None
        used_parent = False
        for s in node.p:
            if len(s) == 3:
                table_sel, attr_name, sel = s
            else:
                table_sel = None
                attr_name, sel = s
            if isinstance(table_sel, tuple):
                table, order = table_sel[0], table_sel[1]
            else:
                table, order = table_sel, None
            if isinstance(sel, int):
                orig.append((table, order, attr_name, None, sel))
            else:
                attr = _get_attrs(attrs, table, order)[attr_name]
                cmn = attr.common.name if attr.common else None
                for val, h in sel.items():
                    if val == cmn:
                        continue
                    orig.append((table, order, attr_name, val, h))
                if node.attr == attr_name and not any(
                    val == node.value for val, _ in sel.items() if val != cmn
                ):
                    used_parent = True
                    used_parent_table = table
                    used_parent_order = order
        if used_parent:
            orig.append((used_parent_table, used_parent_order, node.attr, node.value, 0))
        else:
            orig.append((None, None, node.attr, node.value, 0))

        # Find the parent clique and get projection metadata
        parents = get_parents(source, cliques)
        if not parents:
            # No compatible clique — keep original marginal
            logger.warning(
                f"Node {idx} ({node.attr}.{node.value}): no parent clique found, "
                f"keeping original marginal"
            )
            continue
        parent = min(parents, key=lambda x: get_clique_domain(x, attrs))
        parent_idx = cliques.index(parent)
        meta = get_parent_meta(source, parent, attrs)

        # Start with fitted clique marginal (probability space)
        proc = potentials[parent_idx].copy()

        # Sum out dimensions not in the node
        if meta.sum_dims:
            proc = np.sum(proc, axis=meta.sum_dims)

        # Remap indices from clique domain to node's source domain
        if meta.idx is not None:
            proc = proc.transpose(meta.transpose)
            og_shape = proc.shape
            a_idx_dom = 1
            for dd in og_shape[: len(meta.b_doms)]:
                a_idx_dom *= dd
            b_idx_dom = 1
            for dd in meta.b_doms:
                b_idx_dom *= dd
            rest_dom = 1
            for dd in og_shape[len(meta.b_doms) :]:
                rest_dom *= dd
            proc = proc.reshape((a_idx_dom, -1))
            out = np.zeros((b_idx_dom, rest_dom), dtype=proc.dtype)
            np.add.at(out, meta.idx, proc)
            new_shape = list(meta.b_doms) + list(og_shape[len(meta.b_doms) :])
            proc = out.reshape(new_shape).transpose(meta.transpose_undo)

        # proc is now in source shape (one compressed dim per attribute).
        # Expand each compressed attr dim to per-value naive dims.
        vals_order = []
        per_val_shape = []
        for dim_i, a in enumerate(source):
            if isinstance(a.sel, int):
                attr = _get_attrs(attrs, a.table, a.order)[a.attr]
                per_val_shape.append(attr.common.get_domain(a.sel))
                vals_order.append((a.table, a.order, a.attr, None, a.sel))
            else:
                attr = _get_attrs(attrs, a.table, a.order)[a.attr]
                naive_dom = 1
                for val, h in a.sel:
                    naive_dom *= attr[val].get_domain(h)
                compressed_dom = attr.get_domain(dict(a.sel))

                if naive_dom != compressed_dom:
                    # Expand compressed dim using index mappings.
                    # Compression sums naive→compressed; expansion must
                    # divide by group size to avoid duplicating mass.
                    naive_mapping = attr.get_naive_mapping(dict(a.sel))
                    comp_mapping = attr.get_mapping(dict(a.sel))

                    _, unique_idx = np.unique(naive_mapping, return_index=True)
                    naive_per_comp_dedup = np.zeros(
                        compressed_dom, dtype=np.float32
                    )
                    np.add.at(naive_per_comp_dedup, comp_mapping[unique_idx], 1.0)
                    naive_per_comp_dedup = np.maximum(naive_per_comp_dedup, 1.0)

                    i_map = tuple(
                        naive_mapping if j == dim_i else slice(None)
                        for j in range(len(proc.shape))
                    )
                    o_map = tuple(
                        comp_mapping if j == dim_i else slice(None)
                        for j in range(len(proc.shape))
                    )
                    scale_shape = [1] * len(proc.shape)
                    scale_shape[dim_i] = compressed_dom
                    scaled_proc = proc / naive_per_comp_dedup.reshape(scale_shape)

                    expanded_shape = list(proc.shape)
                    expanded_shape[dim_i] = naive_dom
                    expanded = np.zeros(expanded_shape, dtype=proc.dtype)
                    expanded[i_map] = scaled_proc[o_map]
                    proc = expanded

                for val, h in a.sel:
                    per_val_shape.append(attr[val].get_domain(h))
                    vals_order.append((a.table, a.order, a.attr, val, h))

        # Reshape to per-value dims
        proc = proc.reshape(per_val_shape)

        # Transpose from source (alphabetical) order to orig (node) order
        inv_perm = [vals_order.index(v) for v in orig]
        proc = proc.transpose(inv_perm)

        # Scale to counts and store
        md_marginals[idx] = proc.reshape(orig_marg.shape) * n

    return obs, potentials, cliques, loss_fn, md_sample, junction, md_marginals


class PrivBayesMare(MareModel):
    total_params: int | None = None

    def __init__(
        self,
        *,
        etotal: float | None = None,
        ep: float | None = None,
        e1: float = 0.3,
        e2: float = 0.7,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        unbounded_dp: bool = False,
        random_init: bool = False,
        skip_zero_counts: bool = True,
        minimum_cutoff: int | None = 3,
        rake: bool = True,
        rebalance: bool | dict = False,
        mirror_descent: MirrorDescentParams | bool = False,
        **kwargs,
    ) -> None:
        if etotal is None:
            etotal = 1
        self.ep = ep * etotal if ep is not None else None
        self.e1 = e1 * etotal
        self.e2 = e2 * etotal
        self.theta = theta
        self.use_r = use_r
        self.seed = seed
        self.random_init = random_init
        self.unbounded_dp = unbounded_dp
        self.skip_zero_counts = skip_zero_counts
        self.rake = rake
        self.rebalance = rebalance
        self.mirror_descent = mirror_descent
        self.kwargs = kwargs
        self.minimum_cutoff = minimum_cutoff

    @classmethod
    def visualise(cls, dir: str, models: dict):
        """Render the relational chain of submodels.

        Writes ``graph.html`` (moral graph view) and, when any submodel
        has fit a mirror-descent junction tree, ``junction.html``
        (junction-tree view).  Layout mirrors AdjuvantMare.visualise:
        each table gets a cluster, ctx/seq submodels nest inside, and
        ctx-style evidence on a child table is aliased to its source
        submodel's main node.
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

        # Map table -> seq submodel index (the canonical "owner").
        seq_owner: dict[str, int] = {}
        for i, (ver, _) in enumerate(items):
            if not ver.ctx:
                seq_owner[ver.ver.name] = i
        for i, (ver, _) in enumerate(items):
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
        useful_per_submodel: list[set] = []
        for i, (ver, model) in enumerate(items):
            alias: dict[str, str] = {}
            skip: set[str] = set()
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

            useful: set[str] = set()
            for u, v, ed in model.moral.edges(data=True):
                if ed.get("structure") or ed.get("evidence"):
                    useful.add(u)
                    useful.add(v)
            for n, d in model.moral.nodes(data=True):
                if d.get("table") is None:
                    continue
                if n in alias:
                    continue
                if n not in useful:
                    skip.add(n)
            alias_per_submodel.append(alias)
            skip_per_submodel.append(skip)
            useful_per_submodel.append(useful)

        # Lagged evidence (order != None) gets centralized.
        lagged_usage: dict[tuple[str, int], dict[str, dict[str, set[int]]]] = {}
        for i, (_, model) in enumerate(items):
            useful = useful_per_submodel[i]
            for n, d in model.moral.nodes(data=True):
                if d.get("table") is None or d.get("order") is None:
                    continue
                if n not in useful:
                    continue
                t = d["table"]
                o = d["order"]
                attrs_dict = lagged_usage.setdefault((t, o), {})
                vals_dict = attrs_dict.setdefault(d["attr"], {})
                vals_dict.setdefault(d["value"], set()).add(d["height"])

        canonical_lagged: dict[tuple[str, int, str, str], str] = {}
        for (t, o), attrs_dict in lagged_usage.items():
            for attr, vals in attrs_dict.items():
                for value in vals:
                    canonical_lagged[(t, o, attr, value)] = (
                        f"lag/{t}/{o}/{attr}/{value}"
                    )

        for i, (_, model) in enumerate(items):
            for n, d in model.moral.nodes(data=True):
                if d.get("table") is None or d.get("order") is None:
                    continue
                if n in skip_per_submodel[i] or n in alias_per_submodel[i]:
                    continue
                key = (d["table"], d["order"], d["attr"], d["value"])
                if key in canonical_lagged:
                    alias_per_submodel[i][n] = canonical_lagged[key]

        # Tree of table containers.
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

        # ---------- graph.html ----------
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

        def render_lagged_cluster(parent, table: str, order: int) -> None:
            attrs_dict = lagged_usage.get((table, order), {})
            if not attrs_dict:
                return
            sub = pydot.Cluster(
                f"lag_{table}_{order}",
                label=f"{table}[{order_label(order)}]",
                style="rounded,dashed",
                color="#666666",
                penwidth="1.4",
                margin="8",
                labeljust="l",
            )
            parent.add_subgraph(sub)
            for attr, vals in attrs_dict.items():
                unique_vals = list(vals.keys())
                multi = len(unique_vals) > 1
                if multi:
                    asub = pydot.Cluster(
                        f"lag_{table}_{order}_a_{attr}", label=attr,
                        penwidth="1.2", margin="4",
                    )
                    sub.add_subgraph(asub)
                    target = asub
                else:
                    asub = pydot.Subgraph(
                        f"lag_{table}_{order}_a_{attr}", label="",
                    )
                    sub.add_subgraph(asub)
                    target = asub
                for value in unique_vals:
                    node_id = canonical_lagged[(table, order, attr, value)]
                    if multi:
                        text = value.replace(attr + "_", "")
                    else:
                        text = value
                    target.add_node(pydot.Node(node_id, label=text))

        def render_table(table: str, parent_container) -> None:
            tc_name = f"t_{table}"
            tc = pydot.Cluster(
                tc_name, label=table, style="rounded,filled",
                fillcolor="#fafafa", penwidth="2.5", margin="14",
                labeljust="l", fontname="bold",
            )
            parent_container.add_subgraph(tc)

            for (lt, lo) in sorted(lagged_usage.keys()):
                if lt == table:
                    render_lagged_cluster(tc, lt, lo)

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
                    attrs=model.attrs,
                    prefix=prefixes[sub_idx],
                    skip_nodes=skip_per_submodel[sub_idx],
                    node_alias=alias_per_submodel[sub_idx],
                    order_label=order_label,
                )

            for child in children_by_parent.get(table, []):
                render_table(child, tc)

        for root in children_by_parent.get(None, []):
            render_table(root, o)

        ev_pairs: set[tuple[str, str]] = set()
        for i, (_, model) in enumerate(items):
            alias = alias_per_submodel[i]
            for u, v, ed in model.moral.edges(data=True):
                if not ed.get("evidence"):
                    continue
                if u not in alias or v not in alias:
                    continue
                a_id = alias[u]
                b_id = alias[v]
                if not (a_id.startswith("lag/") and b_id.startswith("lag/")):
                    continue
                if a_id == b_id:
                    continue
                ev_pairs.add(tuple(sorted((a_id, b_id))))
        for a_id, b_id in ev_pairs:
            o.add_edge(pydot.Edge(
                a_id, b_id,
                color="#7f7f7f", style="dashed",
                penwidth="0.8", dir="none",
                constraint="false",
            ))

        display_pydot(
            o,
            edges={"labeldistance": 1.0, "labelfontsize": 8, "arrowsize": 0.6},
            out=os.path.join(dir, "graph.svg"),
        )

        # ---------- junction.html ----------
        # Only emit when at least one submodel has mirror-descent fit.
        has_junction = any(
            getattr(m, "md_junction", None) is not None for _, m in items
        )
        if not has_junction:
            return

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
        for t in submodels_by_table:
            _emit(t)

        svgs: list[tuple[str, bytes]] = []
        for table in ordered_tables:
            # Skip tables whose submodels have no junction (md disabled).
            table_subs = [
                i for i in submodels_by_table[table]
                if getattr(items[i][1], "md_junction", None) is not None
            ]
            if not table_subs:
                continue

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
            for sub_idx in table_subs:
                ver, model = items[sub_idx]
                sub_label = f"{ver.ver.name} ({'ctx' if ver.ctx else 'seq'})"
                sc = pydot.Cluster(
                    f"sj{sub_idx}", label=sub_label, style="rounded",
                    penwidth="1.5", margin="10", labeljust="l",
                )
                jt.add_subgraph(sc)
                junction = model.md_junction
                messages = (
                    get_message_passing_order(junction)
                    if junction.number_of_edges() > 0
                    else None
                )
                node_id = _build_junction_tree_into(
                    sc, junction,
                    attrs=model.attrs,
                    messages=messages,
                    prefix=prefixes[sub_idx],
                )
                for cl in junction.nodes():
                    sub_anchors.append(node_id(cl))
                    break

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

        from ....utils.mlflow import strip_svg_preamble, wrap_zoom_html

        body = "".join(
            f'<section style="margin:18px 12px">'
            f'<h2 style="font:600 14px/1.2 sans-serif;margin:0 0 8px">'
            f"{table}</h2>{strip_svg_preamble(svg)}</section>"
            for table, svg in svgs
        )
        html = wrap_zoom_html(body, "junction")
        with open(os.path.join(dir, "junction.html"), "w") as f:
            f.write(html)

    @make_deterministic
    def preprocess(
        self,
        n: int,
        table: str | None,
        attrs: DatasetAttributes,
        oracle: MarginalOracle,
    ) -> Attributes:
        """Spend `self.ep` budget on Laplace-noised 1-way counts and
        rebalance `attrs[None]` against them."""
        if not self.rebalance:
            return attrs[None]

        if not self.ep:
            logger.warning(
                f"PrivBayes[{table}]: rebalance=True with ep=None, "
                "rebalancing against raw counts (no DP noise on 1-ways)."
            )

        noisy = _noisy_1way_counts(
            oracle, self.ep or 0.0, self.unbounded_dp,
        )
        main_counts = noisy.get(None, {})
        rebalance_kwargs = (
            self.rebalance if isinstance(self.rebalance, dict) else {}
        )
        rebalance_kwargs = {"unbounded_dp": self.unbounded_dp, **rebalance_kwargs}
        return _rebalance_with_noisy_counts(
            attrs[None], main_counts, rebalance_kwargs
        )

    @make_deterministic
    def fit(self, n: int, table: str, attrs: DatasetAttributes, oracle: MarginalOracle):
        from .implementation import MAX_EPSILON, calc_noisy_marginals, greedy_bayes

        # Fit network
        nodes, t = greedy_bayes(
            oracle,
            attrs,
            n,
            self.e1,
            self.e2,
            self.theta,
            self.use_r,
            self.unbounded_dp,
            self.random_init,
            prefer_table=table,
            rake=self.rake,
        )

        # Nodes are a tuple of a x attribute
        self.t = t
        self.nodes = nodes
        self.attrs = attrs
        self.moral = _build_moral(nodes, attrs)
        logger.info(self)

        d = 0
        for attr in cast(Attributes, attrs[None]).values():
            d += len(attr.vals)

        noise = (1 if self.unbounded_dp else 2) * d / self.e2
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0

        self.marginals = calc_noisy_marginals(
            oracle,
            self.nodes,
            noise,
            self.skip_zero_counts,
            minimum_cutoff=self.minimum_cutoff,
        )

        if self.mirror_descent:
            self._fit_mirror_descent_impl(n)
        else:
            self.md_marginals = None
            self.total_params = sum(int(m.size) for m in self.marginals)

    def _fit_mirror_descent_impl(self, n: int):
        d = 0
        for attr in cast(Attributes, self.attrs[None]).values():
            d += len(attr.vals)

        obs, potentials, cliques, loss_fn, md_sample, junction, md_marginals = (
            _fit_mirror_descent(
                self.mirror_descent, self.nodes, self.attrs, self.marginals,
                n, d, self.e2, self.unbounded_dp,
            )
        )
        self.md_obs = obs
        self.md_potentials = potentials
        self.md_cliques = cliques
        self.md_loss_fn = loss_fn
        self.md_sample = md_sample
        self.md_junction = junction
        self.md_marginals = md_marginals
        self.total_params = sum(int(p.size) for p in potentials)

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame:
        from .implementation import sample_rows

        marginals = (
            self.md_marginals if self.md_marginals is not None else self.marginals
        )
        return sample_rows(index, self.attrs, hist, self.nodes, marginals)

    def __str__(self) -> str:
        from .implementation import print_tree

        return print_tree(
            self.attrs,
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
            minimum_cutoff=self.minimum_cutoff,
        )


class PrivBayesSynth(Synth):
    name = "privbayes"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True
    total_params: int | None = None

    def __init__(
        self,
        ep: float | None = None,
        e1: float = 0.3,
        e2: float = 0.7,
        etotal: float | None = None,
        theta: float = 4,
        use_r: bool = True,
        seed: float | None = None,
        rebalance: bool = False,
        unbounded_dp: bool = False,
        random_init: bool = False,
        marginal_mode: MarginalOracle.MODES = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        skip_zero_counts: bool = True,
        minimum_cutoff: int | None = 3,
        mirror_descent: MirrorDescentParams | bool = False,
        **kwargs,
    ) -> None:
        if etotal is None:
            etotal = 1
        self.ep = ep * etotal if ep is not None else None
        self.e1 = e1 * etotal
        self.e2 = e2 * etotal
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
        self.minimum_cutoff = minimum_cutoff
        self.mirror_descent = mirror_descent
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(
        self, meta: dict[str | None, Attributes], data: dict[str, LazyFrame]
    ):
        attrs = meta
        _, tables = data_to_tables(data)
        table_name = next(iter(tables.keys()))
        table = tables[table_name]

        self._n = table.shape[0]
        self._partitions = len(table)
        self.original_attrs = attrs
        self.table_name = table_name

        if self.rebalance:
            if not self.ep:
                logger.warning(
                    "PrivBayesSynth: rebalance=True with ep=None — rebalancing "
                    "against raw counts (no DP noise on 1-ways)."
                )
            single_table: DatasetAttributes = {None: attrs[table_name]}
            with MarginalOracle(
                data,  # type: ignore
                single_table,
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
                preprocess=counts_preprocess,
            ) as o:
                # ep=0 / None → _noisy_1way_counts skips noise.
                noisy = _noisy_1way_counts(o, self.ep or 0.0, self.unbounded_dp)

            rebalance_kwargs = {"unbounded_dp": self.unbounded_dp, **self.kwargs}
            self.attrs = {
                table_name: _rebalance_with_noisy_counts(
                    attrs[table_name],
                    noisy.get(None, {}),
                    rebalance_kwargs,
                )
            }
        else:
            self.attrs = attrs

        self.table_attrs: DatasetAttributes = {None: self.attrs[table_name]}

    @make_deterministic
    def bake(self, data: dict[str, LazyFrame]):
        _, tables = data_to_tables(data)

        assert len(tables) == 1, "Only tabular data supported for now"

        table_name = next(iter(tables.keys()))
        table = tables[table_name]

        with MarginalOracle(
            data,  # type: ignore
            self.table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            self.n, self.d = table.shape
            # Fit network
            nodes, t = greedy_bayes(
                oracle,
                self.table_attrs,
                table.shape[0],
                self.e1,
                self.e2,
                self.theta,
                self.use_r,
                self.unbounded_dp,
                self.random_init,
            )

        # Nodes are a tuple of a x attribute
        self.table_name = table_name
        self.t = t
        self.nodes = nodes
        self.moral = _build_moral(nodes, self.table_attrs)
        logger.info(self)

    @make_deterministic
    def fit(self, data: dict[str, LazyFrame]):
        _, tables = data_to_tables(data)
        table = tables[self.table_name]
        self.partitions = len(table)
        self.n = ceil(table.shape[0] / self.partitions)

        noise = (1 if self.unbounded_dp else 2) * self.d / self.e2
        if self.e2 > MAX_EPSILON:
            logger.warning(f"Considering e2={self.e2} unbounded, sampling without DP.")
            noise = 0

        with MarginalOracle(
            data,  # type: ignore
            self.table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as o:
            self.marginals = calc_noisy_marginals(
                o,
                self.nodes,
                noise,
                self.skip_zero_counts,
                minimum_cutoff=self.minimum_cutoff,
            )

        if self.mirror_descent:
            self._fit_mirror_descent_impl()
        else:
            self.md_marginals = None
            self.total_params = sum(int(m.size) for m in self.marginals)

    def refresh(self, **kwargs):
        if "mirror_descent" in kwargs:
            has_md = bool(self.mirror_descent)
            self.mirror_descent = kwargs["mirror_descent"]
            if (
                has_md
                and isinstance(self.mirror_descent, dict)
                and len(self.mirror_descent) == 1
                and next(iter(self.mirror_descent)) == "sample"
            ):
                # Special case, when testing sampler do not recreate
                # potentials to get 1-1 comparison
                self.md_sample = self.mirror_descent["sample"]
                return
        if self.mirror_descent:
            self._fit_mirror_descent_impl()
        else:
            self.md_obs = None
            self.md_potentials = None
            self.md_cliques = None
            self.md_loss_fn = None
            self.md_sample = False
            self.md_junction = None
            self.total_params = sum(int(m.size) for m in self.marginals)

    def _fit_mirror_descent_impl(self):
        obs, potentials, cliques, loss_fn, md_sample, junction, md_marginals = (
            _fit_mirror_descent(
                self.mirror_descent, self.nodes, self.table_attrs, self.marginals,
                self.n, self.d, self.e2, self.unbounded_dp,
            )
        )
        self.md_obs = obs
        self.md_potentials = potentials
        self.md_cliques = cliques
        self.md_loss_fn = loss_fn
        self.md_sample = md_sample
        self.md_junction = junction
        self.md_marginals = md_marginals
        self.total_params = sum(int(p.size) for p in potentials)

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        import pandas as pd

        if n is None:
            n = self.n

        if getattr(self, "md_sample", False) and self.md_potentials is not None:
            tables = {self.table_name: self._sample_junction_tree(n, pd.RangeIndex(n))}
        else:
            marginals = (
                self.md_marginals if self.md_marginals is not None else self.marginals
            )
            tables = {
                self.table_name: sample_rows(
                    pd.RangeIndex(n),
                    {None: self.attrs[self.table_name]},
                    {},
                    self.nodes,
                    marginals,
                )
            }
        ids = {self.table_name: pd.DataFrame()}

        return tables_to_data(
            ids,
            tables,
            partition=i if self.partitions > 1 else None,
        )

    def _sample_junction_tree(self, n: int, idx) -> "pd.DataFrame":
        import pandas as pd
        from ....graph.sample import (
            create_sampler_meta,
            sample_junction_tree,
        )

        meta = create_sampler_meta(self.md_junction, self.md_cliques, self.table_attrs)
        columns = sample_junction_tree(self.md_potentials, meta, n, self.table_attrs)
        return pd.DataFrame(columns, index=idx)

    def __str__(self) -> str:
        return print_tree(
            {None: self.attrs[self.table_name]},
            self.nodes,
            self.e1,
            self.e2,
            self.theta,
            self.t,
            minimum_cutoff=self.minimum_cutoff,
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

        if getattr(self, "md_junction", None) is not None:
            display_junction_tree(
                self.md_junction, self.moral,
                messages=get_message_passing_order(self.md_junction),
                attrs=self.table_attrs,
                out=os.path.join(dir, "junction.svg"),
            )


def _build_moral(nodes: Sequence[Node], attrs: DatasetAttributes):
    """Build the moral (undirected) graph from a PrivBayes BN.

    Used by the visualise hooks; mirrors what _fit_mirror_descent
    constructs internally but does not require mirror descent to be
    enabled.  The graph carries `structure=True` on chosen BN parent
    edges, `chain=True` / `chain_bridged=True` on height-chain edges,
    and `immorality=True` / `immoral=True` on edges added or touched
    by moralization (set by to_moral).
    """
    from ....graph.hugin import to_moral

    g = derive_graph_from_nodes(nodes, attrs, prune=True)
    return to_moral(g)


def derive_graph_from_nodes(
    nodes: Sequence[Node], attrs: DatasetAttributes, prune: bool = True
):
    import networkx as nx

    def get_name(table, order, attr, val, height):
        out = ""
        if table:
            out += table
            if order is not None:
                out += f"[{order}]"
            out += "_"
        out += f"{attr}.{val}[{height}]"
        return out

    g = nx.DiGraph()
    commons = {}
    max_heights = {}
    for table, tattrs in attrs.items():
        if isinstance(tattrs, SeqAttributes):
            attr_sets = {**tattrs.hist, None: tattrs.attrs}
        else:
            attr_sets = {None: tattrs}

        for order, attr_set in attr_sets.items():
            for name, attr in attr_set.items():
                cmn = attr.common
                if cmn:
                    commons[(table, order, name)] = cmn.name
                    for h in range(cmn.height):
                        g.add_node(
                            get_name(table, order, name, cmn.name, h),
                            table=table,
                            order=order,
                            attr=name,
                            value=cmn.name,
                            height=h,
                        )
                        if h:
                            g.add_edge(
                                get_name(table, order, name, cmn.name, h),
                                get_name(table, order, name, cmn.name, h - 1),
                                chain=True,
                            )

                for v in attr.vals.values():
                    if not isinstance(v, CatValue):
                        continue

                    h_range = v.height if cmn is None else v.height - 1
                    max_heights[(table, order, name, v.name)] = h_range - 1
                    for h in range(h_range):
                        g.add_node(
                            get_name(table, order, name, v.name, h),
                            table=table,
                            order=order,
                            attr=name,
                            value=v.name,
                            height=h,
                        )

                        if h:
                            g.add_edge(
                                get_name(table, order, name, v.name, h),
                                get_name(table, order, name, v.name, h - 1),
                                chain=True,
                            )

                    if cmn:
                        g.add_edge(
                            get_name(table, order, name, cmn.name, 0),
                            get_name(table, order, name, v.name, v.height - 2),
                        )

    for node in nodes:
        for parent in node.p:
            node_name = get_name(None, None, node.attr, node.value, 0)
            order = None
            if len(parent) == 3:
                table, aname, sel = parent
                if isinstance(table, tuple):
                    order = table[1]
                    table = table[0]
            else:
                table = None
                aname, sel = parent

            if isinstance(sel, int):
                if table and order is not None:
                    cmn = cast(SeqAttributes, attrs[table]).hist[order][aname].common
                else:
                    tattrs = attrs[table]
                    if isinstance(tattrs, SeqAttributes):
                        assert tattrs.attrs
                        cmn = tattrs.attrs[aname].common
                    else:
                        cmn = tattrs[aname].common

                assert cmn
                cmn = cmn.name
                cmn_name = get_name(table, order, aname, cmn, sel)

                g.add_edge(cmn_name, node_name, structure=True)
            else:
                for k, v in sel.items():
                    other_name = get_name(table, order, aname, k, v)

                    g.add_edge(other_name, node_name, structure=True)

    if prune:
        for node, d in list(g.nodes(data=True)):
            if not d["height"]:
                continue  # keep all height 0 nodes

            next_neighbor = None
            prev_neighbor = None
            prune_node = True
            for neighbor in chain(g.successors(node), g.predecessors(node)):
                nd = g.nodes[neighbor]

                # Prune all nodes where their only neighbor is a different height
                # of their value
                if (
                    d["table"] != nd["table"]
                    or d["order"] != nd["order"]
                    or d["attr"] != nd["attr"]
                ):
                    prune_node = False
                elif d["value"] != nd["value"]:
                    if (
                        commons.get((d["table"], d["order"], d["attr"]), None)
                        == nd["value"]
                        and d["height"]
                        == max_heights[(d["table"], d["order"], d["attr"], d["value"])]
                    ):
                        prev_neighbor = neighbor
                    else:
                        prune_node = False
                elif d["height"] < nd["height"]:
                    prev_neighbor = neighbor
                else:
                    next_neighbor = neighbor

            if prune_node:
                g.remove_node(node)
                if next_neighbor is not None and prev_neighbor is not None:
                    g.add_edge(prev_neighbor, next_neighbor, chain_bridged=True)
                else:
                    pass

    return g


def derive_obs_from_model(
    nodes: Sequence[Node],
    attrs: DatasetAttributes,
    marginals: Sequence[np.ndarray],
    n: int,
    noise_scale: float = 0.0,
):
    from ....graph.hugin import AttrMeta, get_attrs
    from ....graph.loss import LinearObservation

    lin_obs = []
    for node, obs in zip(nodes, marginals):
        # Create Attr Meta
        out = []
        used_parent = False
        used_parent_table = None
        used_parent_order = None
        orig = []
        for s in node.p:
            if len(s) == 3:
                table_sel, attr_name, sel = s
            else:
                table_sel = None
                attr_name, sel = s

            if isinstance(table_sel, tuple):
                table = table_sel[0]
                order = table_sel[1]
            else:
                table = table_sel
                order = None

            attr = get_attrs(attrs, table, order)[attr_name]
            if isinstance(sel, int):
                new_sel = sel
                orig.append((table, order, attr_name, None, sel))
            else:
                cmn = attr.common.name if attr.common else None
                new_sel = []
                for val, h in sel.items():
                    if val == cmn:
                        continue  # skip common
                    new_sel.append((val, h))
                    orig.append((table, order, attr_name, val, h))
                if node.attr == attr_name and not any(
                    v == node.value for v, _ in new_sel
                ):
                    new_sel.append((node.value, 0))
                    used_parent = True
                    used_parent_table = table
                    used_parent_order = order
                new_sel = tuple(sorted(new_sel))
            out.append(AttrMeta(table, order, attr_name, new_sel))

        if not used_parent:
            out.append(AttrMeta(None, None, node.attr, ((node.value, 0),)))
            orig.append((None, None, node.attr, node.value, 0))
        else:
            orig.append((used_parent_table, used_parent_order, node.attr, node.value, 0))

        # Transpose observation
        source = tuple(sorted(out, key=lambda x: tuple(
            (0, "") if v is None else (1, v) for v in x[:-1]
        )))
        vals = list(
            chain.from_iterable(
                (
                    [(a.table, a.order, a.attr, None, a.sel)]
                    if isinstance(a.sel, int)
                    else [(a.table, a.order, a.attr, v[0], v[1]) for v in a.sel]
                )
                for a in source
            )
        )

        # Find new domain and transpose dimensions to be alphabetical
        new_obs = obs.astype("float32").transpose([orig.index(v) for v in vals])
        new_dom = []
        i = 0
        for a in source:
            if isinstance(a.sel, int):
                l = 1
            else:
                l = len(a.sel)
            nd = 1
            for d in new_obs.shape[i : i + l]:
                nd *= d
            new_dom.append(nd)
            i += l
        new_obs = new_obs.reshape(new_dom)

        # Align naive and compressed representations.
        # The data is at per-value coarse resolution (not leaf level).
        # get_naive_mapping/get_mapping return leaf-level arrays, so we must
        # deduplicate to avoid reading the same naive cell multiple times.
        for i, a in enumerate(source):
            if isinstance(a.sel, int):
                continue

            attr = get_attrs(attrs, a.table, a.order)[a.attr]
            naive_dom = new_obs.shape[i]
            compressed_dom = attr.get_domain(dict(a.sel))

            if naive_dom == compressed_dom:
                # No actual compression needed — domains match.
                continue

            # Deduplicate leaf-level mappings to get naive→compressed mapping.
            # Multiple leaves can map to the same naive cell; we only want
            # each naive cell read once.
            raw_naive = attr.get_naive_mapping(dict(a.sel))
            raw_compressed = attr.get_mapping(dict(a.sel))
            _, unique_idx = np.unique(raw_naive, return_index=True)
            naive_idx = raw_naive[unique_idx]
            compressed_idx = raw_compressed[unique_idx]

            i_map = tuple(
                naive_idx if j == i else slice(None) for j in range(len(new_obs.shape))
            )
            o_map = tuple(
                compressed_idx if j == i else slice(None)
                for j in range(len(new_obs.shape))
            )
            tmp = np.zeros(
                [compressed_dom if j == i else d for j, d in enumerate(new_obs.shape)]
            )
            np.add.at(tmp, o_map, new_obs[i_map])  # type: ignore
            new_obs = tmp

        new_obs /= n
        confidence = 1
        lo = LinearObservation(
            source,
            None,
            new_obs,
            confidence,
        )
        lin_obs.append(lo)

    return lin_obs
