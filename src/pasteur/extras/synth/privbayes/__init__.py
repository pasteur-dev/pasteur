from __future__ import annotations
from itertools import chain

import logging
from math import ceil
from typing import Any, Sequence, TypedDict, cast

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


class PrivBayesMare(MareModel):
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
        self.kwargs = kwargs
        self.minimum_cutoff = minimum_cutoff

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

    def sample(
        self, index: pd.Index, hist: dict[TableSelector, pd.DataFrame]
    ) -> pd.DataFrame:
        from .implementation import sample_rows

        return sample_rows(index, self.attrs, hist, self.nodes, self.marginals)

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


class MirrorDescentParams(TypedDict):
    lr: float
    max_iters: int
    ptol: float
    patience: int
    device: str
    compile: bool
    line_search: bool


MIRROR_DESCENT_DEFAULT: MirrorDescentParams = {
    "lr": 1,
    "max_iters": 10_000,
    "ptol": 2e-4,
    "patience": 50,
    "device": "auto",
    "compile": 10_000_000,
    "line_search": True,
}


class PrivBayesSynth(Synth):
    name = "privbayes"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

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
            with MarginalOracle(
                data,  # type: ignore
                attrs,
                mode=self.marginal_mode,
                min_chunk_size=self.marginal_min_chunk,
                max_worker_mult=self.marginal_worker_mult,
                preprocess=counts_preprocess,
            ) as o:
                counts = o.get_counts(desc="Calculating counts for column rebalancing")

            # TODO: Add noise and remove save support
            self.counts = counts
            self.attrs = {
                k: rebalance_attributes(
                    counts[k],
                    v,
                    unbounded_dp=self.unbounded_dp,
                    **self.kwargs,
                )
                for k, v in attrs.items()
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
            self._fit_mirror_descent()
        else:
            self.md_marginals = None

    def _fit_mirror_descent(self):
        from ....graph.beliefs import create_messages
        from ....graph.hugin import (
            find_elim_order,
            get_junction_tree,
            get_message_passing_order,
            to_moral,
        )
        from ....graph.loss import LinearObservation
        from ....graph.mirror_descent import mirror_descent

        md = self.mirror_descent if isinstance(self.mirror_descent, dict) else {}
        params = {**MIRROR_DESCENT_DEFAULT, **md}
        compress = params.pop("compress", True)
        device = None if params["device"] == "auto" else params["device"]
        params.pop("device", None)

        # Build junction tree
        g = derive_graph_from_nodes(self.nodes, self.table_attrs, prune=True)
        mg = to_moral(g)
        _, tri, _ = find_elim_order(mg, self.table_attrs, 10)

        # Fix compression bottlenecks: for each (attr, val), if 3+ different
        # heights exist, drop all above the 2nd-lowest to the 2nd-lowest.
        # [0, 2, 3] → [0, 2, 2]. Two heights are fine (direct edge, no
        # round-trip). Three+ create fine→coarse→fine bottlenecks.
        # if compress:
        #     height_sets: dict[tuple, list[int]] = {}
        #     for _, data in tri.nodes(data=True):
        #         key = (data["table"], data["order"], data["attr"], data["value"])
        #         height_sets.setdefault(key, []).append(data["height"])
        #     for key, hs in height_sets.items():
        #         unique = sorted(set(hs))
        #         if len(unique) <= 2:
        #             continue
        #         # Set everything above 2nd-lowest to 2nd-lowest
        #         cap = unique[1]
        #         for _, data in tri.nodes(data=True):
        #             k = (data["table"], data["order"], data["attr"], data["value"])
        #             if k == key and data["height"] > cap:
        #                 data["height"] = cap

        junction = get_junction_tree(tri, self.table_attrs, compress=compress)
        generations = get_message_passing_order(junction)
        cliques = list(junction.nodes())
        messages = create_messages(generations, self.table_attrs)

        # Build observations, normalize to probabilities
        noise_scale = (1 if self.unbounded_dp else 2) * self.d / self.e2
        obs = derive_obs_from_model(self.nodes, self.table_attrs, self.marginals, self.n, noise_scale)

        # Run mirror descent — returns (fitted clique potentials, loss_fn)
        potentials, loss_fn = mirror_descent(
            cliques,
            messages,
            obs,
            self.table_attrs,
            device=device,
            **params,
        )

        # Project clique potentials directly to node marginals.
        # Each node's attributes are a subset of some clique's attributes.
        # We marginalize (sum out) extra clique dims and remap indices
        # to the node's native shape. No lossy compress/decompress roundtrip.
        from ....graph.hugin import AttrMeta, get_attrs as _get_attrs, get_clique_domain
        from ....graph.loss import get_smallest_parent, get_parent_meta
        from ....graph.beliefs import convert_sel

        self.md_obs = obs
        self.md_potentials = potentials
        self.md_cliques = cliques
        self.md_loss_fn = loss_fn

        md_marginals = list(self.marginals)
        for idx, node in enumerate(self.nodes):
            orig_marg = self.marginals[idx]

            # Build the node's source AttrMeta (same as derive_obs_from_model)
            out = []
            orig = []
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
                attr = _get_attrs(self.table_attrs, table, order)[attr_name]
                if isinstance(sel, int):
                    orig.append((table, order, attr_name, None))
                else:
                    cmn = attr.common.name if attr.common else None
                    new_sel = []
                    for val, h in sel.items():
                        if val == cmn:
                            continue
                        new_sel.append((val, h))
                        orig.append((table, order, attr_name, val))
                    if node.attr == attr_name:
                        new_sel.append((node.value, 0))
                        used_parent = True
                    new_sel = tuple(sorted(new_sel))
                out.append(AttrMeta(table, order, attr_name, new_sel))
            if not used_parent:
                out.append(AttrMeta(None, None, node.attr, ((node.value, 0),)))
            orig.append((None, None, node.attr, node.value))

            source = tuple(sorted(out, key=lambda x: x[:-1]))

            # Find the parent clique and get projection metadata
            parent = get_smallest_parent(source, cliques, self.table_attrs)
            parent_idx = cliques.index(parent)
            meta = get_parent_meta(source, parent, self.table_attrs)

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
                for d in og_shape[: len(meta.b_doms)]:
                    a_idx_dom *= d
                b_idx_dom = 1
                for d in meta.b_doms:
                    b_idx_dom *= d
                rest_dom = 1
                for d in og_shape[len(meta.b_doms) :]:
                    rest_dom *= d
                proc = proc.reshape((a_idx_dom, -1))
                out = np.zeros((b_idx_dom, rest_dom), dtype=proc.dtype)
                np.add.at(out, meta.idx, proc)
                new_shape = list(meta.b_doms) + list(og_shape[len(meta.b_doms) :])
                proc = out.reshape(new_shape).transpose(meta.transpose_undo)

            # proc is now in source shape (one compressed dim per attribute).
            # Expand each compressed attr dim to per-value naive dims.
            vals_order = []
            per_val_shape = []
            for a in source:
                if isinstance(a.sel, int):
                    attr = _get_attrs(self.table_attrs, a.table, a.order)[a.attr]
                    per_val_shape.append(attr.common.get_domain(a.sel))
                    vals_order.append((a.table, a.order, a.attr, None))
                else:
                    attr = _get_attrs(self.table_attrs, a.table, a.order)[a.attr]
                    for val, h in a.sel:
                        per_val_shape.append(attr[val].get_domain(h))
                        vals_order.append((a.table, a.order, a.attr, val))

            # Reshape compressed dims to per-value dims
            proc = proc.reshape(per_val_shape)

            # Transpose from source (alphabetical) order to orig (node) order
            inv_perm = [vals_order.index(v) for v in orig]
            proc = proc.transpose(inv_perm)

            # Scale to counts and store
            md_marginals[idx] = proc.reshape(orig_marg.shape) * self.n

        self.md_marginals = md_marginals

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        import pandas as pd

        if n is None:
            n = self.n

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

        return tables_to_data(ids, tables)

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

                g.add_edge(cmn_name, node_name)
            else:
                for k, v in sel.items():
                    other_name = get_name(table, order, aname, k, v)

                    g.add_edge(other_name, node_name)

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
                    g.add_edge(prev_neighbor, next_neighbor)
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
                orig.append((table, order, attr_name, None))
            else:
                cmn = attr.common.name if attr.common else None
                new_sel = []
                for val, h in sel.items():
                    if val == cmn:
                        continue  # skip common
                    new_sel.append((val, h))
                    orig.append((table, order, attr_name, val))
                if node.attr == attr_name:
                    new_sel.append((node.value, 0))
                    used_parent = True
                new_sel = tuple(sorted(new_sel))
            out.append(AttrMeta(table, order, attr_name, new_sel))

        if not used_parent:
            out.append(AttrMeta(None, None, node.attr, ((node.value, 0),)))
        orig.append((None, None, node.attr, node.value))

        # Transpose observation
        source = tuple(sorted(out, key=lambda x: x[:-1]))
        vals = list(
            chain.from_iterable(
                (
                    [(a.table, a.order, a.attr, None)]
                    if isinstance(a.sel, int)
                    else [(a.table, a.order, a.attr, v[0]) for v in a.sel]
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

        # Align naive and compressed representations
        for i, a in enumerate(source):
            if isinstance(a.sel, int):  # or len(a.sel) == 1:
                # Skip single dimension
                continue

            attr = get_attrs(attrs, a.table, a.order)[a.attr]
            i_map = tuple(
                attr.get_naive_mapping(dict(a.sel)) if j == i else slice(None)
                for j in range(len(new_obs.shape))
            )
            o_map = tuple(
                attr.get_mapping(dict(a.sel)) if j == i else slice(None)
                for j in range(len(new_obs.shape))
            )
            tmp = np.zeros(
                [
                    attr.get_domain(dict(a.sel)) if j == i else d
                    for j, d in enumerate(new_obs.shape)
                ]
            )
            np.add.at(tmp, o_map, new_obs[i_map])  # type: ignore
            new_obs = tmp

        # Clip negative noise then normalize to probabilities for mirror descent
        new_obs = new_obs.clip(0)
        obs_sum = new_obs.sum()
        if obs_sum != 0:
            new_obs = new_obs / obs_sum

        confidence = n / (n + noise_scale * new_obs.size)
        lo = LinearObservation(
            source,
            None,
            new_obs,
            confidence,
        )
        lin_obs.append(lo)

    return lin_obs
