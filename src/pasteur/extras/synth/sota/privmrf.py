"""PrivMRF: Differentially Private Markov Random Field.

Structure learning via DP local search (greedy edge addition scored by noisy
pairwise TVD), then iterative measure selection and parameter fitting via
mirror descent (entropy descent).

Reference: ../PrivMRF/PrivMRF/
"""

from __future__ import annotations

import itertools
import logging
from math import sqrt
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from ....attribute import Attributes, DatasetAttributes
from ....marginal import MarginalOracle
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data
from ....utils.progress import piter
from .common import (
    cdp_rho,
    measure,
    fit_pgm,
    get_attr_names,
    clique_domain_size,
    attr_domain_size,
    _col_to_attr_sel,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================
# Graph utilities
# ============================================================
def _triangulate(graph: nx.Graph) -> nx.Graph:
    """Minimum-degree elimination triangulation."""
    g = graph.copy()
    fill = []
    while g.number_of_nodes() > 0:
        v = min(g.nodes(), key=lambda n: g.degree(n))
        for u, w in itertools.combinations(g.neighbors(v), 2):
            if not g.has_edge(u, w):
                g.add_edge(u, w)
                fill.append((u, w))
        g.remove_node(v)
    result = graph.copy()
    result.add_edges_from(fill)
    return result


def _graph_score(
    graph: nx.Graph,
    tvd: dict[tuple[str, str], float],
    attrs: DatasetAttributes,
    max_clique_size: float,
    max_param_size: float,
    size_penalty: float,
) -> float:
    """Evaluate graph: sum(edge TVDs) - penalty * total_clique_domain.

    Returns -1e8 if any constraint is violated."""
    g = graph if nx.is_chordal(graph) else _triangulate(graph)

    total_size = 0
    for clique in nx.find_cliques(g):
        cs = clique_domain_size(tuple(sorted(clique)), attrs)
        if cs > max_clique_size:
            return -1e8
        total_size += cs
    if total_size > max_param_size:
        return -1e8

    tvd_sum = sum(tvd.get((a, b), tvd.get((b, a), 0)) for a, b in g.edges())
    return tvd_sum - size_penalty * total_size


# ============================================================
# Structure learning
# ============================================================
def _compute_pairwise_tvd(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    all_attrs: list[str],
    sigma: float,
) -> dict[tuple[str, str], float]:
    """Compute noisy pairwise TVD = |P(a,b) - P(a)P(b)|_1 / 2 for all pairs."""
    # 1-way marginals
    requests_1 = [
        [(_col_to_attr_sel(a, attrs)[0], dict(_col_to_attr_sel(a, attrs)[1]))]
        for a in all_attrs
    ]
    results_1 = oracle.process(requests_1, postprocess=None)
    mar1 = {}
    for a, r in zip(all_attrs, results_1):
        v = r.ravel().astype(np.float64)
        s = v.sum()
        mar1[a] = v / s if s > 0 else v

    # 2-way marginals
    pairs = list(itertools.combinations(all_attrs, 2))
    requests_2 = []
    for a, b in pairs:
        req = {}
        for col_name in (a, b):
            attr_name, sel = _col_to_attr_sel(col_name, attrs)
            if attr_name in req:
                req[attr_name].update(sel)
            else:
                req[attr_name] = dict(sel)
        requests_2.append(list(req.items()))
    results_2 = oracle.process(requests_2, postprocess=None)

    tvd: dict[tuple[str, str], float] = {}
    for (a, b), r in zip(pairs, results_2):
        joint = r.ravel().astype(np.float64)
        s = joint.sum()
        if s > 0:
            joint /= s
        indep = np.outer(mar1[a], mar1[b]).ravel()
        val = float(np.sum(np.abs(joint - indep)) / 2)
        if sigma > 0:
            val += np.random.normal(0, sigma)
        tvd[a, b] = val
        tvd[b, a] = val
    return tvd


def _structure_search(
    all_attrs: list[str],
    tvd: dict[tuple[str, str], float],
    attrs: DatasetAttributes,
    max_clique_size: float,
    max_param_size: float,
    size_penalty: float,
) -> tuple[nx.Graph, list[tuple[str, ...]]]:
    """Greedy edge addition to build a graph structure, scored by noisy TVD.

    Follows the original PrivMRF local_search: at each iteration, try adding
    every non-existing edge and pick the one with highest score. Stop after
    3 consecutive iterations with no valid candidate."""
    graph = nx.Graph()
    graph.add_nodes_from(all_attrs)

    n_attrs = len(all_attrs)
    search_iters = n_attrs * (n_attrs - 1)
    local_count = 0

    for it in piter(range(search_iters), desc="PrivMRF structure"):
        candidates = [
            (a, b)
            for a, b in itertools.combinations(all_attrs, 2)
            if not graph.has_edge(a, b)
        ]
        if not candidates:
            break

        best_score = -1e8
        best_edge = None
        for a, b in candidates:
            graph.add_edge(a, b)
            score = _graph_score(
                graph, tvd, attrs, max_clique_size, max_param_size, size_penalty,
            )
            graph.remove_edge(a, b)
            if score > best_score:
                best_score = score
                best_edge = (a, b)

        if best_edge is None:
            local_count += 1
            if local_count >= 3:
                break
            continue

        local_count = 0
        graph.add_edge(*best_edge)
        logger.info(
            f"PrivMRF structure: iter {it}, added ({best_edge[0]}, {best_edge[1]}), "
            f"score={best_score:.4f}, edges={graph.number_of_edges()}"
        )

    # Final triangulation to ensure chordality (required for junction tree)
    if not nx.is_chordal(graph):
        graph = _triangulate(graph)

    cliques = [tuple(sorted(c)) for c in nx.find_cliques(graph)]
    return graph, cliques


# ============================================================
# Initial measure selection (greedy Bayes within cliques)
# ============================================================
def _greedy_bayes_measures(
    clique_attrs: tuple[str, ...],
    tvd: dict[tuple[str, str], float],
    attrs: DatasetAttributes,
    max_measure_dom: float,
    max_parents: int = 5,
) -> list[tuple[str, ...]]:
    """Within a maximal clique, greedily select measures using TVD correlation.

    Scoring: sum(TVD(child, p_i)) / sqrt(|parents| + sum(TVD(p_i, p_j)))."""
    if len(clique_attrs) <= 1:
        return []

    remaining = list(clique_attrs)
    first = remaining.pop(np.random.randint(len(remaining)))
    placed = [first]
    measures: list[tuple[str, ...]] = []

    while remaining:
        best_score = -1e8
        best_attr = None
        best_meas = None

        for attr in remaining:
            adom = attr_domain_size(attr, attrs)
            max_p = min(len(placed), max_parents)
            for size in range(1, max_p + 1):
                for parents in itertools.combinations(placed, size):
                    dom = adom
                    for p in parents:
                        dom *= attr_domain_size(p, attrs)
                    if dom > max_measure_dom:
                        continue

                    num = sum(
                        tvd.get((attr, p), tvd.get((p, attr), 0)) for p in parents
                    )
                    denom = float(len(parents))
                    for pi, pj in itertools.combinations(parents, 2):
                        denom += tvd.get((pi, pj), tvd.get((pj, pi), 0))
                    denom = max(denom, 1.0)
                    score = num / sqrt(denom)

                    if score > best_score:
                        best_score = score
                        best_attr = attr
                        best_meas = tuple(sorted([attr] + list(parents)))

        if best_attr is None:
            best_attr = remaining[0]

        remaining.remove(best_attr)
        placed.append(best_attr)

        if best_meas is not None and len(best_meas) > 1:
            measures.append(best_meas)

    return measures


# ============================================================
# PrivMRF synthesizer
# ============================================================
class PrivMRF(Synth):
    name = "privmrf"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

    def __init__(
        self,
        e: float = 1.0,
        etotal: float | None = None,
        delta: float = 1e-9,
        beta1: float = 0.10,
        beta3: float = 0.10,
        t: float = 0.8,
        ed_steps: int = 3,
        max_clique_size: float = 1e7,
        max_parameter_size: float = 5e7,
        size_penalty: float = 1e-8,
        max_measure_attr_num: int = 6,
        theta: float = 6,
        measures_per_step: int = 3,
        max_norm_queries: int = 200,
        marginal_mode: "MarginalOracle.MODES" = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        seed: int | None = None,
        n: int | None = None,
        partitions: int | None = None,
        mirror_descent: dict | None = None,
        **kwargs,
    ) -> None:
        self.e = etotal if etotal is not None else e
        self.delta = delta
        self.beta1 = beta1
        self.beta3 = beta3
        self.t = t
        self.ed_steps = ed_steps
        self.max_clique_size = max_clique_size
        self.max_parameter_size = max_parameter_size
        self.size_penalty = size_penalty
        self.max_measure_attr_num = max_measure_attr_num
        self.theta = theta
        self.measures_per_step = measures_per_step
        self.max_norm_queries = max_norm_queries
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
        n = self._n
        table_attrs: DatasetAttributes = {None: self.attrs[self.table]}

        rho = cdp_rho(self.e, self.delta)
        all_attrs = get_attr_names(table_attrs)
        num_attrs = len(all_attrs)

        # Budget allocation:
        #   beta1 * rho  -> structure learning (noisy TVD)
        #   beta3 * rho  -> measure selection (noisy L1 norms)
        #   remaining    -> marginal measurements, split by t
        rho_structure = self.beta1 * rho
        rho_selection = self.beta3 * rho
        rho_measurement = (1 - self.beta1 - self.beta3) * rho
        rho_initial = self.t * rho_measurement
        rho_descent = (1 - self.t) * rho_measurement

        # TVD noise: Gaussian mechanism with sensitivity 2/n
        num_pairs = num_attrs * (num_attrs - 1) / 2
        tvd_sens = 2.0 / n
        sigma_tvd = (
            tvd_sens * sqrt(num_pairs / (2 * rho_structure))
            if rho_structure > 0
            else 0
        )

        # Estimate measurement noise for domain-size thresholds
        est_n_meas = num_attrs + self.ed_steps * self.measures_per_step
        est_sigma = (
            sqrt(est_n_meas / (2 * rho_measurement)) if rho_measurement > 0 else 1
        )
        max_dom_2way = max(n / (est_sigma * self.theta), 100)
        max_dom_highway = max(n / (est_sigma * self.theta), 100)

        with MarginalOracle(
            data,
            table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            # ============================================================
            # Phase 1: Structure learning via noisy pairwise TVD
            # ============================================================
            logger.info(
                f"PrivMRF Phase 1: Structure learning "
                f"({num_attrs} attrs, sigma_tvd={sigma_tvd:.4f})"
            )
            tvd = _compute_pairwise_tvd(oracle, table_attrs, all_attrs, sigma_tvd)
            graph, maximal_cliques = _structure_search(
                all_attrs,
                tvd,
                table_attrs,
                self.max_clique_size,
                self.max_parameter_size,
                self.size_penalty,
            )
            logger.info(
                f"PrivMRF: {graph.number_of_edges()} edges, "
                f"{len(maximal_cliques)} maximal cliques"
            )

            # ============================================================
            # Phase 2: Initial measure selection (greedy Bayes within cliques)
            # ============================================================
            initial_cliques: list[tuple[str, ...]] = []
            for clique in maximal_cliques:
                initial_cliques.extend(
                    _greedy_bayes_measures(
                        clique, tvd, table_attrs, max_dom_2way,
                    )
                )
            initial_cliques = list(set(initial_cliques))
            # Ensure every attribute appears in at least one measure
            for a in all_attrs:
                if not any(a in cl for cl in initial_cliques):
                    initial_cliques.append((a,))
            # Always include all 1-way marginals
            for a in all_attrs:
                ow = (a,)
                if ow not in initial_cliques:
                    initial_cliques.append(ow)
            initial_cliques = list(set(initial_cliques))
            logger.info(
                f"PrivMRF Phase 2: {len(initial_cliques)} initial measures"
            )

            # ============================================================
            # Phase 3: Measure initial marginals + fit
            # ============================================================
            sigma_init = (
                sqrt(len(initial_cliques) / (2 * rho_initial))
                if rho_initial > 0
                else 1
            )
            logger.info(
                f"PrivMRF Phase 3: Measuring {len(initial_cliques)} marginals "
                f"(sigma={sigma_init:.2f})"
            )
            measurements = measure(oracle, table_attrs, initial_cliques, sigma_init)

            model = fit_pgm(
                table_attrs,
                measurements,
                n,
                {**self.md_params, "max_iters": 5000},
            )

            # ============================================================
            # Phase 4: Entropy descent (iterative measure selection + refit)
            # ============================================================
            if self.ed_steps > 0 and rho_descent > 0:
                candidates = self._generate_candidates(
                    maximal_cliques, table_attrs, max_dom_2way, max_dom_highway,
                )
                measured_set = set(m.clique for m in measurements)
                candidates = [c for c in candidates if c not in measured_set]
                logger.info(
                    f"PrivMRF Phase 4: Entropy descent "
                    f"({self.ed_steps} steps, {len(candidates)} candidates)"
                )

                rho_sel_step = (
                    rho_selection / self.ed_steps if self.ed_steps > 0 else 0
                )
                rho_meas_step = rho_descent / self.ed_steps

                for step in range(self.ed_steps):
                    if not candidates:
                        break

                    selected = self._select_measures(
                        oracle,
                        table_attrs,
                        model,
                        candidates,
                        self.measures_per_step,
                        rho_sel_step,
                        n,
                    )
                    if not selected:
                        break

                    candidates = [c for c in candidates if c not in set(selected)]

                    sigma_step = (
                        sqrt(len(selected) / (2 * rho_meas_step))
                        if rho_meas_step > 0
                        else 1
                    )
                    new_meas = measure(oracle, table_attrs, selected, sigma_step)
                    measurements.extend(new_meas)

                    logger.info(
                        f"PrivMRF step {step + 1}/{self.ed_steps}: "
                        f"+{len(selected)} measures (sigma={sigma_step:.2f}), "
                        f"total={len(measurements)}"
                    )

                    model = fit_pgm(
                        table_attrs,
                        measurements,
                        n,
                        {**self.md_params, "max_iters": 5000},
                        prev_model=model,
                    )

            # Final fit (fresh start, more iterations)
            logger.info(
                f"PrivMRF: Final fit with {len(measurements)} measurements"
            )
            self.model = fit_pgm(
                table_attrs,
                measurements,
                n,
                {**self.md_params, "max_iters": 10000},
            )
            self.table_attrs = table_attrs

    def _generate_candidates(
        self,
        maximal_cliques: list[tuple[str, ...]],
        attrs: DatasetAttributes,
        max_dom_2way: float,
        max_dom_highway: float,
    ) -> list[tuple[str, ...]]:
        """Generate candidate measures from maximal cliques (all sub-cliques
        that fit within domain thresholds)."""
        candidates: set[tuple[str, ...]] = set()
        for clique in maximal_cliques:
            for length in range(
                1, min(self.max_measure_attr_num + 1, len(clique) + 1)
            ):
                limit = max_dom_2way if length <= 2 else max_dom_highway
                for combo in itertools.combinations(clique, length):
                    combo = tuple(sorted(combo))
                    if clique_domain_size(combo, attrs) <= limit:
                        candidates.add(combo)
        return list(candidates)

    def _select_measures(
        self,
        oracle: MarginalOracle,
        attrs: DatasetAttributes,
        model,
        candidates: list[tuple[str, ...]],
        n_select: int,
        rho_select: float,
        n: int,
    ) -> list[tuple[str, ...]]:
        """Select measures with largest noisy L1 error vs current model."""
        if not candidates or n_select <= 0:
            return []

        # Subsample if too many candidates
        query_candidates = candidates
        if len(candidates) > self.max_norm_queries:
            idx = np.random.choice(
                len(candidates), self.max_norm_queries, replace=False
            )
            query_candidates = [candidates[i] for i in idx]

        # Compute true marginals for scoring
        requests = []
        for cl in query_candidates:
            req = {}
            for col_name in cl:
                attr_name, sel = _col_to_attr_sel(col_name, attrs)
                if attr_name in req:
                    req[attr_name].update(sel)
                else:
                    req[attr_name] = dict(sel)
            requests.append(list(req.items()))
        true_results = oracle.process(requests, postprocess=None)

        # L1 norm noise: Gaussian mechanism with sensitivity 2/n
        l1_sens = 2.0 / n
        sigma_l1 = (
            l1_sens * sqrt(len(query_candidates) / (2 * rho_select))
            if rho_select > 0
            else 0
        )

        errors = []
        for cl, true_raw in zip(query_candidates, true_results):
            true = true_raw.ravel().astype(np.float64)
            est = model.project(cl, attrs).ravel()
            # Compare in probability space
            ts, es = true.sum(), est.sum()
            tp = true / ts if ts > 0 else true
            ep = est / es if es > 0 else est
            l1 = float(np.linalg.norm(tp - ep, 1))
            if sigma_l1 > 0:
                l1 += np.random.normal(0, sigma_l1)
            errors.append((cl, l1))

        errors.sort(key=lambda x: -x[1])
        return [cl for cl, _ in errors[:n_select]]

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        n = n or self.n
        df = self.model.synthetic_data(n, self.table_attrs)
        return tables_to_data(
            {self.table: pd.DataFrame()},
            {self.table: df},
            partition=i if self.partitions > 1 else None,
        )
