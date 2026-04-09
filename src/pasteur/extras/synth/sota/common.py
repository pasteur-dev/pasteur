"""Shared utilities for MST and AIM: measurement, CDP, PGM fitting, sampling."""

from __future__ import annotations

import logging
from math import exp, log1p
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from scipy.special import softmax

from ....attribute import Attributes, DatasetAttributes, IdxValue
from ....marginal import MarginalOracle
from ....graph.mirror_descent import MIRROR_DESCENT_DEFAULT

logger = logging.getLogger(__name__)


# ============================================================
# CDP <-> (eps, delta)-DP conversion (from cdp2adp.py)
# ============================================================
def _cdp_delta(rho: float, eps: float) -> float:
    if rho == 0:
        return 0.0
    amin, amax = 1.01, (eps + 1) / (2 * rho) + 2
    for _ in range(1000):
        alpha = (amin + amax) / 2
        derivative = (2 * alpha - 1) * rho - eps + log1p(-1.0 / alpha)
        if derivative < 0:
            amin = alpha
        else:
            amax = alpha
    delta = exp(
        (alpha - 1) * (alpha * rho - eps) + alpha * log1p(-1 / alpha)
    ) / (alpha - 1.0)
    return min(delta, 1.0)


def cdp_rho(eps: float, delta: float) -> float:
    """Find smallest rho such that rho-CDP implies (eps, delta)-DP."""
    if delta >= 1:
        return 0.0
    rhomin, rhomax = 0.0, eps + 1
    for _ in range(1000):
        rho = (rhomin + rhomax) / 2
        if _cdp_delta(rho, eps) <= delta:
            rhomin = rho
        else:
            rhomax = rho
    return rhomin


# ============================================================
# Exponential mechanism
# ============================================================
def exponential_mechanism(
    qualities: dict | np.ndarray,
    eps: float,
    sensitivity: float = 1.0,
) -> object:
    """Sample from the exponential mechanism. Returns the selected key."""
    if isinstance(qualities, dict):
        keys = list(qualities.keys())
        q = np.array([qualities[k] for k in keys])
    else:
        keys = np.arange(len(qualities))
        q = np.array(qualities)
    q = q - q.max()
    p = softmax(0.5 * eps / sensitivity * q)
    return keys[np.random.choice(p.size, p=p)]


# ============================================================
# Measurement
# ============================================================
class Measurement(NamedTuple):
    """A noisy measurement of a marginal clique."""

    clique: tuple[str, ...]
    noisy: np.ndarray  # noisy count vector (flat)
    sigma: float  # noise standard deviation


def _attr_sel(attr_name: str, attrs: DatasetAttributes):
    """Build the oracle selector for an attribute at height 0."""
    attr = cast(Attributes, attrs[None])[attr_name]
    if attr.common:
        return 0
    return {v: 0 for v in attr.vals}


def _build_request(clique: tuple[str, ...], attrs: DatasetAttributes):
    """Build a marginal oracle request for a clique at height 0."""
    return [(attr_name, _attr_sel(attr_name, attrs)) for attr_name in clique]


def clique_domain_size(
    clique: tuple[str, ...], attrs: DatasetAttributes
) -> int:
    """Domain size for a clique at height 0."""
    dom = 1
    for attr_name in clique:
        attr = cast(Attributes, attrs[None])[attr_name]
        for val in attr.vals.values():
            if attr.common and val.name == attr.common.name:
                continue
            dom *= cast(IdxValue, val).get_domain(0)
    return dom


def measure(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    cliques: list[tuple[str, ...]],
    sigma: float,
    weights: np.ndarray | None = None,
) -> list[Measurement]:
    """Measure marginals at height 0 with Gaussian noise."""
    if weights is None:
        weights = np.ones(len(cliques))
    weights = weights / np.linalg.norm(weights)

    # Build requests at height 0
    requests = [_build_request(clique, attrs) for clique in cliques]

    results = oracle.process(requests, postprocess=None)

    measurements = []
    for clique, marginal, wgt in zip(cliques, results, weights):
        x = marginal.ravel().astype(np.float64)
        noise = np.random.normal(0, sigma / wgt, size=x.size)
        measurements.append(Measurement(clique, x + noise, sigma / wgt))

    return measurements


# ============================================================
# PGM fitting via our mirror descent engine
# ============================================================
class FittedPGM:
    """Wrapper around fitted clique potentials with projection + sampling."""

    def __init__(
        self,
        potentials: list[np.ndarray],
        raw_theta: list[np.ndarray],
        cliques: list,  # list of CliqueMeta tuples
        clique_names: list[tuple[str, ...]],  # attr name tuples
        n: int,
        loss_fn,
    ):
        self.potentials = potentials
        self.raw_theta = raw_theta  # pre-BP log-potentials for warm starting
        self.cliques = cliques
        self.clique_names = clique_names
        self.n = n
        self.loss_fn = loss_fn

    def project(
        self, clique: tuple[str, ...], attrs: DatasetAttributes
    ) -> np.ndarray:
        """Project fitted model to a marginal over the given attrs."""
        from ....graph.loss import get_parents
        from ....graph.hugin import get_clique_domain, AttrMeta, get_attrs
        from ....graph.beliefs import convert_sel

        # Build source meta for the requested clique
        source = []
        for attr_name in sorted(clique):
            attr = get_attrs(attrs, None, None)[attr_name]
            if attr.common:
                sel: int | tuple = 0
            else:
                vals = [(v, 0) for v in attr.vals]
                sel = tuple(sorted(vals))
            source.append(AttrMeta(None, None, attr_name, sel))
        source_tuple = tuple(source)

        parents = get_parents(source_tuple, self.cliques)
        if not parents:
            # Try to find a clique that contains all requested attrs
            for i, cl_names in enumerate(self.clique_names):
                if set(clique).issubset(set(cl_names)):
                    parent = self.cliques[i]
                    break
            else:
                # No single clique contains all attrs — approximate with
                # independence: outer product of per-attribute marginals
                marginals = []
                for attr_name in sorted(clique):
                    m = self.project((attr_name,), attrs).ravel()
                    m = m / m.sum() if m.sum() > 0 else m
                    marginals.append(m)
                result = marginals[0]
                for m in marginals[1:]:
                    result = np.outer(result, m).ravel()
                return result * self.n
        else:
            parent = min(
                parents, key=lambda x: get_clique_domain(x, attrs)
            )

        parent_idx = self.cliques.index(parent)
        proc = self.potentials[parent_idx].copy()

        # Sum out dims not in the requested clique
        parent_attrs = [a.attr for a in parent]
        sum_dims = tuple(
            i
            for i, a in enumerate(parent)
            if a.attr not in clique
        )
        if sum_dims:
            proc = proc.sum(axis=sum_dims)

        return proc * self.n

    def synthetic_data(
        self, n: int, attrs: DatasetAttributes
    ) -> pd.DataFrame:
        """Generate synthetic data by ancestral sampling from the junction tree."""
        from ....graph.beliefs import convert_sel
        from ....graph.hugin import get_attrs

        all_attrs = cast(Attributes, attrs[None])
        columns = {}

        # For each attribute, find the clique with the smallest domain
        # that contains it, and sample from the marginal
        for attr_name in all_attrs:
            attr = all_attrs[attr_name]
            marginal = self.project((attr_name,), attrs).ravel()
            marginal = marginal.clip(0)
            total = marginal.sum()
            if total > 0:
                probs = marginal / total
            else:
                probs = np.ones(len(marginal)) / len(marginal)
            columns[attr_name] = np.random.choice(len(probs), size=n, p=probs)

        return pd.DataFrame(columns)


def fit_pgm(
    attrs: DatasetAttributes,
    measurements: list[Measurement],
    n: int,
    md_params: dict | None = None,
    prev_model: FittedPGM | None = None,
) -> FittedPGM:
    """Fit a PGM model from measurements using our mirror descent + BP."""
    from ....graph.beliefs import create_messages
    from ....graph.hugin import (
        AttrMeta,
        get_attrs,
        get_junction_tree_from_cliques,
        get_message_passing_order,
    )
    from ....graph.loss import LinearObservation
    from ....graph.mirror_descent import mirror_descent
    from ....graph.beliefs import convert_sel

    params = {**MIRROR_DESCENT_DEFAULT, **(md_params or {})}
    device = params.pop("device", "auto")
    device = None if device == "auto" else device
    params.pop("compress", None)
    params.pop("sample", None)
    params.pop("tree", None)

    # Build CliqueMeta for each measurement clique (at height 0)
    obs_list = []
    clique_metas = []
    clique_names = []
    for meas in measurements:
        source = []
        for attr_name in sorted(meas.clique):
            attr = get_attrs(attrs, None, None)[attr_name]
            if attr.common:
                sel: int | tuple = 0
            else:
                vals = [(v, 0) for v in attr.vals]
                sel = tuple(sorted(vals))
            source.append(AttrMeta(None, None, attr_name, sel))
        source_tuple = tuple(source)
        clique_metas.append(source_tuple)
        clique_names.append(meas.clique)

        # Normalize to probabilities, clip negatives
        obs = meas.noisy.copy()
        # Reshape to match source domain
        shape = []
        for a in source_tuple:
            attr = get_attrs(attrs, a.table, a.order)[a.attr]
            shape.append(attr.get_domain(convert_sel(a.sel)))
        obs = obs.reshape(shape)
        obs = obs.clip(0)
        obs_sum = obs.sum()
        if obs_sum > 0:
            obs = obs / obs_sum
        obs = obs.astype(np.float32)

        confidence = n / (n + meas.sigma * obs.size)
        obs_list.append(LinearObservation(source_tuple, None, obs, confidence))

    # Deduplicate clique metas (multiple measurements can map to same clique)
    unique_metas = list(dict.fromkeys(clique_metas))

    # Build junction tree from unique observation cliques
    junction = get_junction_tree_from_cliques(unique_metas)
    generations = get_message_passing_order(junction)
    jt_cliques = list(junction.nodes())
    messages = create_messages(generations, attrs)

    # Warm start: use previous model's raw theta (pre-BP) where cliques match
    init_potentials = None
    if prev_model is not None:
        init_potentials = {}
        for i, cl in enumerate(jt_cliques):
            if cl in prev_model.cliques:
                prev_idx = prev_model.cliques.index(cl)
                init_potentials[i] = prev_model.raw_theta[prev_idx]

    # Run mirror descent
    potentials, loss_fn, raw_theta = mirror_descent(
        jt_cliques, messages, obs_list, attrs,
        device=device, init_potentials=init_potentials, **params
    )

    return FittedPGM(potentials, raw_theta, jt_cliques, clique_names, n, loss_fn)
