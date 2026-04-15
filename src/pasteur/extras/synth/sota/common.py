"""Shared utilities for MST and AIM: measurement, CDP, PGM fitting, sampling.

Cliques use ATTRIBUTE names (e.g., 'admittime', 'income'), not value names.
Multi-value attributes are handled internally by the oracle and AttrMeta."""

from __future__ import annotations

import logging
from math import exp, log1p
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from scipy.special import softmax

from ....attribute import Attributes, DatasetAttributes
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
# Attribute helpers
# ============================================================
#
# AIM/MST/PrivMRF operate on "columns" — one per CatValue.
# Multi-value attributes (e.g. admittime with 4 values) are split
# into separate columns so the PGM domain stays manageable.
# Attributes with a common value (nullable) are kept combined
# because the null/non-null gating requires a joint representation.
#
# A "column name" is the VALUE name for split attrs, or the
# ATTRIBUTE name for single-value / common attrs.
# ============================================================

def _is_split_attr(attr) -> bool:
    """Whether an attribute should be split into per-value columns."""
    return len(attr.vals) > 1 and not attr.common


def _col_to_attr_sel(col_name: str, attrs: DatasetAttributes):
    """Map a column name to (attribute_name, oracle_sel).

    For split attrs, col_name is a value name → sel = {col_name: 0}.
    For combined attrs, col_name is the attr name → sel = {v: 0 for all v}."""
    all_attrs = cast(Attributes, attrs[None])
    # Check if col_name is an attribute name
    if col_name in all_attrs:
        attr = all_attrs[col_name]
        if not _is_split_attr(attr):
            return col_name, {v: 0 for v in attr.vals}
    # Otherwise it's a value name from a split attribute
    for attr_name, attr in all_attrs.items():
        if _is_split_attr(attr) and col_name in attr.vals:
            return attr_name, {col_name: 0}
    raise KeyError(f"Column '{col_name}' not found in attributes")


def _attr_sel(col_name: str, attrs: DatasetAttributes):
    """Build the oracle selector for a column at height 0."""
    _, sel = _col_to_attr_sel(col_name, attrs)
    return sel


def attr_domain_size(col_name: str, attrs: DatasetAttributes) -> int:
    """Domain size for a column at height 0."""
    attr_name, sel = _col_to_attr_sel(col_name, attrs)
    attr = cast(Attributes, attrs[None])[attr_name]
    return attr.get_domain(sel)


def clique_domain_size(
    clique: tuple[str, ...], attrs: DatasetAttributes
) -> int:
    """Domain size for a clique of column names at height 0."""
    dom = 1
    for col_name in clique:
        dom *= attr_domain_size(col_name, attrs)
    return dom


def get_attr_names(attrs: DatasetAttributes) -> list[str]:
    """Get column names for the PGM model.

    Multi-value attributes without common values are split into
    per-value columns for manageable domain sizes. Single-value
    and common-value attributes are kept as one column."""
    result = []
    for attr_name, attr in cast(Attributes, attrs[None]).items():
        if not attr.vals:
            continue
        if _is_split_attr(attr):
            result.extend(attr.vals.keys())
        else:
            result.append(attr_name)
    return result


# ============================================================
# Measurement
# ============================================================
class Measurement(NamedTuple):
    """A noisy measurement of a marginal clique (attribute names)."""

    clique: tuple[str, ...]
    noisy: np.ndarray  # noisy count vector (flat)
    sigma: float  # noise standard deviation


def measure(
    oracle: MarginalOracle,
    attrs: DatasetAttributes,
    cliques: list[tuple[str, ...]],
    sigma: float,
    weights: np.ndarray | None = None,
) -> list[Measurement]:
    """Measure marginals at height 0 with Gaussian noise.

    Cliques are tuples of attribute names."""
    if weights is None:
        weights = np.ones(len(cliques))
    weights = weights / np.linalg.norm(weights)

    # Build oracle requests — column names map back to (attr_name, sel) pairs.
    # Multiple columns from the same split attribute in one clique are merged
    # into a single oracle request entry with a combined sel.
    requests = []
    for clique in cliques:
        req = {}
        for col_name in clique:
            attr_name, sel = _col_to_attr_sel(col_name, attrs)
            if attr_name in req:
                req[attr_name].update(sel)
            else:
                req[attr_name] = dict(sel)
        requests.append(list(req.items()))

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
        self.raw_theta = raw_theta
        self.cliques = cliques
        self.clique_names = clique_names
        self.n = n
        self.loss_fn = loss_fn

    def _build_source(self, clique: tuple[str, ...], attrs: DatasetAttributes):
        """Build AttrMeta source tuple for a column-name clique."""
        from ....graph.hugin import AttrMeta
        # Merge columns belonging to the same attribute
        attr_sels: dict[str, dict[str, int]] = {}
        for col_name in clique:
            attr_name, sel = _col_to_attr_sel(col_name, attrs)
            if attr_name in attr_sels:
                attr_sels[attr_name].update(sel)
            else:
                attr_sels[attr_name] = dict(sel)

        source = []
        for attr_name, sel_dict in attr_sels.items():
            sel_tuple: int | tuple = tuple(sorted(sel_dict.items()))
            source.append(AttrMeta(None, None, attr_name, sel_tuple))
        return tuple(sorted(source, key=lambda x: x[:-1]))

    def project(
        self, clique: tuple[str, ...], attrs: DatasetAttributes
    ) -> np.ndarray:
        """Project fitted model to a marginal over the given attrs."""
        from ....graph.loss import get_parents
        from ....graph.hugin import get_clique_domain

        source_tuple = self._build_source(clique, attrs)

        parents = get_parents(source_tuple, self.cliques)
        if not parents:
            if len(clique) == 1:
                # Single attr not in any clique — return uniform
                logger.warning(
                    f"project({clique}): single attr not in tree, returning uniform"
                )
                dom = attr_domain_size(clique[0], attrs)
                return np.ones(dom) / dom * self.n
            # No single clique contains all attrs — approximate with
            # independence: outer product of per-attribute marginals
            marginals = []
            for attr_name in clique:
                m = self.project((attr_name,), attrs).ravel()
                m = m / m.sum() if m.sum() > 0 else m
                marginals.append(m)
            result = marginals[0]
            for m in marginals[1:]:
                result = np.outer(result, m).ravel()
            return result * self.n

        parent = min(parents, key=lambda x: get_clique_domain(x, attrs))
        parent_idx = self.cliques.index(parent)
        proc = self.potentials[parent_idx].copy()

        # Sum out dims not in the requested clique
        requested = {a.attr for a in source_tuple}
        remaining = [a for a in parent if a.attr in requested]
        sum_dims = tuple(
            i for i, a in enumerate(parent) if a.attr not in requested
        )
        if sum_dims:
            proc = proc.sum(axis=sum_dims)

        # Transpose from sorted AttrMeta order back to requested clique order.
        # Use attribute names (not column names) for matching.
        sorted_names = [a.attr for a in remaining]
        req_attr_names = []
        for col_name in clique:
            aname, _ = _col_to_attr_sel(col_name, attrs)
            if aname not in req_attr_names:
                req_attr_names.append(aname)
        if sorted_names != req_attr_names:
            perm = [sorted_names.index(a) for a in req_attr_names]
            proc = proc.transpose(perm)

        return proc * self.n

    def synthetic_data(
        self, n: int, attrs: DatasetAttributes
    ) -> pd.DataFrame:
        """Generate synthetic data by independent sampling per column.

        Columns correspond to get_attr_names(): per-value for split
        multi-value attributes, per-attribute otherwise."""
        all_attrs = cast(Attributes, attrs[None])
        columns = {}

        for col_name in get_attr_names(attrs):
            attr_name, sel = _col_to_attr_sel(col_name, attrs)
            attr = all_attrs[attr_name]

            marginal = self.project((col_name,), attrs).ravel()
            marginal = marginal.clip(0)
            total = marginal.sum()
            if total > 0:
                probs = marginal / total
            else:
                probs = np.ones(len(marginal)) / len(marginal)
            flat_idx = np.random.choice(len(probs), size=n, p=probs)

            if len(sel) == 1:
                # Single-value column: flat index is the column value
                val_name = next(iter(sel))
                columns[val_name] = flat_idx
            else:
                # Combined multi-value column (common-value attrs):
                # decompose into per-value columns.
                from ....graph.sample import _decompose_dim

                decomposed = _decompose_dim(attr, sel, flat_idx)
                for vname in attr.vals:
                    if vname in decomposed:
                        columns[vname] = decomposed[vname]

        return pd.DataFrame(columns)


def fit_pgm(
    attrs: DatasetAttributes,
    measurements: list[Measurement],
    n: int,
    md_params: dict | None = None,
    prev_model: FittedPGM | None = None,
) -> FittedPGM:
    """Fit a PGM model from measurements using our mirror descent + BP.

    Measurement cliques are attribute-name tuples."""
    from ....graph.beliefs import create_messages, convert_sel
    from ....graph.hugin import (
        AttrMeta,
        get_attrs,
        get_junction_tree_from_cliques,
        get_message_passing_order,
    )
    from ....graph.loss import LinearObservation
    from ....graph.mirror_descent import mirror_descent

    params = {**MIRROR_DESCENT_DEFAULT, **(md_params or {})}
    device = params.pop("device", "auto")
    device = None if device == "auto" else device
    params.pop("compress", None)
    params.pop("sample", None)
    params.pop("tree", None)

    # Build CliqueMeta for each measurement (column-name cliques).
    # Column names may be value names (split attrs) or attr names (combined).
    # Multiple columns from the same attribute are merged into one AttrMeta
    # with a combined sel.
    obs_list = []
    clique_metas = []
    clique_names = []
    for meas in measurements:
        # Merge columns belonging to the same attribute
        attr_sels: dict[str, dict[str, int]] = {}
        for col_name in meas.clique:
            attr_name, sel = _col_to_attr_sel(col_name, attrs)
            if attr_name in attr_sels:
                attr_sels[attr_name].update(sel)
            else:
                attr_sels[attr_name] = dict(sel)

        source = []
        for attr_name, sel_dict in attr_sels.items():
            sel_tuple: int | tuple = tuple(sorted(sel_dict.items()))
            source.append(AttrMeta(None, None, attr_name, sel_tuple))
        source_tuple = tuple(sorted(source, key=lambda x: x[:-1]))
        clique_metas.append(source_tuple)
        clique_names.append(meas.clique)

        # The oracle returns data in the request order (meas.clique order).
        # AttrMeta is sorted alphabetically. Transpose to match.
        obs = meas.noisy.copy()

        # Build per-dim sizes for request order (column order)
        req_dims = []
        for col_name in meas.clique:
            req_dims.append(attr_domain_size(col_name, attrs))

        src_dims = []
        for a in source_tuple:
            attr = get_attrs(attrs, a.table, a.order)[a.attr]
            src_dims.append(attr.get_domain(convert_sel(a.sel)))

        # Reshape to request order, transpose to source order.
        # Map column names to attr names for matching.
        obs = obs.reshape(req_dims)
        src_attr_names = [a.attr for a in source_tuple]
        req_col_to_attr = []
        for col_name in meas.clique:
            aname, _ = _col_to_attr_sel(col_name, attrs)
            if aname not in req_col_to_attr:
                req_col_to_attr.append(aname)
        if req_col_to_attr != src_attr_names:
            perm = [req_col_to_attr.index(a) for a in src_attr_names]
            obs = obs.transpose(perm)

        obs = obs.reshape(src_dims)
        obs = obs.clip(0)
        obs_sum = obs.sum()
        if obs_sum > 0:
            obs = obs / obs_sum
        obs = obs.astype(np.float32)

        confidence = n / (n + meas.sigma * obs.size)
        obs_list.append(LinearObservation(source_tuple, None, obs, confidence))

    # Deduplicate clique metas
    unique_metas = list(dict.fromkeys(clique_metas))

    # Build junction tree from unique observation cliques
    junction = get_junction_tree_from_cliques(unique_metas)
    generations = get_message_passing_order(junction)
    jt_cliques = list(junction.nodes())
    messages = create_messages(generations, attrs)

    # Warm start: use previous model's raw theta where cliques match
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
