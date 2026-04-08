"""Junction tree sampler for fitted clique potentials.

After mirror descent (or any inference method) produces consistent clique
potentials, this module samples synthetic rows by traversing the junction
tree top-down: sample the root clique from its joint, then condition each
child on the separator values from its parent.
"""

import logging
from collections import deque
from typing import NamedTuple, Sequence

import networkx as nx
import numpy as np

from ..attribute import CatValue, DatasetAttributes, get_dtype
from .beliefs import convert_sel, has_attr, is_same
from .hugin import AttrMeta, CliqueMeta, get_attrs

logger = logging.getLogger(__name__)


class DimInfo(NamedTuple):
    """Metadata for one dimension of a clique."""

    attr_meta: AttrMeta
    domain: int


class SeparatorDim(NamedTuple):
    """One shared dimension between parent and child."""

    parent_dim: int
    child_dim: int
    # Mapping from parent compressed index -> child compressed index.
    # None if domains match exactly.
    index_map: np.ndarray | None


class ChildInfo(NamedTuple):
    """How to condition a child clique on its parent."""

    parent_idx: int
    child_idx: int
    separator: tuple[SeparatorDim, ...]
    # Child dimensions that need to be sampled (not in separator)
    sample_dims: tuple[int, ...]


class SamplerMeta(NamedTuple):
    """Pre-computed metadata for junction tree sampling."""

    cliques: list[CliqueMeta]
    # Per-clique dimension info
    dim_info: list[list[DimInfo]]
    # Root clique index
    root: int
    # BFS traversal: list of ChildInfo (root is not included)
    children: list[ChildInfo]
    # For each (table, order, attr): which clique index owns it
    # (the first clique in BFS order that has this attribute)
    owners: dict[tuple, int]


def _build_index_map(
    parent_meta: AttrMeta,
    child_meta: AttrMeta,
    attrs: DatasetAttributes,
) -> np.ndarray | None:
    """Build a mapping from parent's compressed domain to child's compressed domain.

    Returns None if both use the same sel (domains match)."""
    if parent_meta.sel == child_meta.sel:
        return None

    attr = get_attrs(attrs, parent_meta.table, parent_meta.order)[parent_meta.attr]
    p_map = attr.get_mapping(convert_sel(parent_meta.sel))
    c_map = attr.get_mapping(convert_sel(child_meta.sel))
    p_dom = attr.get_domain(convert_sel(parent_meta.sel))

    # For each parent compressed index, find a raw index, then look up child compressed
    index_map = np.zeros(p_dom, dtype=get_dtype(max(1, int(c_map.max()))))
    for pi in range(p_dom):
        raw = int(np.argmax(p_map == pi))
        index_map[pi] = c_map[raw]

    return index_map


def create_sampler_meta(
    junction: nx.Graph,
    cliques: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
) -> SamplerMeta:
    """Pre-compute all metadata needed for junction tree sampling."""
    cliques = list(cliques)

    # Build dimension info per clique
    dim_info = []
    for cl in cliques:
        dims = []
        for meta in cl:
            dom = get_attrs(attrs, meta.table, meta.order)[meta.attr].get_domain(
                convert_sel(meta.sel)
            )
            dims.append(DimInfo(meta, dom))
        dim_info.append(dims)

    # Pick root: largest total domain for maximal joint coverage
    root = max(range(len(cliques)), key=lambda i: sum(d.domain for d in dim_info[i]))

    # BFS from root
    visited = {root}
    queue = deque([root])
    children: list[ChildInfo] = []
    owners: dict[tuple, int] = {}

    # Root owns all its attributes
    for meta in cliques[root]:
        key = (meta.table, meta.order, meta.attr)
        owners[key] = root

    while queue:
        parent_idx = queue.popleft()
        parent_cl = cliques[parent_idx]

        for neighbor_cl in junction.neighbors(parent_cl):
            child_idx = cliques.index(neighbor_cl)
            if child_idx in visited:
                continue
            visited.add(child_idx)
            queue.append(child_idx)

            child_cl = cliques[child_idx]

            # Find separator dims and sample dims
            separator = []
            sample_dims = []
            for ci, child_meta in enumerate(child_cl):
                found = False
                for pi, parent_meta in enumerate(parent_cl):
                    if is_same(child_meta, parent_meta):
                        idx_map = _build_index_map(parent_meta, child_meta, attrs)
                        separator.append(SeparatorDim(pi, ci, idx_map))
                        found = True
                        break
                if not found:
                    sample_dims.append(ci)

            children.append(
                ChildInfo(parent_idx, child_idx, tuple(separator), tuple(sample_dims))
            )

            # Child owns its non-separator attributes
            for ci in sample_dims:
                meta = child_cl[ci]
                key = (meta.table, meta.order, meta.attr)
                if key not in owners:
                    owners[key] = child_idx

    return SamplerMeta(cliques, dim_info, root, children, owners)


def sample_junction_tree(
    potentials: Sequence[np.ndarray],
    meta: SamplerMeta,
    n: int,
) -> dict[tuple, np.ndarray]:
    """Sample n rows from fitted clique potentials.

    Args:
        potentials: List of numpy arrays (one per clique), in probability
            space (non-negative, each sums to ~1).
        meta: Pre-computed sampler metadata from create_sampler_meta.
        n: Number of rows to sample.

    Returns:
        Dict mapping (table, order, attr) to numpy arrays of compressed
        domain indices. Use reverse_map to convert to column values.
    """
    # Per-clique sampled indices: clique_idx -> {dim_idx: array of n indices}
    sampled: list[dict[int, np.ndarray]] = [{} for _ in meta.cliques]

    # Step 1: Sample root clique from joint distribution
    root_p = potentials[meta.root].copy()
    root_p = root_p.clip(min=0)
    root_sum = root_p.sum()
    if root_sum < 1e-12:
        logger.warning("Root clique has near-zero total probability, sampling uniform.")
        root_p = np.ones_like(root_p)
        root_sum = root_p.sum()
    root_p /= root_sum

    flat = root_p.ravel()
    indices = np.random.choice(len(flat), size=n, p=flat)
    per_dim = np.unravel_index(indices, root_p.shape)
    for dim_idx, dim_vals in enumerate(per_dim):
        sampled[meta.root][dim_idx] = dim_vals

    # Step 2: Sample children in BFS order
    for child_info in meta.children:
        cidx = child_info.child_idx
        child_p = potentials[cidx].copy()
        child_p = child_p.clip(min=0)

        # Get separator values from BFS parent, translated to child domain
        pidx = child_info.parent_idx
        sep_vals = []
        sep_child_dims = []
        sep_domains = []
        for sep in child_info.separator:
            parent_vals = sampled[pidx][sep.parent_dim]
            if sep.index_map is not None:
                child_vals = sep.index_map[parent_vals]
            else:
                child_vals = parent_vals
            sep_vals.append(child_vals)
            sep_child_dims.append(sep.child_dim)
            sep_domains.append(meta.dim_info[cidx][sep.child_dim].domain)

        # Also store separator values in sampled
        for sep, vals in zip(child_info.separator, sep_vals):
            sampled[cidx][sep.child_dim] = vals

        sample_dims = child_info.sample_dims
        if not sample_dims:
            # All dims are separator, nothing to sample
            continue

        sample_shape = tuple(meta.dim_info[cidx][d].domain for d in sample_dims)
        sample_size = int(np.prod(sample_shape))

        # Encode separator values into a single group key
        if len(sep_vals) == 1:
            group_keys = sep_vals[0]
        else:
            group_keys = np.zeros(n, dtype=np.int64)
            mul = 1
            for sv, dom in zip(sep_vals, sep_domains):
                group_keys += sv.astype(np.int64) * mul
                mul *= dom

        # Sample per group
        out_dims = {d: np.empty(n, dtype=get_dtype(max(1, s - 1))) for d, s in zip(sample_dims, sample_shape)}

        for group_key in np.unique(group_keys):
            mask = group_keys == group_key
            group_n = int(mask.sum())

            # Extract conditional: fix separator dims, get distribution over sample dims
            # Build index tuple to slice the child potential
            idx = [slice(None)] * len(child_p.shape)
            if len(sep_vals) == 1:
                idx[sep_child_dims[0]] = int(group_key)
            else:
                remaining = int(group_key)
                for sd, dom in zip(sep_child_dims, sep_domains):
                    idx[sd] = remaining % dom
                    remaining //= dom

            conditional = child_p[tuple(idx)]

            # conditional now has shape with only the sample_dims remaining
            # (separator dims are scalar-indexed away)
            cond_flat = conditional.ravel()
            cond_sum = cond_flat.sum()
            if cond_sum < 1e-12:
                # Fallback to uniform
                cond_flat = np.ones_like(cond_flat)
                cond_sum = cond_flat.sum()
            cond_flat /= cond_sum

            group_indices = np.random.choice(len(cond_flat), size=group_n, p=cond_flat)
            group_per_dim = np.unravel_index(group_indices, conditional.shape)

            # Map back: conditional.shape corresponds to the sample_dims in order
            for j, dim_idx in enumerate(sample_dims):
                out_dims[dim_idx][mask] = group_per_dim[j]

        for dim_idx, arr in out_dims.items():
            sampled[cidx][dim_idx] = arr

    # Step 3: Collect output columns (one per unique attribute)
    output: dict[tuple, np.ndarray] = {}
    for cl_idx, owner_clique in enumerate(meta.cliques):
        for dim_idx, meta_dim in enumerate(meta.dim_info[cl_idx]):
            key = (meta_dim.attr_meta.table, meta_dim.attr_meta.order, meta_dim.attr_meta.attr)
            if meta.owners.get(key) != cl_idx:
                continue
            if dim_idx in sampled[cl_idx]:
                output[key] = sampled[cl_idx][dim_idx]

    return output



def reverse_map_columns(
    sampled: dict[tuple, np.ndarray],
    cliques: Sequence[CliqueMeta],
    meta: SamplerMeta,
    attrs: DatasetAttributes,
) -> dict[str, np.ndarray]:
    """Convert compressed domain indices to original column values.

    Args:
        sampled: Output of sample_junction_tree.
        cliques: List of clique metadata.
        meta: Sampler metadata.
        attrs: Dataset attribute metadata.

    Returns:
        Dict mapping column value names (e.g. 'admittime_week') to numpy
        arrays of raw domain indices.
    """
    output: dict[str, np.ndarray] = {}

    for (table, order, attr_name), indices in sampled.items():
        attr = get_attrs(attrs, table, order)[attr_name]
        owner_idx = meta.owners[(table, order, attr_name)]
        # Find the AttrMeta for this attribute in the owner clique
        owner_cl = cliques[owner_idx]
        attr_meta = None
        for am in owner_cl:
            if am.table == table and am.order == order and am.attr == attr_name:
                attr_meta = am
                break
        assert attr_meta is not None

        sel = convert_sel(attr_meta.sel)

        if isinstance(sel, int):
            # Common value at a height — upsample to height 0
            cmn = attr.common
            assert cmn is not None
            output[cmn.name] = cmn.upsample(indices, sel)
        else:
            # Multi-value: the combined index maps to individual values
            # Decompose the combined index into per-value indices
            vals = list(sel.keys())
            cat_vals = [attr[v] for v in vals]
            assert all(isinstance(cv, CatValue) for cv in cat_vals)

            remaining = indices.copy().astype(np.int64)
            for v, cv in zip(vals, cat_vals):
                h = sel[v]
                dom = cv.get_domain(h)
                per_val = (remaining % dom).astype(get_dtype(max(1, dom - 1)))
                remaining //= dom
                output[v] = cv.upsample(per_val, h)

    return output
