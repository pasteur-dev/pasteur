"""Junction tree sampler for fitted clique potentials.

After mirror descent (or any inference method) produces consistent clique
potentials, this module samples synthetic rows by traversing the junction
tree top-down: sample the root clique from its joint, then condition each
child on the separator values from its parent.
"""

import logging
from collections import defaultdict, deque
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
    """One shared dimension between parent and child (exact match)."""

    parent_dim: int
    child_dim: int
    # Mapping from parent compressed index -> child compressed index.
    # None if domains match exactly.
    index_map: np.ndarray | None


class MaskedDim(NamedTuple):
    """A shared dimension where parent is coarser than child.

    The child dimension is sampled (not fixed), but constrained to values
    compatible with the parent's coarse bin."""

    parent_dim: int
    child_dim: int
    child_domain: int
    # For each parent compressed index, a boolean mask of shape (child_domain,)
    # indicating which child values are compatible.
    masks: np.ndarray  # shape (parent_domain, child_domain)


class ChildInfo(NamedTuple):
    """How to condition a child clique on its parent."""

    parent_idx: int
    child_idx: int
    separator: tuple[SeparatorDim, ...]
    # Child dimensions that need to be sampled (not in separator)
    sample_dims: tuple[int, ...]
    # Dimensions shared with parent but at different resolutions
    masked_dims: tuple[MaskedDim, ...]


class ValueOwner(NamedTuple):
    """Which clique dimension provides a specific value at the finest height."""

    clique_idx: int
    dim_idx: int
    height: int


class SamplerMeta(NamedTuple):
    """Pre-computed metadata for junction tree sampling."""

    cliques: list[CliqueMeta]
    dim_info: list[list[DimInfo]]
    root: int
    children: list[ChildInfo]
    # For each (table, order, attr): which clique dim to use for sampling
    # (the clique that first appears in BFS — for separator consistency)
    attr_owners: dict[tuple, int]
    # For each (table, order, attr, value_name): best clique + dim + height
    # (the clique with the lowest height for this value)
    value_owners: dict[tuple, ValueOwner]


def _build_index_map(
    parent_meta: AttrMeta,
    child_meta: AttrMeta,
    attrs: DatasetAttributes,
) -> np.ndarray | None:
    """Build a mapping from parent's compressed domain to child's compressed domain.

    Returns None if both use the same sel (domains match exactly)."""
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


def _build_mask_map(
    parent_meta: AttrMeta,
    child_meta: AttrMeta,
    attrs: DatasetAttributes,
) -> np.ndarray:
    """Build a boolean mask (parent_domain, child_domain) for coarser parent.

    mask[pi, ci] is True if child compressed index ci is compatible with
    parent compressed index pi."""
    attr = get_attrs(attrs, parent_meta.table, parent_meta.order)[parent_meta.attr]
    p_map = attr.get_mapping(convert_sel(parent_meta.sel))
    c_map = attr.get_mapping(convert_sel(child_meta.sel))
    p_dom = attr.get_domain(convert_sel(parent_meta.sel))
    c_dom = attr.get_domain(convert_sel(child_meta.sel))

    mask = np.zeros((p_dom, c_dom), dtype=bool)
    for raw_idx in range(len(p_map)):
        mask[p_map[raw_idx], c_map[raw_idx]] = True

    return mask


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
    attr_owners: dict[tuple, int] = {}

    # Root owns all its attributes
    for meta in cliques[root]:
        key = (meta.table, meta.order, meta.attr)
        attr_owners[key] = root

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

            # Find separator dims, masked dims, and sample dims
            separator = []
            masked = []
            sample_dims = []
            for ci, child_meta in enumerate(child_cl):
                found = False
                for pi, parent_meta in enumerate(parent_cl):
                    if is_same(child_meta, parent_meta):
                        if parent_meta.sel == child_meta.sel:
                            # Exact match — fix child value from parent
                            separator.append(SeparatorDim(pi, ci, None))
                        else:
                            # Different resolutions — check if 1-to-1 or coarser
                            idx_map = _build_index_map(parent_meta, child_meta, attrs)
                            assert idx_map is not None
                            p_dom = len(idx_map)
                            c_dom = dim_info[child_idx][ci].domain
                            if p_dom >= c_dom:
                                # Parent is finer or equal — use direct map
                                separator.append(SeparatorDim(pi, ci, idx_map))
                            else:
                                # Parent is coarser — use mask so child
                                # can sample among compatible fine values
                                mask_map = _build_mask_map(
                                    parent_meta, child_meta, attrs
                                )
                                masked.append(
                                    MaskedDim(pi, ci, c_dom, mask_map)
                                )
                        found = True
                        break
                if not found:
                    sample_dims.append(ci)

            children.append(
                ChildInfo(
                    parent_idx, child_idx,
                    tuple(separator), tuple(sample_dims), tuple(masked),
                )
            )

            # Child owns its non-separator attributes
            for ci in sample_dims:
                meta = child_cl[ci]
                key = (meta.table, meta.order, meta.attr)
                if key not in attr_owners:
                    attr_owners[key] = child_idx

    # Build per-value ownership: for each individual value, find the clique
    # that has it at the lowest height (finest resolution)
    value_owners: dict[tuple, ValueOwner] = {}
    for cl_idx, cl in enumerate(cliques):
        for dim_idx, meta in enumerate(cl):
            sel = convert_sel(meta.sel)
            if isinstance(sel, int):
                # Common value
                vkey = (meta.table, meta.order, meta.attr, "__common__")
                if vkey not in value_owners or sel < value_owners[vkey].height:
                    value_owners[vkey] = ValueOwner(cl_idx, dim_idx, sel)
            else:
                for val_name, h in sel.items():
                    vkey = (meta.table, meta.order, meta.attr, val_name)
                    if vkey not in value_owners or h < value_owners[vkey].height:
                        value_owners[vkey] = ValueOwner(cl_idx, dim_idx, h)

    return SamplerMeta(cliques, dim_info, root, children, attr_owners, value_owners)


def _cond_dim(child_dim: int, sep_child_dims: list[int]) -> int:
    """Map a child clique dim index to its position in the conditional array
    (after separator dims have been removed by indexing)."""
    offset = sum(1 for sd in sep_child_dims if sd < child_dim)
    return child_dim - offset


def _cond_dims(dims: list[int], sep_child_dims: list[int]) -> list[int]:
    """Map multiple child dims to conditional positions."""
    return [_cond_dim(d, sep_child_dims) for d in dims]


def sample_junction_tree(
    potentials: Sequence[np.ndarray],
    meta: SamplerMeta,
    n: int,
) -> dict[int, dict[int, np.ndarray]]:
    """Sample n rows from fitted clique potentials.

    Returns:
        Dict mapping clique_idx -> {dim_idx: array of n sampled indices}.
    """
    sampled: dict[int, dict[int, np.ndarray]] = defaultdict(dict)

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
        pidx = child_info.parent_idx
        child_p = potentials[cidx].copy()
        child_p = child_p.clip(min=0)

        # Get exact separator values from BFS parent
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

        # Store exact separator values
        for sep, vals in zip(child_info.separator, sep_vals):
            sampled[cidx][sep.child_dim] = vals

        # Masked dims + sample dims are all sampled together
        # Masked dims are constrained by parent's coarse value
        masked_parent_vals = []
        for md in child_info.masked_dims:
            masked_parent_vals.append(sampled[pidx][md.parent_dim])

        all_sample_dims = list(child_info.sample_dims) + [
            md.child_dim for md in child_info.masked_dims
        ]
        if not all_sample_dims:
            continue

        all_sample_shape = tuple(
            meta.dim_info[cidx][d].domain for d in all_sample_dims
        )

        # Encode separator + masked parent values into group keys
        # (each unique combination needs separate conditional sampling)
        all_group_vals = list(sep_vals) + [v.astype(np.int64) for v in masked_parent_vals]
        all_group_domains = list(sep_domains) + [
            md.masks.shape[0] for md in child_info.masked_dims
        ]

        if len(all_group_vals) == 0:
            group_keys = np.zeros(n, dtype=np.int64)
        elif len(all_group_vals) == 1:
            group_keys = all_group_vals[0].astype(np.int64)
        else:
            group_keys = np.zeros(n, dtype=np.int64)
            mul = 1
            for sv, dom in zip(all_group_vals, all_group_domains):
                group_keys += sv.astype(np.int64) * mul
                mul *= dom

        # Sample per group
        out_dims = {
            d: np.empty(n, dtype=get_dtype(max(1, s - 1)))
            for d, s in zip(all_sample_dims, all_sample_shape)
        }

        for group_key in np.unique(group_keys):
            row_mask = group_keys == group_key
            group_n = int(row_mask.sum())

            # Extract conditional: fix exact separator dims
            idx = [slice(None)] * len(child_p.shape)
            remaining = int(group_key)
            for sd, dom in zip(sep_child_dims, sep_domains):
                idx[sd] = remaining % dom
                remaining //= dom

            # Get masked parent values for this group
            masked_parent_group = []
            for md in child_info.masked_dims:
                masked_parent_group.append(remaining % md.masks.shape[0])
                remaining //= md.masks.shape[0]

            conditional = child_p[tuple(idx)]

            # Apply masks for coarse-parent dims: zero out incompatible values
            for md, parent_val in zip(child_info.masked_dims, masked_parent_group):
                # Find dimension index within conditional (after removing sep dims)
                cond_dim = _cond_dim(md.child_dim, sep_child_dims)
                dim_mask = md.masks[parent_val]
                # Broadcast mask to conditional shape
                shape = [1] * conditional.ndim
                shape[cond_dim] = md.child_domain
                conditional = conditional * dim_mask.reshape(shape)

            cond_flat = conditional.ravel()
            cond_sum = cond_flat.sum()
            if cond_sum < 1e-12:
                cond_flat = np.ones_like(cond_flat)
                cond_sum = cond_flat.sum()
            cond_flat /= cond_sum

            group_indices = np.random.choice(len(cond_flat), size=group_n, p=cond_flat)
            group_per_dim = np.unravel_index(group_indices, conditional.shape)

            # Map back from conditional dim indices to child clique dim indices
            cond_dim_list = _cond_dims(all_sample_dims, sep_child_dims)
            for j, dim_idx in enumerate(all_sample_dims):
                out_dims[dim_idx][row_mask] = group_per_dim[cond_dim_list[j]]

        for dim_idx, arr in out_dims.items():
            sampled[cidx][dim_idx] = arr

    return dict(sampled)


def reverse_map_columns(
    sampled: dict[int, dict[int, np.ndarray]],
    meta: SamplerMeta,
    attrs: DatasetAttributes,
) -> dict[str, np.ndarray]:
    """Convert sampled clique indices to output column values.

    For each value in each attribute, finds the clique that has it at the
    finest height (lowest h), decomposes the combined domain index, and
    extracts the per-value index at that height. No upsampling is used;
    values are output at whatever resolution the clique tree provides.

    Args:
        sampled: Output of sample_junction_tree (clique_idx -> dim_idx -> indices).
        meta: Sampler metadata.
        attrs: Dataset attribute metadata.

    Returns:
        Dict mapping column value names to numpy arrays of indices at the
        finest available height.
    """
    output: dict[str, np.ndarray] = {}

    for (table, order, attr_name, val_name), owner in meta.value_owners.items():
        if val_name == "__common__":
            continue  # common values are derived from their children

        cl_idx = owner.clique_idx
        dim_idx = owner.dim_idx
        if cl_idx not in sampled or dim_idx not in sampled[cl_idx]:
            continue

        combined_idx = sampled[cl_idx][dim_idx]
        attr_meta = meta.dim_info[cl_idx][dim_idx].attr_meta
        sel = convert_sel(attr_meta.sel)

        if isinstance(sel, int):
            # Common-only dimension: this value is the common value itself
            attr = get_attrs(attrs, table, order)[attr_name]
            cmn = attr.common
            assert cmn is not None
            # Output the compressed index directly (at this height)
            output[cmn.name] = combined_idx
        else:
            # Multi-value dimension: decompose combined index to find this value
            # The combined index encodes values in sel order (alphabetical)
            remaining = combined_idx.copy().astype(np.int64)
            for v_name in sel:
                attr = get_attrs(attrs, table, order)[attr_name]
                cv = attr[v_name]
                assert isinstance(cv, CatValue)
                h = sel[v_name]
                dom = cv.get_domain(h)
                per_val = (remaining % dom).astype(get_dtype(max(1, dom - 1)))
                remaining //= dom
                if v_name == val_name:
                    output[val_name] = per_val
                    break

    return output
