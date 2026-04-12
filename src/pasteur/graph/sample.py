"""Junction tree sampler for fitted clique potentials.

After mirror descent (or any inference method) produces consistent clique
potentials, this module samples synthetic rows by traversing the junction
tree top-down: sample the root clique from its joint, then condition each
child on the separator values from its parent.  Whenever a clique dimension
has height-0 values, those are extracted directly into the output dict.
"""

import logging
from collections import deque
from typing import NamedTuple, Sequence

import networkx as nx
import numpy as np

from ..attribute import CatValue, DatasetAttributes, get_dtype
from .beliefs import convert_sel, is_same
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


class SamplerMeta(NamedTuple):
    """Pre-computed metadata for junction tree sampling."""

    cliques: list[CliqueMeta]
    dim_info: list[list[DimInfo]]
    root: int
    children: list[ChildInfo]


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

    # For each parent compressed index, find a raw index, then look up child compressed.
    # Use argsort to find the first raw index for each parent bin in one pass.
    order = np.argsort(p_map, kind="stable")
    # order[0] is the raw index of the smallest p_map value (0), etc.
    # Group boundaries: first occurrence of each parent bin.
    first_raw = np.empty(p_dom, dtype=np.intp)
    first_raw[p_map[order]] = order  # last write per bin = last in order = highest raw
    # Reverse to get first occurrence (stable sort preserves raw order within bin)
    first_raw[p_map[order[::-1]]] = order[::-1]
    index_map = c_map[first_raw].astype(get_dtype(max(1, int(c_map.max()))))

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
    mask[p_map, c_map] = True

    return mask


def _can_match_as_separator_or_mask(
    parent_sel, child_sel
) -> str:
    """Classify how a parent/child dim pair should be matched.

    Only called after ``is_same`` confirms both dims refer to the same
    attribute, so ``_build_mask_map`` (raw-space mask) always works.

    Returns:
      "exact"   — same value names (both int or both tuple with equal
                   names): SeparatorDim (if parent finer) or domain-based
                   MaskedDim.  ``_build_index_map`` is safe.
      "partial" — same attribute but different value representation
                   (int vs tuple, or non-equal value names): always
                   MaskedDim via raw-space mask.
    """
    if isinstance(parent_sel, int) and isinstance(child_sel, int):
        return "exact"
    if isinstance(parent_sel, int) or isinstance(child_sel, int):
        return "partial"
    parent_names = {name for name, _h in parent_sel}
    child_names = {name for name, _h in child_sel}
    if child_names == parent_names:
        return "exact"
    return "partial"


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
                            found = True
                        else:
                            # Check if the child has value names not in
                            # the parent.  When the child's sel contains
                            # extra values (e.g. parent {A,B} vs child
                            # {A,B,C}), a SeparatorDim would pin the
                            # extra values to an arbitrary constant.
                            # Use a MaskedDim instead so the child can
                            # sample the missing values.
                            match_type = _can_match_as_separator_or_mask(
                                parent_meta.sel, child_meta.sel
                            )

                            if match_type == "exact":
                                # Same value names — safe to use index_map
                                # for SeparatorDim / domain-based MaskedDim
                                idx_map = _build_index_map(
                                    parent_meta, child_meta, attrs
                                )
                                assert idx_map is not None
                                p_dom = len(idx_map)
                                c_dom = dim_info[child_idx][ci].domain
                                if p_dom >= c_dom:
                                    separator.append(
                                        SeparatorDim(pi, ci, idx_map)
                                    )
                                else:
                                    mask_map = _build_mask_map(
                                        parent_meta, child_meta, attrs
                                    )
                                    masked.append(
                                        MaskedDim(pi, ci, c_dom, mask_map)
                                    )
                                found = True
                            elif match_type == "partial":
                                # Overlapping but not equal value names —
                                # always MaskedDim (index_map would collapse
                                # extra values in parent or child)
                                c_dom = dim_info[child_idx][ci].domain
                                mask_map = _build_mask_map(
                                    parent_meta, child_meta, attrs
                                )
                                masked.append(
                                    MaskedDim(pi, ci, c_dom, mask_map)
                                )
                                found = True
                            # else "none": no overlap — free sample_dim
                        if found:
                            break
                if not found:
                    sample_dims.append(ci)

            children.append(
                ChildInfo(
                    parent_idx, child_idx,
                    tuple(separator), tuple(sample_dims), tuple(masked),
                )
            )

    return SamplerMeta(cliques, dim_info, root, children)


def _cond_dim(child_dim: int, sep_child_dims: list[int]) -> int:
    """Map a child clique dim index to its position in the conditional array
    (after separator dims have been removed by indexing)."""
    offset = sum(1 for sd in sep_child_dims if sd < child_dim)
    return child_dim - offset


def _cond_dims(dims: list[int], sep_child_dims: list[int]) -> list[int]:
    """Map multiple child dims to conditional positions."""
    return [_cond_dim(d, sep_child_dims) for d in dims]


def _decompose_dim(
    attr,
    sel: dict,
    combined_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose a combined compressed dim index into per-value arrays.

    ``attr.get_mapping(sel)`` (multi-value) and ``attr.get_mapping({vn: h})``
    (single-value) both dispatch to the same ``CatValue.get_mapping_multiple``
    implementation, so they share the same raw-index ordering.  We exploit
    this: for each compressed bin we find the per-value encoded index by
    pairing the multi-value and single-value mappings at the same raw index.
    """
    comp_map = attr.get_mapping(sel)  # raw → comp_bin
    comp_dom = attr.get_domain(sel)

    # For each value in sel, compute a single-value mapping that uses the
    # SAME raw-index ordering as comp_map.
    val_maps: dict[str, np.ndarray] = {}
    for vn, h in sel.items():
        val_maps[vn] = attr.get_mapping({vn: h})  # raw → encoded_vn

    # Build comp_bin → per_value_encoded lookup (first-seen wins; within
    # a bin all raw indices share the same per-value encoded index because
    # the bin encodes a unique group combination).
    lookup: dict[str, np.ndarray] = {
        vn: np.zeros(comp_dom, dtype=np.int64) for vn in sel
    }
    seen = np.zeros(comp_dom, dtype=bool)
    for r in range(len(comp_map)):
        c = int(comp_map[r])
        if seen[c]:
            continue
        seen[c] = True
        for vn in sel:
            lookup[vn][c] = int(val_maps[vn][r])
        if seen.all():
            break

    # Apply lookup to sampled indices.
    result: dict[str, np.ndarray] = {}
    for vn in sel:
        result[vn] = lookup[vn][combined_idx.astype(np.int64)]

    return result


def _extract_h0(
    cl_idx: int,
    meta: SamplerMeta,
    attrs: DatasetAttributes,
    clique_vals: dict[int, np.ndarray],
    out: dict[str, np.ndarray],
):
    """Extract height-0 values from a clique's sampled dims into out.

    First-write wins: if a value name is already in out, skip it (earlier
    cliques in BFS order are better conditioned)."""
    cl = meta.cliques[cl_idx]

    for dim_idx, a_meta in enumerate(cl):
        if dim_idx not in clique_vals:
            continue

        combined_idx = clique_vals[dim_idx]
        sel = convert_sel(a_meta.sel)
        attr = get_attrs(attrs, a_meta.table, a_meta.order)[a_meta.attr]

        if isinstance(sel, int):
            # Common-only dim (gate variable): not a data column, skip.
            pass
        elif len(sel) == 1:
            # Single-value dim: combined_idx IS the per-value index
            vn, h = next(iter(sel.items()))
            if h == 0 and vn not in out:
                dom = max(2, int(combined_idx.max()) + 1) if len(combined_idx) > 0 else 2
                out[vn] = combined_idx.astype(get_dtype(max(1, dom - 1)))
        else:
            # Multi-value dim: check if any values are at h=0
            h0_vals = [vn for vn, h in sel.items() if h == 0 and vn not in out]
            if h0_vals:
                decomposed = _decompose_dim(attr, sel, combined_idx)
                for vn in h0_vals:
                    if vn in decomposed:
                        arr = decomposed[vn]
                        dom = max(2, int(arr.max()) + 1) if len(arr) > 0 else 2
                        out[vn] = arr.astype(get_dtype(max(1, dom - 1)))


def sample_junction_tree(
    potentials: Sequence[np.ndarray],
    meta: SamplerMeta,
    n: int,
    attrs: DatasetAttributes,
    evidence: dict[tuple[int, int], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Sample n rows from fitted clique potentials.

    Follows the junction tree elimination order (BFS from root).
    For the root, samples from the joint distribution.  For each child,
    conditions on shared separator values from its parent and samples
    the remaining dimensions.  Height-0 values are extracted into the
    output dict as each clique is processed.

    If ``evidence`` is provided, it maps ``(clique_idx, dim_idx)`` to
    per-row value arrays (compressed indices, shape ``(n,)``).  Evidence
    dims are fixed rather than sampled, and other dims are conditioned
    on them, this is used by MARE to condition on hist/parent values.

    Returns dict mapping column names to arrays of height-0 indices.
    """
    out: dict[str, np.ndarray] = {}
    # Per-clique sampled dim values: clique_idx -> {dim_idx: array}
    clique_vals: dict[int, dict[int, np.ndarray]] = {}

    # Step 1: Sample root clique (conditioned on evidence if present)
    root_p = potentials[meta.root].copy().clip(min=0)
    root_sum = root_p.sum()
    if root_sum < 1e-12:
        logger.warning("Root clique has near-zero total probability, sampling uniform.")
        root_p = np.ones_like(root_p)
        root_sum = root_p.sum()
    root_p /= root_sum

    # Collect evidence dims for the root
    root_ev_dims: list[int] = []
    root_ev_vals: list[np.ndarray] = []
    root_ev_domains: list[int] = []
    if evidence:
        for di in range(len(root_p.shape)):
            key = (meta.root, di)
            if key in evidence:
                root_ev_dims.append(di)
                root_ev_vals.append(evidence[key])
                root_ev_domains.append(root_p.shape[di])

    if not root_ev_dims:
        # No evidence — original joint sampling
        flat = root_p.ravel()
        indices = np.random.choice(len(flat), size=n, p=flat)
        per_dim = np.unravel_index(indices, root_p.shape)
        clique_vals[meta.root] = {i: arr for i, arr in enumerate(per_dim)}
    else:
        # Condition on evidence dims, sample the rest
        root_sample_dims = [d for d in range(len(root_p.shape)) if d not in root_ev_dims]
        root_dims: dict[int, np.ndarray] = {}
        for di, ev in zip(root_ev_dims, root_ev_vals):
            root_dims[di] = ev

        # Group by evidence values
        if len(root_ev_vals) == 1:
            root_group_keys = root_ev_vals[0].astype(np.int64)
        else:
            root_group_keys = np.zeros(n, dtype=np.int64)
            mul = 1
            for sv, dom in zip(root_ev_vals, root_ev_domains):
                root_group_keys += sv.astype(np.int64) * mul
                mul *= dom

        root_out_dims: dict[int, np.ndarray] = {}
        for d in root_sample_dims:
            root_out_dims[d] = np.empty(n, dtype=get_dtype(max(1, root_p.shape[d] - 1)))

        root_cond_dim_list = _cond_dims(root_sample_dims, root_ev_dims)

        for gk in np.unique(root_group_keys):
            row_mask = root_group_keys == gk
            group_n = int(row_mask.sum())

            idx = [slice(None)] * len(root_p.shape)
            remaining = int(gk)
            for sd, dom in zip(root_ev_dims, root_ev_domains):
                idx[sd] = remaining % dom
                remaining //= dom

            conditional = root_p[tuple(idx)]
            cond_flat = conditional.ravel()
            cond_sum = cond_flat.sum()
            if cond_sum < 1e-12:
                cond_flat = np.ones_like(cond_flat)
                cond_sum = cond_flat.sum()
            cond_flat /= cond_sum

            group_indices = np.random.choice(len(cond_flat), size=group_n, p=cond_flat)
            group_per_dim = np.unravel_index(group_indices, conditional.shape)
            for j, dim_idx in enumerate(root_sample_dims):
                root_out_dims[dim_idx][row_mask] = group_per_dim[root_cond_dim_list[j]]

        for di, arr in root_out_dims.items():
            root_dims[di] = arr
        clique_vals[meta.root] = root_dims

    _extract_h0(meta.root, meta, attrs, clique_vals[meta.root], out)

    # Step 2: Sample children in BFS order
    for child_info in meta.children:
        cidx = child_info.child_idx
        pidx = child_info.parent_idx
        child_p = potentials[cidx].copy().clip(min=0)

        dims: dict[int, np.ndarray] = {}

        # Fix separator dims from parent
        sep_child_dims: list[int] = []
        sep_vals: list[np.ndarray] = []
        sep_domains: list[int] = []
        for sep in child_info.separator:
            parent_vals = clique_vals[pidx][sep.parent_dim]
            child_vals = sep.index_map[parent_vals] if sep.index_map is not None else parent_vals
            dims[sep.child_dim] = child_vals
            sep_child_dims.append(sep.child_dim)
            sep_vals.append(child_vals)
            sep_domains.append(meta.dim_info[cidx][sep.child_dim].domain)

        # Add evidence dims for this clique (not already in separator)
        if evidence:
            for di in range(len(child_p.shape)):
                key = (cidx, di)
                if key in evidence and di not in sep_child_dims:
                    dims[di] = evidence[key]
                    sep_child_dims.append(di)
                    sep_vals.append(evidence[key])
                    sep_domains.append(meta.dim_info[cidx][di].domain)

        # Masked dims are sampled freely from the child potential
        # (conditioned on separators + evidence only).  Propagating
        # coarse→fine constraints through masks cascades BP inconsistency
        # — the same error that privbayes avoids by using per-node
        # conditionals.  Treating masked dims as free preserves the
        # elimination order while skipping lossy intermediate conditioning.
        all_sample_dims = list(child_info.sample_dims) + [
            md.child_dim for md in child_info.masked_dims
        ]
        # Remove any evidence dims that ended up in sample_dims/masked_dims
        all_sample_dims = [d for d in all_sample_dims if d not in sep_child_dims]

        if not all_sample_dims:
            clique_vals[cidx] = dims
            _extract_h0(cidx, meta, attrs, dims, out)
            continue

        all_sample_shape = tuple(
            meta.dim_info[cidx][d].domain for d in all_sample_dims
        )

        # Group keys from separator values only (no masked parent values)
        all_group_vals = list(sep_vals)
        all_group_domains = list(sep_domains)

        if not all_group_vals:
            group_keys = np.zeros(n, dtype=np.int64)
        elif len(all_group_vals) == 1:
            group_keys = all_group_vals[0].astype(np.int64)
        else:
            group_keys = np.zeros(n, dtype=np.int64)
            mul = 1
            for sv, dom in zip(all_group_vals, all_group_domains):
                group_keys += sv.astype(np.int64) * mul
                mul *= dom

        # Allocate output arrays for sampled dims
        out_dims: dict[int, np.ndarray] = {
            d: np.empty(n, dtype=get_dtype(max(1, s - 1)))
            for d, s in zip(all_sample_dims, all_sample_shape)
        }

        # Common-value gating: if a sample dim belongs to an attribute
        # with a common value, and a sibling value has already been
        # generated, derive the common category per row and mask
        # incompatible values before sampling.  This ensures that e.g.
        # deathtime_day is null exactly when deathtime_time is null.
        gate_cats: np.ndarray | None = None
        gate_masks: dict[int, np.ndarray] = {}  # dim -> (common_dom, dim_dom)
        gate_dom = 0
        child_cl = meta.cliques[cidx]
        for d in all_sample_dims:
            if gate_cats is not None:
                break
            a_meta = child_cl[d]
            sel = convert_sel(a_meta.sel)
            if isinstance(sel, int):
                continue
            attr = get_attrs(attrs, a_meta.table, a_meta.order)[a_meta.attr]
            if attr.common is None:
                continue
            for sib_name, sib_val in attr.vals.items():
                if sib_name not in out or not isinstance(sib_val, CatValue):
                    continue
                # Derive common category from the already-generated sibling
                sib_ameta = AttrMeta(a_meta.table, a_meta.order, a_meta.attr, ((sib_name, 0),))
                common_ameta = AttrMeta(a_meta.table, a_meta.order, a_meta.attr, 0)
                idx_map = _build_index_map(sib_ameta, common_ameta, attrs)
                gate_dom = attr.common.get_domain(0)
                if idx_map is not None:
                    gate_cats = idx_map[out[sib_name]].astype(np.int64)
                else:
                    gate_cats = out[sib_name].astype(np.int64)
                # Build masks for all dims from the same attribute
                for d2 in all_sample_dims:
                    a2 = child_cl[d2]
                    if (a2.table, a2.order, a2.attr) != (a_meta.table, a_meta.order, a_meta.attr):
                        continue
                    s2 = convert_sel(a2.sel)
                    if isinstance(s2, int):
                        continue
                    gate_masks[d2] = _build_mask_map(common_ameta, a2, attrs)
                break

        if not gate_masks:
            # Vectorized path: build a (num_groups, cond_size) matrix from the
            # potential, normalize each row, compute CDFs, and sample all n
            # rows at once via inverse-CDF + searchsorted/broadcast.

            # Transpose potential: reversed sep dims first (to match the
            # column-major group_keys encoding), then sample dims.
            perm = list(reversed(sep_child_dims)) + all_sample_dims
            child_p_t = child_p.transpose(perm)
            group_product = (
                int(np.prod([child_p.shape[d] for d in sep_child_dims]))
                if sep_child_dims
                else 1
            )
            cond_product = int(np.prod(all_sample_shape))

            matrix = child_p_t.reshape(group_product, cond_product).astype(
                np.float64
            )

            # Normalize rows (uniform fallback for zero-sum rows)
            row_sums = matrix.sum(axis=1, keepdims=True)
            zero_rows = (row_sums < 1e-12).ravel()
            if zero_rows.any():
                matrix[zero_rows] = 1.0
                row_sums[zero_rows, 0] = cond_product
            matrix /= row_sums

            # CDF per group
            cdf = np.cumsum(matrix, axis=1)

            # Inverse-CDF sampling
            u = np.random.uniform(size=n)

            if cond_product <= 1:
                flat_indices = np.zeros(n, dtype=np.int64)
            elif n * cond_product <= 10_000_000:
                # Broadcast: build (n, cond_product), find first bin >= u
                row_cdfs = cdf[group_keys]  # (n, cond_product)
                flat_indices = (row_cdfs < u[:, np.newaxis]).sum(axis=1)
                np.minimum(flat_indices, cond_product - 1, out=flat_indices)
            else:
                # Large conditional: per-group searchsorted with pre-computed CDF
                flat_indices = np.empty(n, dtype=np.int64)
                for gk in np.unique(group_keys):
                    mask = group_keys == gk
                    flat_indices[mask] = np.searchsorted(cdf[gk], u[mask]).clip(
                        0, cond_product - 1
                    )

            per_dim = np.unravel_index(flat_indices, all_sample_shape)
            for j, dim_idx in enumerate(all_sample_dims):
                out_dims[dim_idx] = per_dim[j].astype(out_dims[dim_idx].dtype)
        else:
            # Gate masks present: use per-group loop to apply per-category masks.
            if gate_cats is not None:
                loop_keys = group_keys * gate_dom + gate_cats
            else:
                loop_keys = group_keys

            cond_dim_list = _cond_dims(all_sample_dims, sep_child_dims)

            for gk in np.unique(loop_keys):
                row_mask = loop_keys == gk
                group_n = int(row_mask.sum())

                if gate_cats is not None:
                    common_cat = int(gk % gate_dom)
                    sep_gk = int(gk // gate_dom)
                else:
                    sep_gk = int(gk)

                idx = [slice(None)] * len(child_p.shape)
                remaining = sep_gk
                for sd, dom in zip(sep_child_dims, sep_domains):
                    idx[sd] = remaining % dom
                    remaining //= dom

                conditional = child_p[tuple(idx)].copy()

                for d, mask in gate_masks.items():
                    j = all_sample_dims.index(d)
                    axis = cond_dim_list[j]
                    valid = mask[common_cat]
                    slicer = [slice(None)] * conditional.ndim
                    slicer[axis] = ~valid
                    conditional[tuple(slicer)] = 0

                cond_flat = conditional.ravel()
                cond_sum = cond_flat.sum()
                if cond_sum < 1e-12:
                    cond_flat = np.ones_like(cond_flat)
                    cond_sum = cond_flat.sum()
                cond_flat /= cond_sum

                group_indices = np.random.choice(
                    len(cond_flat), size=group_n, p=cond_flat
                )
                group_per_dim = np.unravel_index(
                    group_indices, conditional.shape
                )

                for j, dim_idx in enumerate(all_sample_dims):
                    out_dims[dim_idx][row_mask] = group_per_dim[cond_dim_list[j]]

        # Store sampled dims
        for dim_idx, arr in out_dims.items():
            dims[dim_idx] = arr

        clique_vals[cidx] = dims
        _extract_h0(cidx, meta, attrs, dims, out)

    return out
