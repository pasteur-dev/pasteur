from functools import reduce
from typing import Callable, NamedTuple, Sequence

import numpy as np

from ..attribute import DatasetAttributes, get_dtype
from .beliefs import IndexArg, convert_sel, has_attr, is_same
from .hugin import CliqueMeta, get_attrs, get_clique_domain


def sel_is_subset(source_sel, clique_sel):
    if isinstance(source_sel, int):
        # If source sel is an int, if clique sel is not it eclipses it,
        # if it is we compare heights
        if isinstance(clique_sel, int):
            return source_sel >= clique_sel
        else:
            return True

    # Source sel is not an int, if clique sel is, that means it does not
    # eclipse source sel
    if isinstance(clique_sel, int):
        return False

    for sn, sh in source_sel:
        found = False
        for cn, ch in clique_sel:
            if cn == sn:
                if sh < ch:
                    return False
                found = True
        if not found:
            return False
    return True


def clique_is_subset(source, clique):
    for c in source:
        found = False
        for s in clique:
            if c.table == s.table and c.order == s.order and c.attr == s.attr:
                found = True
                if not sel_is_subset(c.sel, s.sel):
                    return False
        if not found:
            return False

    return True


def get_parents(source: CliqueMeta, cliques: Sequence[CliqueMeta]):
    return [c for c in cliques if clique_is_subset(source, c)]


def get_smallest_parent(
    source: CliqueMeta, cliques: Sequence[CliqueMeta], attrs: DatasetAttributes
):
    p = get_parents(source, cliques)
    assert p, f"No parents found for clique: {source}. Should be impossible."
    return min(p, key=lambda x: get_clique_domain(x, attrs))


class LinearObservation(NamedTuple):
    """Observations against which the loss is computed.

    Mapping can be any differentiable function, but in this case we limit it to
    a matrix multiplication or none.

    All cliques that superset completely the `source` clique will be adjusted to
    it, and matrix multiplied to `mapping` to produce the expected `obs'`.
    Then, the selected loss function will be applied to them.

    `confidence` is a normalized value that signifies how trustworthy this
    observation is from [0, 1] and is used to scale the loss produced by the
    observation. It should be drawn from the expected SNR of the observation,
    given the expected sampling noise and the noise added by differential privacy.

    The `mapping` size is very large (O(N^2) of the clique), so it should be avoided
    if possible. If set to None, a matrix multiplication is not performed.
    """

    source: CliqueMeta
    mapping: np.ndarray | None
    obs: np.ndarray
    confidence: float


class ParentMeta(NamedTuple):
    """Contains the necessary metadata to align a parent clique to an observation
    clique.

    First, attributes that are not present in an observation clique are summed
    out. Then the common dimensions that are different are transposed to the
    front and indexed. Finally, the transpose is undone and the cliques are
    aligned."""

    sum_dims: tuple[int, ...]

    idx: np.ndarray | None = None
    b_doms: tuple[int, ...] = tuple()
    transpose: tuple[int, ...] = tuple()
    transpose_undo: tuple[int, ...] = tuple()


def get_parent_meta(
    source: CliqueMeta, parent: CliqueMeta, attrs: DatasetAttributes
) -> ParentMeta:
    # Use the parent with the smallest domain as that should be faster
    # computationally (?)
    sum_dims = []
    args: list[IndexArg | None] = []

    for i in range(len(parent)):
        if has_attr(source, parent[i]):
            j = 0
            while not is_same(parent[i], source[j]):
                j += 1

            if parent[i].sel != source[j].sel:
                attr = get_attrs(attrs, parent[i].table, parent[i].order)[
                    parent[i].attr
                ]

                a_map = attr.get_mapping(convert_sel(parent[i].sel))
                a_dom = attr.get_domain(convert_sel(parent[i].sel))
                b_map = attr.get_mapping(convert_sel(source[j].sel))
                b_dom = attr.get_domain(convert_sel(source[j].sel))
                assert len(a_map) == len(b_map)
                assert np.max(a_map) < a_dom and np.max(b_map) < b_dom

                args.append(IndexArg(tuple(a_map), tuple(b_map), a_dom, b_dom))
            else:
                args.append(None)
        else:
            sum_dims.append(i)

    transpose_front = []
    transpose_back = []

    idx_a_dims = []
    idx_a_doms = []
    idx_b_dims = []
    idx_b_doms = []

    for i, arg in enumerate(args):
        if arg:
            # FIXME: Unique does nothing
            uniques = np.unique(np.stack([arg.a_map, arg.b_map]), axis=1)
            idx_a_dims.append(uniques[0, :])
            idx_a_doms.append(arg.a_dom)
            idx_b_dims.append(uniques[1, :])
            idx_b_doms.append(arg.b_dom)

            transpose_front.append(i)
        else:
            transpose_back.append(i)

    # When reshaping is not needed, the `if arg` block
    # of the loop won't run so arrays will be empty. SKip.
    if not idx_a_dims:
        return ParentMeta(tuple(sum_dims), None)

    dtype_a = get_dtype(reduce(lambda a, b: a * b, idx_a_doms))
    dtype_b = get_dtype(reduce(lambda a, b: a * b, idx_b_doms))
    mesh_a = np.meshgrid(*idx_a_dims, indexing="ij")
    mesh_b = np.meshgrid(*idx_b_dims, indexing="ij")
    idx_a = np.zeros_like(mesh_a[0], dtype=dtype_a)
    idx_b = np.zeros_like(mesh_a[0], dtype=dtype_b)
    a_dom = 1
    b_dom = 1
    for i in reversed(list(range(len(idx_a_doms)))):
        idx_a += a_dom * mesh_a[i].astype(dtype_a)
        idx_b += b_dom * mesh_b[i].astype(dtype_b)
        a_dom *= idx_a_doms[i]
        b_dom *= idx_b_doms[i]

    idx_a = idx_a.reshape(-1)
    idx_b = idx_b.reshape(-1)

    # Only a single index op is performed to align parent to source
    # where each unique value in idx_a is mapped to idx_b
    _, uidx = np.unique(idx_a, return_index=True)
    idx = idx_b[uidx]

    b_doms = tuple(idx_b_doms)
    transpose = tuple(transpose_front + transpose_back)
    transpose_undo = tuple(transpose.index(i) for i in range(len(transpose)))

    return ParentMeta(tuple(sum_dims), idx, b_doms, transpose, transpose_undo)
