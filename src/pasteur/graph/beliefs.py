from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NamedTuple,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np

from ..attribute import DatasetAttributes, get_dtype
from .hugin import CliqueMeta, get_attrs


A = TypeVar("A")
B = TypeVar("B", covariant=True)

def deduplicate(arr1, arr2):
    check = set()
    out = []
    for t in zip(arr1, arr2):
        if t not in check:
            out.append(t)
            check.add(t)
    return np.stack(out).T

class IndexArg(NamedTuple):
    a_map: tuple[int, ...]
    b_map: tuple[int, ...]
    a_dom: int
    b_dom: int


class Message:
    # This message is sent from clique a -> b
    # Cliques use an alphabetical canonical order for their attributes, so
    # common attributes of a, b are in the same order.
    a: CliqueMeta
    b: CliqueMeta

    # This message can be either part of the forward pass (a -> b before b -> a)
    # or part of the backward pass.
    # If it's part of the backward pass, the forward message (b -> a)
    # which has been sent needs to be subtracted from it, so a version of the
    # forward message that is broadcastable has to be stored during the forward pass.
    forward: bool

    # The broadcastable version will contain a subset of dims, the others
    # should be summed out.
    sum_dims: tuple[int, ...]

    # The attribute domains between cliques are expected to be different.
    # In this case, we use indexing operations to move between them.
    # After the summing operations, only the message attributes are left
    # In the order they appear in, index_args provides either an op or None for no-op.
    index_args: tuple[IndexArg | None, ...]

    # In the end, we have to add singleton dims so that the resulting message
    # is broadcastable to clique b.
    # Reshape dims specifies where singleton dimensions should be added.
    # False: add singleton dim with len 1, True: add original dimension
    # `len(reshape_dims) == len(b.shape)`, `sum(reshape_dims) == len(msg)`
    reshape_dims: tuple[bool, ...]

    def __init__(self, a, b, forward, sum_dims, index_args, reshape_dims):
        self.a = a
        self.b = b
        self.forward = forward
        self.sum_dims = sum_dims
        self.index_args = index_args
        self.reshape_dims = reshape_dims


def is_same(c, d):
    return c.table == d.table and c.order == d.order and c.attr == d.attr


def has_attr(cl, a):
    for attr in cl:
        if is_same(attr, a):
            return True
    return False


def convert_sel(sel):
    if isinstance(sel, int):
        return sel
    else:
        return dict(sel)


def create_messages(
    generations: Sequence[Sequence[tuple[CliqueMeta, CliqueMeta]]],
    attrs: DatasetAttributes,
) -> Sequence[Message]:
    done = set()
    messages = []
    for generation in generations:
        for a, b in generation:
            sum_dims = []
            args = []

            for i in range(len(a)):
                if has_attr(b, a[i]):
                    j = 0
                    while not is_same(a[i], b[j]):
                        j += 1

                    if a[i].sel != b[j].sel:
                        attr = get_attrs(attrs, a[i].table, a[i].order)[a[i].attr]

                        a_map = attr.get_mapping(convert_sel(a[i].sel))
                        a_dom = attr.get_domain(convert_sel(a[i].sel))
                        b_map = attr.get_mapping(convert_sel(b[j].sel))
                        b_dom = attr.get_domain(convert_sel(b[j].sel))
                        assert len(a_map) == len(b_map)
                        assert np.max(a_map) < a_dom and np.max(b_map) < b_dom

                        args.append(IndexArg(tuple(a_map), tuple(b_map), a_dom, b_dom))
                    else:
                        args.append(None)
                else:
                    sum_dims.append(i)

            unsqueeze_dims = []
            for i in range(len(b)):
                if has_attr(a, b[i]):
                    unsqueeze_dims.append(True)
                else:
                    unsqueeze_dims.append(False)

            msg = Message(
                a,
                b,
                (b, a) not in done,
                tuple(sum_dims),
                tuple(args),
                tuple(unsqueeze_dims),
            )
            messages.append(msg)
            done.add((a, b))
    return messages


def get_clique_shapes(cliques: Sequence[CliqueMeta], attrs: DatasetAttributes):
    shapes = []
    for cl in cliques:
        shape = []
        for meta in cl:
            shape.append(
                get_attrs(attrs, meta.table, meta.order)[meta.attr].get_domain(
                    convert_sel(meta.sel)
                )
            )
        shapes.append(shape)
    return shapes


def get_clique_weights(cliques: Sequence[CliqueMeta], attrs: DatasetAttributes):
    weights = []
    for cl in cliques:
        weight = []
        for meta in cl:
            attr = get_attrs(attrs, meta.table, meta.order)[meta.attr]
            mapping = attr.get_mapping(convert_sel(meta.sel))
            dom = attr.get_domain(convert_sel(meta.sel))
            
            w = np.zeros(dom)
            np.add.at(w, mapping, 1)
            w /= len(mapping)
            weight.append(w)
        weights.append(weight)
    return weights


def numpy_create_cliques(cliques: Sequence[CliqueMeta], attrs: DatasetAttributes):
    return [np.zeros(shape) for shape in get_clique_shapes(cliques, attrs)]


def numpy_gen_multi_index(messages: Sequence[Message]):
    index_args = []
    for m in messages:
        index_args.append(
            tuple(
                # np.unique(np.stack([idx.a_map, idx.b_map]), axis=1)
                deduplicate(idx.a_map, idx.b_map)
                if idx is not None
                else None
                for idx in m.index_args
            )
        )
    return tuple(index_args)


def numpy_perform_pass_multi_index(
    cliques: dict[CliqueMeta, np.ndarray],
    messages: Sequence[Message],
    index_args: tuple[tuple[np.ndarray | None, ...], ...],
):
    done = {}
    for i, m in enumerate(messages):
        # Start with clique
        proc = cliques[m.a]

        if m.sum_dims:
            proc = np.sum(proc, axis=m.sum_dims)

        # If backward pass, remove duplicate beliefs
        if not m.forward:
            proc = proc - done[(m.b, m.a)]

        # Apply indexing ops
        assert len(m.index_args) == len(proc.shape)
        for j, ia in enumerate(m.index_args):
            if not ia:
                continue
            op = index_args[i][j]
            assert op is not None

            new_dom = ia.b_dom
            a_slice = [
                op[0, :] if k == j else slice(None) for k in range(len(proc.shape))
            ]
            b_slice = [
                op[1, :] if k == j else slice(None) for k in range(len(proc.shape))
            ]
            tmp = np.zeros([new_dom if j == k else s for k, s in enumerate(proc.shape)])
            tmp2 = proc[tuple(a_slice)]
            np.add.at(tmp, tuple(b_slice), tmp2)  # type: ignore
            proc = tmp

        # Keep version for backprop
        if m.forward:
            done[(m.a, m.b)] = proc

        # Expand dims
        shape = proc.shape
        new_shape = []
        ofs = 0
        for is_common in m.reshape_dims:
            if is_common:
                new_shape.append(shape[ofs])
                ofs += 1
            else:
                new_shape.append(1)
        proc = proc.reshape(new_shape)

        # Apply to clique
        cliques[m.b] = cliques[m.b] + proc

    return cliques


class NumpyIndexArgs(NamedTuple):
    transpose: tuple[int, ...]
    transpose_undo: tuple[int, ...]

    idx_a: np.ndarray
    idx_b: np.ndarray
    b_doms: tuple[int, ...]

def numpy_gen_args(messages: Sequence[Message]):
    index_args = []
    for m in messages:
        transpose_front = []
        transpose_back = []

        idx_a_dims = []
        idx_a_doms = []
        idx_b_dims = []
        idx_b_doms = []

        for i, arg in enumerate(m.index_args):
            if arg:
                # uniques = np.unique(np.stack([arg.a_map, arg.b_map]), axis=1)
                uniques = deduplicate(arg.a_map, arg.b_map)
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
            index_args.append(None)
            continue

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

        b_doms = tuple(idx_b_doms)
        transpose = tuple(transpose_front + transpose_back)
        transpose_undo = tuple(transpose.index(i) for i in range(len(transpose)))
        index_args.append(
            NumpyIndexArgs(transpose, transpose_undo, idx_a, idx_b, b_doms)
        )

    return tuple(index_args)


def numpy_perform_pass(
    cliques: dict[CliqueMeta, np.ndarray],
    messages: Sequence[Message],
    args: tuple[NumpyIndexArgs | None, ...],
):
    done = {}
    for i, m in enumerate(messages):
        # Start with clique
        proc: np.ndarray = cliques[m.a]

        if m.sum_dims:
            proc = np.sum(proc, axis=m.sum_dims)

        # If backward pass, remove duplicate beliefs
        if not m.forward:
            proc = proc - done[(m.b, m.a)]

        # Apply single indexing op
        # Reshape common attributes in cliques a, b that have different heights
        # by bringing them in front, reshaping the message to have one front dim
        # and one back dim, and then reindexing the front dim into a new tensor.
        # By combining all indexing ops into one big op, memory accesses are minimized.
        # By bringing the indexing dims to the front, spatial locality is maximized.
        arg = args[i]
        if arg:
            proc = proc.transpose(arg.transpose)
            og_shape = proc.shape

            # Find domains of front dimensions and back dimension
            # the front dimension changes during indexing, the back stays the same.
            a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(arg.b_doms)])
            b_idx_dom = reduce(lambda a, b: a * b, arg.b_doms)
            rest_dom = reduce(lambda a, b: a * b, og_shape[len(arg.b_doms) :], 1)

            # Perform index add on new tensor
            tmp = np.zeros((b_idx_dom, rest_dom), dtype="float32")
            tmp2 = proc.reshape((a_idx_dom, -1))[arg.idx_a]
            # np.add.at(tmp, arg.idx_b, tmp2)
            tmp[arg.idx_b] += tmp2

            # Reshape to individual columns and undo transpose
            proc = tmp.reshape(
                list(arg.b_doms) + list(og_shape[len(arg.b_doms) :])
            ).transpose(arg.transpose_undo)

        # Keep version for backprop
        if m.forward:
            done[(m.a, m.b)] = proc

        # Expand dims
        shape = proc.shape
        new_shape = []
        ofs = 0
        for is_common in m.reshape_dims:
            if is_common:
                new_shape.append(shape[ofs])
                ofs += 1
            else:
                new_shape.append(1)
        proc = proc.reshape(new_shape)

        # Apply to clique
        cliques[m.b] = cliques[m.b] + proc

    return cliques
