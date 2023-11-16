from typing import (
    TYPE_CHECKING,
    Generic,
    NamedTuple,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np

from ..attribute import DatasetAttributes
from .hugin import CliqueMeta, get_attrs

if TYPE_CHECKING:
    import torch


A = TypeVar("A")
B = TypeVar("B", covariant=True)


class MappingReprFun(Protocol, Generic[B]):
    def __call__(
        self, a_map: np.ndarray, b_map: np.ndarray, a_dom: int, b_dom: int
    ) -> B:
        raise NotImplementedError()


class Message(Generic[A]):
    # This message is sent from clique a -> b
    # Cliques use an alphabetical canonical order for their attributes, so
    # common attributes of a, b are in the same order.
    a: CliqueMeta
    b: CliqueMeta

    # This message can be either part of the forward pass (a -> b before b -> a)
    # or part of the backward pass.
    # If it's part of the backward pass, the forward message  (b -> a)
    # which has been sent needs to be subtracted from it, so a version of the
    # forward message that is broadcastable has to be stored during the forward pass.
    forward: bool

    # The broadcastable version will contain a subset of dims, the others
    # should be summed out.
    sum_dims: tuple[int, ...]

    # The attribute domains between cliques are expected to be different.
    # In this case, we indexing operations to move between them.
    # After the summing operations, only the message attributes are left
    # In the order they appear in, index_args provides either an op or None for no-op.
    index_args: tuple[A | None, ...]

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
    repr_fun: MappingReprFun[A],
) -> Sequence[Message[A]]:
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

                        args.append(repr_fun(a_map, b_map, a_dom, b_dom))
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
    return messages


def torch_create_cliques(cliques: Sequence[CliqueMeta], attrs: DatasetAttributes):
    import torch

    theta = {}
    for cl in cliques:
        shape = []
        for meta in cl:
            shape.append(
                get_attrs(attrs, meta.table, meta.order)[meta.attr].get_domain(
                    convert_sel(meta.sel)
                )
            )
        theta[cl] = torch.zeros(shape)
    return theta


def torch_to_mapping_repr(
    a_map: np.ndarray, b_map: np.ndarray, a_dom: int, b_dom: int
) -> tuple["torch.Tensor", int]:
    import torch

    uniques = np.unique(np.stack([a_map, b_map]), axis=1)
    return torch.from_numpy(uniques.astype("int32")), b_dom


def torch_perform_pass(
    cliques: dict[CliqueMeta, "torch.Tensor"],
    messages: Sequence[Message[tuple["torch.Tensor", int]]],
):
    import torch

    cliques = dict(cliques)

    with torch.no_grad():
        done = {}
        for m in messages:
            # Start with clique
            proc = cliques[m.a]

            if m.sum_dims:
                proc = torch.sum(proc, dim=m.sum_dims)

            # If backward pass, remove duplicate beliefs
            if not m.forward:
                proc = proc - done[(m.b, m.a)]

            # Apply indexing ops
            assert len(m.index_args) == len(proc.shape)
            for j, op_dom in enumerate(m.index_args):
                if not op_dom:
                    continue
                op, new_dom = op_dom
                a_slice = [
                    op[0, :] if k == j else slice(None) for k in range(len(proc.shape))
                ]
                proc = torch.zeros(
                    size=[new_dom if j == k else s for k, s in enumerate(proc.shape)]
                ).index_add_(j, op[1, :], proc[a_slice])

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


def numpy_create_cliques(cliques: Sequence[CliqueMeta], attrs: DatasetAttributes):
    theta = {}
    for cl in cliques:
        shape = []
        for meta in cl:
            shape.append(
                get_attrs(attrs, meta.table, meta.order)[meta.attr].get_domain(
                    convert_sel(meta.sel)
                )
            )
        theta[cl] = np.zeros(shape)
    return theta


def numpy_to_mapping_repr(a_map: np.ndarray, b_map: np.ndarray, a_dom: int, b_dom: int):
    return np.unique(np.stack([a_map, b_map]), axis=1), b_dom


def numpy_perform_pass(
    cliques: dict[CliqueMeta, np.ndarray],
    messages: Sequence[Message[np.ndarray]],
):
    done = {}
    for m in messages:
        # Start with clique
        proc = cliques[m.a]

        if m.sum_dims:
            proc = np.sum(proc, axis=m.sum_dims)

        # If backward pass, remove duplicate beliefs
        if not m.forward:
            proc = proc - done[(m.b, m.a)]

        # Apply indexing ops
        assert len(m.index_args) == len(proc.shape)
        for j, op_dom in enumerate(m.index_args):
            if not op_dom:
                continue
            op, new_dom = op_dom
            a_slice = [
                op[0, :] if k == j else slice(None) for k in range(len(proc.shape))
            ]
            b_slice = [
                op[1, :] if k == j else slice(None) for k in range(len(proc.shape))
            ]
            tmp = np.zeros([new_dom if j == k else s for k, s in enumerate(proc.shape)])
            tmp[tuple(b_slice)] += proc[tuple(a_slice)]
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
