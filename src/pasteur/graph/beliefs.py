from typing import NamedTuple, Sequence
import numpy as np

from ..attribute import DatasetAttributes
from .hugin import CliqueMeta, get_message_passing_order, get_attrs


class MappingMatrix(NamedTuple):
    """A mapping matrix is a sparse (M, N) matrix that maps an attribute into
    a different complexity representation. Implementation varies between
    pytorch and numpy, so implemented agnostically."""

    a_dom: int
    a_map: np.ndarray
    b_dom: int
    b_map: np.ndarray


class Message(NamedTuple):
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
    # should be summed out. If part of the forward pass, those dims will
    # be summed as an intermediary operation and the result will be stored.
    sum_dims: tuple[int, ...]

    # The attribute domains between cliques are expected to be different.
    # In this case, we can use matrix multiplications to move between them.
    # The function `einsum` can perform these operations, as well as the summing
    # of the non-common dimensions (if not done to store the forward message).
    es_in_dims: tuple[int, ...]
    es_args: tuple[tuple[tuple[int, ...], MappingMatrix], ...]
    es_out_dims: tuple[int, ...]

    # In the end, we have to add singleton dims so that the resulting message
    # is broadcastable to clique b.
    # Reshape dims specifies where singleton dimensions should be added.
    # False: add singleton dim with len 1, True: add original dimension
    # `len(reshape_dims) == len(b.shape)`, `sum(reshape_dims) == len(msg)`
    reshape_dims: tuple[bool, ...]


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
            in_dims = []
            out_dims = []
            args = []

            ofs = len(a)
            j = 0
            for i in range(len(a)):
                in_dims.append(i)

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
                        mapping = MappingMatrix(a_dom, a_map, b_dom, b_map)

                        args.append(([i, ofs], mapping))
                        out_dims.append(ofs)
                        ofs += 1
                    else:
                        out_dims.append(i)
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
                tuple(in_dims),
                tuple(args),
                tuple(out_dims),
                tuple(unsqueeze_dims),
            )
            messages.append(msg)
    return messages