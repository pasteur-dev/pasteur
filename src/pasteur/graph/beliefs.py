from functools import reduce
from typing import (
    NamedTuple,
    Sequence,
)

import numpy as np
import torch

from ..attribute import DatasetAttributes, get_dtype
from .hugin import CliqueMeta, get_attrs


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
            dom = attr.get_domain(convert_sel(meta.sel))
            weight.append(np.ones(dom) / dom)
        weights.append(weight)
    return weights


# --- Torch belief propagation ---


class _NumpyIndexArgs(NamedTuple):
    transpose: tuple[int, ...]
    transpose_undo: tuple[int, ...]

    idx_a: np.ndarray
    idx_b: np.ndarray
    b_doms: tuple[int, ...]


def _numpy_gen_args(messages: Sequence[Message]):
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
                uniques = deduplicate(arg.a_map, arg.b_map)
                idx_a_dims.append(uniques[0, :])
                idx_a_doms.append(arg.a_dom)
                idx_b_dims.append(uniques[1, :])
                idx_b_doms.append(arg.b_dom)

                transpose_front.append(i)
            else:
                transpose_back.append(i)

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
            _NumpyIndexArgs(transpose, transpose_undo, idx_a, idx_b, b_doms)
        )

    return tuple(index_args)


class _TorchIndexArgs(NamedTuple):
    transpose: tuple[int, ...]
    transpose_undo: tuple[int, ...]
    b_doms: tuple[int, ...]
    idx_a: int | None
    idx_b: int | None


def _torch_gen_args(messages: Sequence[Message]):
    args = []
    idx_a = []
    idx_b = []
    for a in _numpy_gen_args(messages):
        if a:
            idx_a_idx = idx_b_idx = None
            if a.idx_a[-1] != len(a.idx_a) - 1:
                idx_a_idx = len(idx_a)
                idx_a.append(
                    torch.nn.Parameter(
                        torch.from_numpy(a.idx_a.astype("int64")), requires_grad=False
                    )
                )
            if a.idx_b[-1] != len(a.idx_b) - 1:
                idx_b_idx = len(idx_b)
                idx_b.append(
                    torch.nn.Parameter(
                        torch.from_numpy(a.idx_b.astype("int64")), requires_grad=False
                    )
                )

            args.append(
                _TorchIndexArgs(
                    a.transpose, a.transpose_undo, a.b_doms, idx_a_idx, idx_b_idx
                )
            )
        else:
            args.append(None)

    return args, torch.nn.ParameterList(idx_a), torch.nn.ParameterList(idx_b)


def create_cliques(
    cliques: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
    device: "torch.device | None" = None,
):
    out = []
    for shape, weight in zip(
        get_clique_shapes(cliques, attrs), get_clique_weights(cliques, attrs)
    ):
        v = torch.scalar_tensor(1).to(device)
        for s, w in zip(shape, weight):
            v = torch.from_numpy(w).type(torch.float32).to(device) * v.unsqueeze(-1)
            assert v.shape[-1] == s
        out.append(v.log())
    return out


class BeliefPropagation(torch.nn.Module):
    def __init__(
        self,
        cliques: Sequence[CliqueMeta],
        messages: Sequence[Message],
        observed: set[int] | None = None,
        block_unobserved: bool = False,
    ) -> None:
        super().__init__()
        self.cliques = cliques
        self.messages = messages
        self.idx = [(cliques.index(m.a), cliques.index(m.b)) for m in messages]
        self.args, self.idx_a, self.idx_b = _torch_gen_args(messages)
        self.observed = observed
        self.block_unobserved = block_unobserved

    def forward(self, theta: Sequence[torch.Tensor], debug: bool = False):
        theta = list(theta)
        if debug:
            print("A: ", [float(t.logsumexp(list(range(len(t.shape))))) for t in theta])

        with torch.no_grad():
            # Shafer-Shenoy: for message a→b, build belief at a from
            # original potential + all incoming messages except from b,
            # then marginalize. Avoids the numerically unstable
            # subtraction of the Hugin algorithm.
            received: dict[int, list[tuple[torch.Tensor, int]]] = {}

            for i, (m, (a_idx, b_idx)) in enumerate(zip(self.messages, self.idx)):
                # For unobserved senders, skip their own potential entirely
                # so they act as pure pass-throughs for messages from
                # observed cliques, without injecting uninformed prior.
                _block = (
                    self.block_unobserved
                    and self.observed is not None
                    and a_idx not in self.observed
                )
                proc = torch.zeros_like(theta[a_idx]) if _block else theta[a_idx]
                if a_idx in received:
                    for msg, from_idx in received[a_idx]:
                        if from_idx != b_idx:
                            proc = proc + msg

                if m.sum_dims:
                    proc = torch.logsumexp(proc, dim=m.sum_dims)

                arg = self.args[i]
                if arg:
                    proc = proc.permute(arg.transpose)
                    og_shape = proc.shape

                    a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(arg.b_doms)])
                    b_idx_dom = reduce(lambda a, b: a * b, arg.b_doms)
                    rest_dom = reduce(
                        lambda a, b: a * b, og_shape[len(arg.b_doms) :], 1
                    )

                    proc = proc.reshape((a_idx_dom, -1))
                    if arg.idx_a is not None:
                        proc = torch.index_select(proc, 0, self.idx_a[arg.idx_a])
                    if arg.idx_b is not None:
                        # Stable scatter logsumexp: subtract per-group max
                        # before exp to avoid float32 underflow, then add back.
                        idx_b = self.idx_b[arg.idx_b]
                        idx_exp = idx_b.unsqueeze(1).expand_as(proc)
                        max_vals = proc.new_full(
                            (b_idx_dom, rest_dom), float("-inf")
                        )
                        max_vals.scatter_reduce_(
                            0, idx_exp, proc, reduce="amax", include_self=True
                        )
                        shifted = proc - max_vals.gather(0, idx_exp)
                        shifted = shifted.nan_to_num(0.0)  # -inf - (-inf) -> 0
                        result = proc.new_zeros((b_idx_dom, rest_dom))
                        result.index_add_(0, idx_b, shifted.exp())
                        proc = result.log() + max_vals

                    new_shape = list(arg.b_doms) + list(og_shape[len(arg.b_doms) :])
                    proc = proc.reshape(new_shape).permute(arg.transpose_undo)

                # Broadcast reshape to target clique shape
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

                if debug:
                    print(
                        a_idx,
                        b_idx,
                        proc.logsumexp(list(range(len(proc.shape)))).cpu().numpy(),
                        m.forward,
                    )

                if b_idx not in received:
                    received[b_idx] = []
                received[b_idx].append((proc, a_idx))

            # Compute final beliefs: original potential + all received messages
            theta = list(theta)
            for idx in range(len(theta)):
                if idx in received:
                    for msg, _ in received[idx]:
                        theta[idx] = theta[idx] + msg

            # Normalize resulting cliques
            if debug:
                print(
                    "Z: ",
                    [float(t.logsumexp(list(range(len(t.shape))))) for t in theta],
                )
            for i in range(len(theta)):
                Z = theta[i].logsumexp(list(range(len(theta[i].shape))))
                theta[i] = theta[i] - Z

        return theta
