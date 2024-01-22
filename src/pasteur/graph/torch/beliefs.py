from functools import reduce
from typing import NamedTuple, cast
import numpy as np
import torch
from git import Sequence

from ...attribute import DatasetAttributes
from ..beliefs import Message, convert_sel, numpy_gen_args, get_clique_shapes
from ..hugin import CliqueMeta, get_attrs


def torch_create_cliques(
    cliques: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
    device: "torch.device | None" = None,
):
    return [
        torch.zeros(shape, device=device, requires_grad=True)
        for shape in get_clique_shapes(cliques, attrs)
    ]


def torch_to_mapping_repr(
    a_map: tuple[int, ...],
    b_map: tuple[int, ...],
    a_dom: int,
    b_dom: int,
    device: "torch.device | None" = None,
) -> "torch.Tensor":
    uniques = np.unique(np.stack([a_map, b_map]), axis=1)
    out = torch.from_numpy(uniques.astype("int32"))
    out.requires_grad = False
    if device:
        out = out.to(device)
    return out


class BeliefPropagation(torch.nn.Module):
    def __init__(
        self, cliques: Sequence[CliqueMeta], messages: Sequence[Message]
    ) -> None:
        super().__init__()
        self.cliques = cliques
        self.messages = messages

        # Create tensors for index args
        self.doms = [[i.b_dom if i else None for i in m.index_args] for m in messages]
        self.idx = [(cliques.index(m.a), cliques.index(m.b)) for m in messages]
        index_args = []
        for m in messages:
            index_args.append(
                torch.nn.ParameterList(
                    [
                        torch.nn.Parameter(
                            torch_to_mapping_repr(*i), requires_grad=False
                        )
                        for i in m.index_args
                        if i is not None
                    ]
                )
            )
        self.index_args = torch.nn.ModuleList(index_args)

    def forward(self, theta: Sequence[torch.Tensor]):
        theta = list(theta)

        with torch.no_grad():
            done = {}
            for i, (m, (a_idx, b_idx)) in enumerate(zip(self.messages, self.idx)):
                # Start with clique
                proc = theta[a_idx]

                if m.sum_dims:
                    proc = torch.sum(proc, dim=m.sum_dims)

                # If backward pass, remove duplicate beliefs
                if not m.forward:
                    proc = proc - done[(m.b, m.a)]

                # Apply indexing ops
                # assert len(m.index_args) == len(proc.shape)
                l = 0
                for j, dom in enumerate(self.doms[i]):
                    if not dom:
                        continue

                    op = cast(torch.nn.ParameterList, self.index_args[i])[l]
                    l += 1

                    a_slice = [
                        op[0, :] if k == j else slice(None)
                        for k in range(len(proc.shape))
                    ]
                    proc = torch.zeros(
                        size=[dom if j == k else s for k, s in enumerate(proc.shape)],
                        device=proc.device,
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
                theta[b_idx] = theta[b_idx] + proc

        return theta


def torch_to_matrix_mul(
    a_map: tuple[int, ...],
    b_map: tuple[int, ...],
    a_dom: int,
    b_dom: int,
    device: "torch.device | None" = None,
):
    uniques = np.unique(np.stack([a_map, b_map]), axis=1)

    with torch.no_grad():
        idx = torch.from_numpy(uniques.astype("int64")).to(device)

        idx = b_dom * idx[0, :] + idx[1, :]
        out = torch.zeros((a_dom * b_dom), dtype=torch.float32, device=device)
        out.requires_grad = False
        out.index_fill_(0, idx, 1)
        out = out.reshape((a_dom, b_dom))
        return out

        # Einsum does not support sparse vectors
        # return torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (a_dom, b_dom))


class BeliefPropagationMul(torch.nn.Module):
    """Worse in every way than default."""

    def __init__(
        self, cliques: Sequence[CliqueMeta], messages: Sequence[Message]
    ) -> None:
        super().__init__()
        self.cliques = cliques
        self.messages = messages

        self.doms = [[i.b_dom if i else None for i in m.index_args] for m in messages]
        self.idx = [(cliques.index(m.a), cliques.index(m.b)) for m in messages]

        # Create tensors for mul args
        mmul_args = []
        for m in messages:
            mmul_args.append(
                torch.nn.ParameterList(
                    [
                        torch.nn.Parameter(torch_to_matrix_mul(*i), requires_grad=False)
                        for i in m.index_args
                        if i is not None
                    ]
                )
            )
        self.mmul_args = torch.nn.ModuleList(mmul_args)

    def forward(self, theta: Sequence[torch.Tensor], use_indexing: bool = False):
        theta = list(theta)

        with torch.no_grad():
            done = {}
            for i, (m, (a_idx, b_idx)) in enumerate(zip(self.messages, self.idx)):
                # Start with clique
                proc = theta[a_idx]

                if m.sum_dims:
                    proc = torch.sum(proc, dim=m.sum_dims)

                # If backward pass, remove duplicate beliefs
                if not m.forward:
                    proc = proc - done[(m.b, m.a)]

                # Apply mat mul with einsum
                l = 0
                ofs = len(self.doms[i])
                args = [proc, list(range(ofs))]
                out_dims = []
                for j, dom in enumerate(self.doms[i]):
                    if dom:
                        op = cast(torch.nn.ParameterList, self.mmul_args[i])[l]
                        args.append(op)
                        args.append([j, ofs + l])
                        out_dims.append(ofs + l)
                        l += 1
                    else:
                        out_dims.append(j)

                if l > 0:
                    proc = torch.einsum(*args, out_dims)

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
                theta[b_idx] = theta[b_idx] + proc

        return theta


class TorchIndexArgs(NamedTuple):
    transpose: tuple[int, ...]
    transpose_undo: tuple[int, ...]
    b_doms: tuple[int, ...]
    idx_a: int | None
    idx_b: int | None


def torch_gen_args(
    messages: Sequence[Message],
):
    args = []
    idx_a = []
    idx_b = []
    for a in numpy_gen_args(messages):
        if a:
            # TODO: Verify this function works
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
                TorchIndexArgs(
                    a.transpose, a.transpose_undo, a.b_doms, idx_a_idx, idx_b_idx
                )
            )
        else:
            args.append(None)

    return args, torch.nn.ParameterList(idx_a), torch.nn.ParameterList(idx_b)


class BeliefPropagationSingle(torch.nn.Module):
    def __init__(
        self, cliques: Sequence[CliqueMeta], messages: Sequence[Message]
    ) -> None:
        super().__init__()
        self.cliques = cliques
        self.messages = messages
        self.idx = [(cliques.index(m.a), cliques.index(m.b)) for m in messages]
        self.args, self.idx_a, self.idx_b = torch_gen_args(messages)

    def forward(self, theta: Sequence[torch.Tensor]):
        with torch.no_grad():
            done = {}
            for i, (m, (a_idx, b_idx)) in enumerate(zip(self.messages, self.idx)):
                # Start with clique
                proc = theta[a_idx]

                if m.sum_dims:
                    proc = torch.sum(proc, dim=m.sum_dims)

                # If backward pass, remove duplicate beliefs
                if not m.forward:
                    proc = proc - done[(m.b, m.a)]

                arg = self.args[i]
                if arg:
                    proc = proc.permute(arg.transpose)
                    og_shape = proc.shape

                    # Find domains of front dimensions and back dimension
                    # the front dimension changes during indexing, the back stays the same.
                    a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(arg.b_doms)])
                    b_idx_dom = reduce(lambda a, b: a * b, arg.b_doms)
                    rest_dom = reduce(
                        lambda a, b: a * b, og_shape[len(arg.b_doms) :], 1
                    )

                    # Perform index add on new tensor
                    # because index_add is expensive, only do it when it's required.
                    proc = proc.reshape((a_idx_dom, -1))
                    if arg.idx_a is not None:
                        proc = torch.index_select(proc, 0, self.idx_a[arg.idx_a])
                    if arg.idx_b is not None:
                        proc = proc.new_zeros((b_idx_dom, rest_dom)).index_add_(
                            0, self.idx_b[arg.idx_b], proc
                        )

                    # Reshape to individual columns and undo transpose
                    new_shape = list(arg.b_doms) + list(og_shape[len(arg.b_doms) :])
                    proc = proc.reshape(new_shape).permute(arg.transpose_undo)

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
                # print(a_idx, b_idx, proc.sum().cpu().numpy(), m.forward)
                theta[b_idx].add_(proc)

        return theta
