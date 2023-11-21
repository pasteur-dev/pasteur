from typing import cast
import numpy as np
import torch
from git import Sequence

from ..attribute import DatasetAttributes
from .beliefs import Message, convert_sel
from .hugin import CliqueMeta, get_attrs


def torch_create_cliques(
    cliques: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
    device: "torch.device | None" = None,
):
    theta = {}
    for cl in cliques:
        shape = []
        for meta in cl:
            shape.append(
                get_attrs(attrs, meta.table, meta.order)[meta.attr].get_domain(
                    convert_sel(meta.sel)
                )
            )
        theta[cl] = torch.zeros(shape, device=device)
    return list(theta.values())


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

