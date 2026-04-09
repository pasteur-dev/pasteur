from functools import reduce
from typing import NamedTuple, cast
import numpy as np
import torch
from git import Sequence

from ...attribute import DatasetAttributes
from ..beliefs import (
    Message,
    convert_sel,
    numpy_gen_args,
    get_clique_shapes,
    get_clique_weights,
    deduplicate,
)
from ..hugin import CliqueMeta, get_attrs


def dom(a):
    return reduce(lambda a, b: a * b, a)


def torch_create_cliques(
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


def torch_to_mapping_repr(
    a_map: tuple[int, ...],
    b_map: tuple[int, ...],
    a_dom: int,
    b_dom: int,
    device: "torch.device | None" = None,
) -> "torch.Tensor":
    # uniques = np.unique(np.stack([a_map, b_map]), axis=1)
    uniques = deduplicate(a_map, b_map)
    out = torch.from_numpy(uniques.astype("int32"))
    out.requires_grad = False
    if device:
        out = out.to(device)
    return out


def torch_to_matrix_mul(
    a_map: tuple[int, ...],
    b_map: tuple[int, ...],
    a_dom: int,
    b_dom: int,
    device: "torch.device | None" = None,
):
    # uniques = np.unique(np.stack([a_map, b_map]), axis=1)
    uniques = deduplicate(a_map, b_map)

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
                # Build belief at a excluding message from b
                proc = theta[a_idx]
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
