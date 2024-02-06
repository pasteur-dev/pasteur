from functools import reduce
from typing import NamedTuple, Sequence

import torch

from ...attribute import DatasetAttributes
from ..hugin import CliqueMeta
from ..loss import LinearObservation, ParentMeta, get_parent_meta, get_smallest_parent
from .beliefs import torch_to_mapping_repr


class TorchParentMeta(NamedTuple):
    sum_dims: tuple[int, ...]

    idx: int | None = None
    b_doms: tuple[int, ...] = tuple()
    transpose: tuple[int, ...] = tuple()
    transpose_undo: tuple[int, ...] = tuple()


def torch_get_parent_meta(
    obs: Sequence[LinearObservation],
    parents: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
) -> tuple[list[TorchParentMeta], torch.nn.ParameterList]:
    out = []
    idx = []
    for o, p in zip(obs, parents):
        meta = get_parent_meta(o.source, p, attrs)
        if meta.idx is not None:
            idx_idx = len(idx)
            idx.append(
                torch.nn.Parameter(
                    torch.from_numpy(meta.idx.astype("int64")), requires_grad=False
                )
            )
        else:
            idx_idx = None
        out.append(
            TorchParentMeta(
                meta.sum_dims,
                idx_idx,
                meta.b_doms,
                meta.transpose,
                meta.transpose_undo,
            )
        )

    return out, torch.nn.ParameterList(idx)


class LinearLoss(torch.nn.Module):
    def __init__(
        self,
        obs: Sequence[LinearObservation],
        cliques: Sequence[CliqueMeta],
        attrs: DatasetAttributes,
        loss_fun=torch.nn.MSELoss(),
    ) -> None:
        super().__init__()
        self.obs_meta = obs
        self.obs = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.from_numpy(o.obs).to(torch.float32), requires_grad=False
                )
                for o in obs
            ]
        )
        self.parents = [get_smallest_parent(o.source, cliques, attrs) for o in obs]

        self.cidx = [cliques.index(p) for p in self.parents]
        self.parent_meta, self.idx = torch_get_parent_meta(
            self.obs_meta, self.parents, attrs
        )

        self.loss_fun = loss_fun

    def forward(self, theta: Sequence[torch.Tensor]):
        losses = []

        for idx, obs, ometa, pmeta in zip(
            self.cidx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = theta[idx].exp()

            if pmeta.sum_dims:
                proc = torch.sum(proc, dim=pmeta.sum_dims)

            # Peform alignment if necessary
            if pmeta.idx is not None:
                proc = proc.permute(pmeta.transpose)
                og_shape = proc.shape

                # Find domains of front dimensions and back dimension
                # the front dimension changes during indexing, the back stays the same.
                a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(pmeta.b_doms)])
                b_idx_dom = reduce(lambda a, b: a * b, pmeta.b_doms)
                rest_dom = reduce(lambda a, b: a * b, og_shape[len(pmeta.b_doms) :], 1)

                # Perform index add on new tensor
                # because index_add is expensive, only do it when it's required.
                proc = proc.reshape((a_idx_dom, -1))
                proc = proc.new_zeros((b_idx_dom, rest_dom)).index_add_(
                    0, self.idx[pmeta.idx], proc
                )

                # Reshape to individual columns and undo transpose
                new_shape = list(pmeta.b_doms) + list(og_shape[len(pmeta.b_doms) :])
                proc = proc.reshape(new_shape).permute(pmeta.transpose_undo)

            # Apply loss function
            if ometa.mapping is not None:
                proc = ometa.mapping @ proc
            obs_loss = self.loss_fun(obs, proc)
            if ometa.confidence != 1:
                obs_loss *= ometa.confidence

            losses.append(obs_loss)

        return torch.mean(torch.stack(losses))
