from functools import reduce
from typing import NamedTuple, Sequence

import torch

from ...attribute import DatasetAttributes
from ..hugin import CliqueMeta
from ..loss import LinearObservation, ParentMeta, get_parent_meta, get_smallest_parent
from .beliefs import torch_to_mapping_repr


class TorchParentMeta(NamedTuple):
    sum_dims: tuple[int, ...]

    idx_a: int | None = None
    b_doms: tuple[int, ...] = tuple()
    transpose: tuple[int, ...] = tuple()
    transpose_undo: tuple[int, ...] = tuple()


def torch_get_parent_meta(
    obs: Sequence[LinearObservation],
    parents: Sequence[CliqueMeta],
    attrs: DatasetAttributes,
) -> tuple[list[TorchParentMeta], torch.nn.ParameterList]:
    out = []
    idx_a = []
    for o, p in zip(obs, parents):
        meta = get_parent_meta(o.source, p, attrs)
        if meta.idx_a is not None:
            idx_idx_a = len(idx_a)
            idx_a.append(torch.from_numpy(meta.idx_a))
        else:
            idx_idx_a = None
        out.append(
            TorchParentMeta(
                meta.sum_dims,
                idx_idx_a,
                meta.b_doms,
                meta.transpose,
                meta.transpose_undo,
            )
        )

    return out, torch.nn.ParameterList(idx_a)


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
        self.obs = torch.nn.ParameterList([torch.from_numpy(o.obs) for o in obs])
        self.parents = [get_smallest_parent(o.source, cliques, attrs) for o in obs]

        self.idx = [cliques.index(p) for p in self.parents]
        self.parent_meta, self.idx_a = torch_get_parent_meta(
            self.obs_meta, self.parents, self.attrs
        )

        self.loss_fun = loss_fun

    def forward(self, theta: Sequence[torch.Tensor]):
        loss = torch.scalar_tensor(0)

        for idx, obs, ometa, pmeta in zip(
            self.idx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = theta[idx]

            if pmeta.sum_dims:
                proc = torch.sum(proc, dim=pmeta.sum_dims)

            # Peform alignment if necessary
            if pmeta.idx_a:
                proc = proc.permute(pmeta.transpose)
                og_shape = proc.shape

                # Find domains of front dimensions and back dimension
                # the front dimension changes during indexing, the back stays the same.
                a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(pmeta.b_doms)])

                # Perform index add on new tensor
                # because index_add is expensive, only do it when it's required.
                proc = proc.reshape((a_idx_dom, -1))
                proc = torch.index_select(proc, 0, self.idx_a[pmeta.idx_a])

                # Reshape to individual columns and undo transpose
                new_shape = list(pmeta.b_doms) + list(og_shape[len(pmeta.b_doms) :])
                proc = proc.reshape(new_shape).permute(pmeta.transpose_undo)

            # Apply loss function
            if ometa.mapping is not None:
                proc = ometa.mapping @ proc
            obs_loss = self.loss_fun(proc, obs)
            if ometa.confidence != 1:
                obs_loss *= ometa.confidence
            loss += obs_loss

        return loss
