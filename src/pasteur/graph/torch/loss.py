from functools import reduce
from typing import NamedTuple, Sequence

import numpy as np
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
        loss_fun: str = "mse",
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

        self.loss_type = loss_fun

    def _project_probs(self, mu: Sequence[torch.Tensor], idx: int, pmeta: TorchParentMeta):
        """Project probability-space clique marginals to an observation's shape."""
        proc = mu[idx]

        if pmeta.sum_dims:
            proc = torch.sum(proc, dim=pmeta.sum_dims)

        if pmeta.idx is not None:
            proc = proc.permute(pmeta.transpose)
            og_shape = proc.shape

            a_idx_dom = reduce(lambda a, b: a * b, og_shape[: len(pmeta.b_doms)])
            b_idx_dom = reduce(lambda a, b: a * b, pmeta.b_doms)
            rest_dom = reduce(lambda a, b: a * b, og_shape[len(pmeta.b_doms) :], 1)

            proc = proc.reshape((a_idx_dom, -1))
            proc = proc.new_zeros((b_idx_dom, rest_dom)).index_add_(
                0, self.idx[pmeta.idx], proc
            )

            new_shape = list(pmeta.b_doms) + list(og_shape[len(pmeta.b_doms) :])
            proc = proc.reshape(new_shape).permute(pmeta.transpose_undo)

        return proc

    def _project(self, theta: Sequence[torch.Tensor], idx: int, pmeta: TorchParentMeta):
        """Project log-space clique potentials to an observation's shape."""
        return self._project_probs([t.exp() for t in theta], idx, pmeta)

    def forward_probs(self, mu: Sequence[torch.Tensor]):
        """Compute loss on probability-space clique marginals.

        Gradients are w.r.t. mu (marginals), matching the mirror descent
        update rule from private-pgm: theta -= alpha * grad_mu(L)."""
        losses = []

        for idx, obs, ometa, pmeta in zip(
            self.cidx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = self._project_probs(mu, idx, pmeta)

            if ometa.mapping is not None:
                proc = ometa.mapping @ proc

            obs_loss = torch.nn.functional.mse_loss(obs, proc)

            if ometa.confidence != 1:
                obs_loss *= ometa.confidence

            losses.append(obs_loss)

        return torch.mean(torch.stack(losses))

    def forward(self, theta: Sequence[torch.Tensor]):
        """Compute loss on log-space potentials (legacy interface)."""
        losses = []

        for idx, obs, ometa, pmeta in zip(
            self.cidx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = self._project(theta, idx, pmeta)

            if ometa.mapping is not None:
                proc = ometa.mapping @ proc

            if self.loss_type == "kl":
                log_ratio = torch.log(obs + 1e-30) - torch.log(proc + 1e-30)
                obs_loss = torch.sum(obs * log_ratio) / obs.numel()
            elif self.loss_type == "mse":
                obs_loss = torch.nn.functional.mse_loss(obs, proc)
            else:
                obs_loss = self.loss_type(obs, proc)

            if ometa.confidence != 1:
                obs_loss *= ometa.confidence

            losses.append(obs_loss)

        return torch.mean(torch.stack(losses))

    def project_marginals(self, theta: Sequence[torch.Tensor]) -> list[np.ndarray]:
        """Project clique potentials back to per-observation marginals.

        Returns a list of numpy arrays (one per observation) in the same
        shape as the original observations, normalized to probabilities."""
        result = []
        with torch.no_grad():
            for idx, ometa, pmeta in zip(
                self.cidx, self.obs_meta, self.parent_meta
            ):
                proc = self._project(theta, idx, pmeta)
                if ometa.mapping is not None:
                    proc = ometa.mapping @ proc
                p = proc.cpu().numpy()
                p_sum = p.sum()
                if p_sum > 0:
                    p /= p_sum
                result.append(p)
        return result
