from functools import reduce
from typing import NamedTuple, Sequence

import torch

from ..attribute import DatasetAttributes
from .hugin import CliqueMeta
from .loss import LinearObservation, ParentMeta, get_parent_meta, get_smallest_parent


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


def _adjoint_project(
    grad_proj: torch.Tensor,
    mu_shape: torch.Size,
    pmeta: TorchParentMeta,
    idx_list: torch.nn.ParameterList,
) -> torch.Tensor:
    """Adjoint (transpose) of ``LinearLoss._project_probs``.

    Given dL/d(projected), returns dL/d(mu[idx]).
    """
    grad = grad_proj

    if pmeta.idx is not None:
        # Compute og_shape: shape after sum then permute
        if pmeta.sum_dims:
            summed_shape = tuple(
                s for i, s in enumerate(mu_shape) if i not in pmeta.sum_dims
            )
        else:
            summed_shape = tuple(mu_shape)
        og_shape = tuple(summed_shape[d] for d in pmeta.transpose)
        n_b = len(pmeta.b_doms)
        b_idx_dom = reduce(lambda a, b: a * b, pmeta.b_doms) if pmeta.b_doms else 1
        rest_dom = reduce(lambda a, b: a * b, og_shape[n_b:], 1)

        # Adjoint of permute(transpose_undo) + reshape(new_shape)
        grad = grad.permute(pmeta.transpose)
        grad = grad.reshape((b_idx_dom, rest_dom))

        # Adjoint of index_add_(0, idx, proc)
        grad = torch.index_select(grad, 0, idx_list[pmeta.idx])

        # Adjoint of reshape((a_idx_dom, -1)) + permute(transpose)
        grad = grad.reshape(og_shape)
        grad = grad.permute(pmeta.transpose_undo)

    if pmeta.sum_dims:
        # Adjoint of sum: unsqueeze each summed dim and expand
        for d in sorted(pmeta.sum_dims):
            grad = grad.unsqueeze(d)
        grad = grad.expand(mu_shape)

    return grad


class LinearLoss(torch.nn.Module):
    """Precomputed observation data + projection logic for mirror descent.

    Inherits nn.Module only for .to(device) on stored tensors."""

    def __init__(
        self,
        obs: Sequence[LinearObservation],
        cliques: Sequence[CliqueMeta],
        attrs: DatasetAttributes,
        loss_type: str = "l2",
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
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

    def forward(
        self,
        mu: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward loss + analytical backward, replacing autograd.

        Returns (loss, grads) where grads[i] = dL/d(mu[i]).
        """
        grads = [torch.zeros_like(m) for m in mu]
        loss = torch.tensor(0.0, device=mu[0].device)

        for idx, obs, ometa, pmeta in zip(
            self.cidx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = self._project_probs(mu, idx, pmeta)

            if ometa.mapping is not None:
                proc = ometa.mapping @ proc

            diff = obs - proc

            # Forward loss
            if self.loss_type == "l1":
                obs_loss = torch.sum(torch.abs(diff))
            elif self.loss_type == "l1l2":
                obs_loss = torch.sum(torch.abs(diff)) + 0.5 * torch.sum(diff * diff)
            else:
                obs_loss = 0.5 * torch.sum(diff * diff)
            obs_loss = obs_loss * ometa.confidence
            loss = loss + obs_loss

            # Analytical gradient: dL/d(proj)
            if self.loss_type == "l1":
                grad_proj = -diff.sign() * ometa.confidence
            elif self.loss_type == "l1l2":
                grad_proj = (-diff.sign() - diff) * ometa.confidence
            else:
                grad_proj = -diff * ometa.confidence

            # Reverse optional mapping
            if ometa.mapping is not None:
                grad_proj = ometa.mapping.T @ grad_proj

            # Adjoint of projection: dL/d(mu[idx])
            grad_mu = _adjoint_project(grad_proj, mu[idx].shape, pmeta, self.idx)
            grads[idx] = grads[idx] + grad_mu

        return loss, grads
