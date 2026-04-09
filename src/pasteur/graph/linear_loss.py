from functools import reduce
from typing import NamedTuple, Sequence

import numpy as np
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


class LinearLoss(torch.nn.Module):
    def __init__(
        self,
        obs: Sequence[LinearObservation],
        cliques: Sequence[CliqueMeta],
        attrs: DatasetAttributes,
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

    def forward(self, mu: Sequence[torch.Tensor]):
        """Compute loss on probability-space clique marginals.

        Uses 0.5 * sum(diff²) as base loss (matching private-pgm).
        Confidence from LinearObservation handles per-observation weighting."""
        loss = torch.tensor(0.0, device=mu[0].device)

        for idx, obs, ometa, pmeta in zip(
            self.cidx, self.obs, self.obs_meta, self.parent_meta
        ):
            proc = self._project_probs(mu, idx, pmeta)

            if ometa.mapping is not None:
                proc = ometa.mapping @ proc

            diff = obs - proc
            obs_loss = 0.5 * torch.sum(diff * diff)
            obs_loss *= ometa.confidence

            loss = loss + obs_loss

        return loss

    def per_obs_loss(self, mu: Sequence[torch.Tensor]) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
        """Return per-observation (loss, max_dev, projected, target) for diagnostics."""
        device = next(self.parameters()).device
        mu = [m.to(device) for m in mu]
        result = []
        with torch.no_grad():
            for idx, obs, ometa, pmeta in zip(
                self.cidx, self.obs, self.obs_meta, self.parent_meta
            ):
                proc = self._project_probs(mu, idx, pmeta)
                if ometa.mapping is not None:
                    proc = ometa.mapping @ proc
                diff = obs - proc
                obs_loss = 0.5 * torch.sum(diff * diff) * ometa.confidence
                proj_np = proc.cpu().numpy()
                tgt_np = obs.cpu().numpy()
                max_dev = float(diff.abs().max())
                result.append((float(obs_loss), max_dev, proj_np, tgt_np))
        return result

    def project_marginals(self, theta: Sequence[torch.Tensor]) -> list[np.ndarray]:
        """Project clique potentials back to per-observation marginals.

        Returns a list of numpy arrays (one per observation) in the same
        shape as the original observations, normalized to probabilities."""
        result = []
        with torch.no_grad():
            for idx, ometa, pmeta in zip(
                self.cidx, self.obs_meta, self.parent_meta
            ):
                mu = [t.exp() for t in theta]
                proc = self._project_probs(mu, idx, pmeta)
                if ometa.mapping is not None:
                    proc = ometa.mapping @ proc
                p = proc.cpu().numpy()
                p_sum = p.sum()
                if p_sum > 0:
                    p /= p_sum
                result.append(p)
        return result
