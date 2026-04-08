import logging
from typing import Sequence

import numpy as np
import torch

from ...utils.progress import piter, IS_AGENT

from ...attribute import DatasetAttributes
from ..hugin import (
    CliqueMeta,
)
from ..loss import LinearObservation
from .beliefs import BeliefPropagationSingle, torch_create_cliques
from .loss import LinearLoss

logger = logging.getLogger(__name__)


def mirror_descent(
    cliques: Sequence[CliqueMeta],
    messages: Sequence,
    obs: Sequence[LinearObservation],
    attrs: DatasetAttributes,
    *,
    lr: float = 0.07,
    max_iters: int = 10_000,
    atol: float = 1e-5,
    patience: int = 50,
    checkpoint_every: int = 50,
    device: torch.device | str | None = None,
    compile: bool = False,
) -> list[np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Build modules
    bp = BeliefPropagationSingle(cliques, messages).to(device)
    loss_fn = LinearLoss(obs, cliques, attrs).to(device)

    # Initialize potentials (uniform weighted prior in log-space)
    theta = torch_create_cliques(cliques, attrs, device=device)
    theta = [t.requires_grad_(True) for t in theta]

    optimizer = torch.optim.Adam(theta, lr=lr)

    def compute_grad(theta, bp, loss_fn):
        # 1. Forward: BP to get consistent log-potentials, then exponentiate
        with torch.no_grad():
            theta_bp = bp(list(theta))
            mu = [t.exp() for t in theta_bp]

        # 2. Compute loss and gradient w.r.t. probability-space marginals
        mu = [m.requires_grad_(True) for m in mu]
        loss = loss_fn.forward_probs(mu)
        loss.backward()

        # 3. Apply marginal-space gradient directly to potentials
        #    (this is the correct mirror descent update: theta -= lr * ∂L/∂mu)
        for t, m in zip(theta, mu):
            if t.grad is None:
                t.grad = m.grad
            elif m.grad is not None:
                t.grad.copy_(m.grad)
        return loss

    if compile:
        logger.info("Compiling mirror descent compute graph...")
        compute_grad = torch.compile(compute_grad)

    logger.info(
        f"Mirror descent: {len(cliques)} cliques, {len(obs)} observations, "
        f"lr={lr}, device={device}, compile={compile}"
    )

    best_loss = float("inf")
    stale = 0
    total_iters = 0
    converged = False
    pbar = piter(range(max_iters), total=max_iters, desc="Mirror descent")

    while total_iters < max_iters:
        block_size = min(checkpoint_every, max_iters - total_iters)
        losses = []
        for _ in range(block_size):
            loss = compute_grad(theta, bp, loss_fn)
            losses.append(loss)
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
                for t in theta:
                    Z = t.logsumexp(list(range(len(t.shape))))
                    t -= Z
        total_iters += block_size
        pbar.update(block_size)

        # Sync GPU and check convergence
        loss_vals = [l.item() for l in losses]
        for cur_loss in loss_vals:
            if cur_loss < best_loss:
                best_loss = cur_loss
            if cur_loss <= atol or cur_loss > best_loss:
                stale += 1
            else:
                stale = 0

        desc = (
            f"Mirror descent: loss={loss_vals[-1]:.2e}, best={best_loss:.2e}, "
            f"stale={stale}/{patience}"
        )
        pbar.set_description(desc)
        if IS_AGENT:
            logger.info(desc)

        if stale >= patience:
            converged = True
            break

    pbar.close()
    if converged:
        logger.info(f"Mirror descent converged at iter {total_iters}.")
    else:
        logger.warning(
            f"Mirror descent did not converge after {max_iters} iterations "
            f"(best loss={best_loss:.6e}, atol={atol})."
        )

    # Final BP pass to get consistent clique potentials
    with torch.no_grad():
        theta_bp = bp(list(theta))
        result = []
        for t in theta_bp:
            p = t.exp().cpu().numpy()
            p /= p.sum()
            result.append(p)

    return result, loss_fn
