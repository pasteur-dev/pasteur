"""Mirror descent optimization for clique potentials on a junction tree.

Given noisy marginal observations (e.g. from differential privacy), fits
clique potentials by iterating:
  1. Belief propagation (projection onto marginal polytope)
  2. Loss computation against observations
  3. Gradient update on log-potentials (entropic mirror descent)
"""

import logging
from typing import Sequence

import numpy as np
import torch

from ...utils.progress import piter

from ...attribute import DatasetAttributes
from ..beliefs import create_messages, get_clique_shapes
from ..hugin import (
    CliqueMeta,
    get_junction_tree,
    get_message_passing_order,
    to_moral,
    find_elim_order,
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
    """Run mirror descent to fit clique potentials to observations.

    Args:
        cliques: List of clique metadata from junction tree.
        messages: Compiled message schedule from create_messages.
        obs: Observed marginals as LinearObservation objects (should be
            normalized to probabilities).
        attrs: Dataset attribute metadata.
        lr: Learning rate for Adam optimizer.
        max_iters: Maximum number of iterations.
        atol: Absolute loss threshold for convergence.
        patience: Stop after this many consecutive iterations below atol
            or without improvement.
        checkpoint_every: How often to sync GPU and check convergence.
        device: Torch device. None for auto-detect.
        compile: Whether to use torch.compile on the gradient computation.

    Returns:
        List of numpy arrays (one per clique) in probability space,
        normalized so each sums to 1.
    """
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
        theta_bp = bp(list(theta))
        theta_bp = [t.detach().requires_grad_(True) for t in theta_bp]
        loss = loss_fn(theta_bp)
        loss.backward()
        for t, t_bp in zip(theta, theta_bp):
            if t.grad is None:
                t.grad = t_bp.grad
            elif t_bp.grad is not None:
                t.grad.copy_(t_bp.grad)
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

        pbar.set_description(
            f"Mirror descent: loss={loss_vals[-1]:.2e}, best={best_loss:.2e}, "
            f"stale={stale}/{patience}"
        )

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

    # Final BP pass to get consistent marginals, then convert to numpy
    with torch.no_grad():
        theta_bp = bp(list(theta))
        result = []
        for t in theta_bp:
            p = t.exp().cpu().numpy()
            p /= p.sum()
            result.append(p)

    return result
