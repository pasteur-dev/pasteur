import logging
from typing import Sequence, TypedDict

import numpy as np
import torch

from ..utils.progress import piter, IS_AGENT

from ..attribute import DatasetAttributes
from .hugin import (
    CliqueMeta,
)
from .loss import LinearObservation
from .beliefs import BeliefPropagation, create_cliques
from .linear_loss import LinearLoss

logger = logging.getLogger(__name__)


class MirrorDescentParams(TypedDict, total=False):
    lr: float
    max_iters: int
    ptol: float
    patience: int
    device: str
    compile: int | bool
    optim: str  # "sgd", "line_search", or "adam"


MIRROR_DESCENT_DEFAULT: MirrorDescentParams = {
    "lr": 1,
    "max_iters": 10_000,
    "ptol": 2e-4,
    "patience": 50,
    "device": "auto",
    "compile": 10_000_000,
    "optim": "line_search",
}


def mirror_descent(
    cliques: Sequence[CliqueMeta],
    messages: Sequence,
    obs: Sequence[LinearObservation],
    attrs: DatasetAttributes,
    *,
    lr: float = 0.07,
    max_iters: int = 10_000,
    ptol: float = 2e-4,
    patience: int = 50,
    checkpoint_every: int = 50,
    device: torch.device | str | None = None,
    compile: int = 10_000_000,
    optim: str = "sgd",
    # Backwards compat
    line_search: bool | None = None,
) -> list[np.ndarray]:
    # Backwards compat: line_search=True -> optim="line_search"
    if line_search is not None:
        optim = "line_search" if line_search else "sgd"
    use_line_search = optim == "line_search"
    use_adam = optim == "adam"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Build modules
    bp = BeliefPropagation(cliques, messages).to(device)
    loss_fn = LinearLoss(obs, cliques, attrs).to(device)

    # Initialize potentials (uniform weighted prior in log-space)
    theta = create_cliques(cliques, attrs, device=device)

    theta = [t.requires_grad_(True) for t in theta]

    # Adam optimizer (uses mirror descent gradients ∂L/∂mu, not ∂L/∂theta)
    adam_opt = None
    if use_adam:
        adam_opt = torch.optim.Adam(theta, lr=lr)

    def compute_grad(theta, bp, loss_fn):
        # 1. Forward: BP to get consistent log-potentials, then exponentiate
        with torch.no_grad():
            theta_bp = bp(list(theta))
            mu = [t.exp() for t in theta_bp]

        # 2. Compute loss and gradient w.r.t. probability-space marginals
        mu_detached = mu
        mu = [m.requires_grad_(True) for m in mu]
        loss = loss_fn(mu)
        loss.backward()

        # 3. Apply marginal-space gradient directly to potentials
        #    (this is the correct mirror descent update: theta -= lr * ∂L/∂mu)
        for t, m in zip(theta, mu):
            if t.grad is None:
                t.grad = m.grad
            elif m.grad is not None:
                t.grad.copy_(m.grad)
        return loss, mu_detached

    total_params = sum(t.numel() for t in theta)
    do_compile = total_params >= compile if isinstance(compile, int) else compile
    if do_compile:
        logger.info("Compiling mirror descent compute graph...")
        compute_grad = torch.compile(compute_grad)

    logger.info(
        f"Mirror descent: {len(cliques)} cliques, {len(obs)} observations, "
        f"{total_params:_} params, "
        f"lr={lr}, device={device}, compile={do_compile}, optim={optim}"
    )

    alpha = torch.tensor(lr, device=device)
    best_loss = float("inf")
    stale = 0
    total_iters = 0
    converged = False
    pbar = piter(range(max_iters), total=max_iters, desc="Mirror descent")
    prev_loss, prev_mu, prev_grads = None, None, None

    while total_iters < max_iters:
        block_size = min(checkpoint_every, max_iters - total_iters)
        losses = []
        for _ in range(block_size):
            loss, mu = compute_grad(theta, bp, loss_fn)
            losses.append(loss)

            with torch.no_grad():
                if use_adam:
                    # Adam handles the step internally using the mirror
                    # descent gradients (∂L/∂mu) already stored in theta.grad
                    pass

                elif use_line_search:
                    # Adjust alpha using previous iteration's Armijo condition
                    # (delayed by one step to avoid a duplicate BP pass)
                    if (
                        prev_grads is not None
                        and prev_loss is not None
                        and prev_mu is not None
                    ):
                        dot = sum(
                            (g * (m1 - m2)).sum()
                            for g, m1, m2 in zip(prev_grads, prev_mu, mu)
                            if g is not None
                        )
                        sufficient = (prev_loss - loss) >= 0.5 * alpha * dot
                        alpha = torch.where(sufficient, alpha * 1.01, alpha * 0.5)

                    prev_grads = [
                        t.grad.clone() if t.grad is not None else None for t in theta
                    ]
                    prev_loss = loss
                    prev_mu = mu

            if use_adam:
                adam_opt.step()
                adam_opt.zero_grad()
            else:
                with torch.no_grad():
                    # Step: theta = theta - alpha * grad
                    for t in theta:
                        if t.grad is not None:
                            t.sub_(alpha * t.grad)
                            t.grad.zero_()

        total_iters += block_size
        pbar.update(block_size)

        # Sync GPU and check convergence
        loss_vals = [l.item() for l in losses]
        for cur_loss in loss_vals:
            if abs((best_loss - cur_loss) / best_loss) < ptol:
                stale += 1
            else:
                stale = 0
            if cur_loss < best_loss:
                best_loss = cur_loss

        desc = (
            f"Mirror descent: loss={loss_vals[-1]:.2e}, best={best_loss:.2e}, "
            f"stale={stale}/{patience}"
        )
        if use_line_search:
            desc += f", alpha={alpha.item():.2e}"
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
            f"(best loss={best_loss:.6e}, ptol={ptol})."
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
