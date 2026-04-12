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
    loss_type: str  # "l2", "l1", or "l1l2"
    elim_factor_cost: float  # Cost factor for elimination order clique domain
    elim_max_attempts: int  # Number of stochastic elimination order attempts
    tree: str  # "hugin", "maximal", "hugin_comp"


MIRROR_DESCENT_DEFAULT: MirrorDescentParams = {
    "lr": 1,
    "max_iters": 10_000,
    "ptol": 2e-4,
    "patience": 50,
    "device": "auto",
    "compile": 10_000_000,
    "optim": "line_search",
    "elim_factor_cost": 1.15,
    "elim_max_attempts": 5000,
    "tree": "hugin",
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
    checkpoint_every: int = 100,
    device: torch.device | str | None = None,
    compile: int = 10_000_000,
    optim: str = "sgd",
    init_potentials: dict[int, np.ndarray] | None = None,
    loss_type: str = "l2",
    block_unobserved: bool = False,
    # Backwards compat
    line_search: bool | None = None,
    **_,
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

    # Seed PyTorch from numpy's current RNG state (which make_deterministic controls)
    torch_seed = int(np.random.randint(0, 2**31))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.use_deterministic_algorithms(True)
    # Required for deterministic scatter/index ops on CUDA
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Build modules
    loss_fn = LinearLoss(obs, cliques, attrs, loss_type=loss_type).to(device)

    # Identify observed cliques (those targeted by at least one observation)
    observed: set[int] | None = None
    if block_unobserved:
        observed = set(loss_fn.cidx)

    bp = BeliefPropagation(
        cliques, messages,
        observed=observed,
        block_unobserved=block_unobserved,
    ).to(device)

    # Initialize potentials (uniform weighted prior in log-space)
    theta = create_cliques(cliques, attrs, device=device)

    # Warm start: override with previous model's raw theta (already log-space)
    if init_potentials:
        for idx, raw in init_potentials.items():
            theta[idx] = torch.from_numpy(raw).to(device).float()

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

                    if alpha < 1e-4:
                        converged = True
                        break

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

    # Save raw theta (pre-BP) for warm starting
    raw_theta = [t.detach().cpu().numpy() for t in theta]

    # Final BP pass to get consistent clique potentials
    with torch.no_grad():
        theta_bp = bp(list(theta))
        result = []
        for t in theta_bp:
            p = t.exp().cpu().numpy()
            p /= p.sum()
            result.append(p)

    return result, loss_fn, raw_theta


def build_junction_tree(
    obs: Sequence[LinearObservation],
    attrs: DatasetAttributes,
    tree_mode: str = "hugin",
    compress: bool = True,
    moral_graph=None,
    elim_max_attempts: int = 5000,
    elim_factor_cost: float = 1,
):
    """Build a junction tree and message schedule from observations.

    Args:
        obs: Linear observations (needed for maximal mode).
        attrs: Dataset attributes.
        tree_mode: "maximal", "hugin", "hugin_comp", "hugin_uncomp",
                   or "hugin_unvalley".
        compress: Whether to compress clique meta.
        moral_graph: Pre-built moral graph (required for hugin modes).
        elim_max_attempts: Number of stochastic elimination order attempts.

    Returns:
        (junction, cliques, messages)
    """
    from .beliefs import create_messages
    from .hugin import (
        cap_heights,
        find_elim_order,
        get_junction_tree,
        get_junction_tree_from_cliques,
        get_message_passing_order,
    )

    if tree_mode == "maximal":
        obs_cliques = [o.source for o in obs]
        junction = get_junction_tree_from_cliques(obs_cliques)
    else:
        assert moral_graph is not None, (
            "moral_graph is required for hugin tree modes"
        )
        if tree_mode != "hugin_comp":
            cap_heights(moral_graph, mode=tree_mode)
        _, tri, _ = find_elim_order(moral_graph, attrs, elim_max_attempts, elim_factor_cost)
        junction = get_junction_tree(tri, attrs, compress=compress)

    generations = get_message_passing_order(junction)
    cliques = list(junction.nodes())
    messages = create_messages(generations, attrs)

    return junction, cliques, messages


def fit_model(
    obs: Sequence[LinearObservation],
    attrs: DatasetAttributes,
    tree_mode: str = "hugin",
    compress: bool = True,
    moral_graph=None,
    device: torch.device | str | None = None,
    init_potentials: dict[int, np.ndarray] | None = None,
    **md_params,
):
    """Build junction tree and fit clique potentials via mirror descent.

    This is the generic entry point for any algorithm that produces
    observations (LinearObservation) and wants fitted clique potentials.

    Args:
        obs: Linear observations to fit.
        attrs: Dataset attributes.
        tree_mode: Junction tree construction mode.
        compress: Whether to compress clique meta.
        moral_graph: Pre-built moral graph (required for hugin modes).
        device: Torch device for mirror descent.
        init_potentials: Warm-start potentials keyed by clique index.
        **md_params: Additional mirror descent parameters (lr, max_iters,
                     ptol, patience, optim, loss_type, etc.).
            Also accepts elim_max_attempts and elim_factor_cost.

    Returns:
        (potentials, junction, cliques, messages, loss_fn, raw_theta)
    """
    elim_max_attempts = md_params.pop("elim_max_attempts", 5000)
    elim_factor_cost = md_params.pop("elim_factor_cost", 1)
    junction, cliques, messages = build_junction_tree(
        obs, attrs, tree_mode, compress, moral_graph, elim_max_attempts,
        elim_factor_cost,
    )

    potentials, loss_fn, raw_theta = mirror_descent(
        cliques,
        messages,
        obs,
        attrs,
        device=device,
        init_potentials=init_potentials,
        **md_params,
    )

    return potentials, junction, cliques, messages, loss_fn, raw_theta
