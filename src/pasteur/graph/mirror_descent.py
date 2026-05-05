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
from .beliefs import BeliefPropagation, create_cliques, HAS_TRITON
from .linear_loss import LinearLoss

logger = logging.getLogger(__name__)


class MirrorDescentParams(TypedDict, total=False):
    lr: float
    max_iters: int
    ptol: float
    atol: float
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
    "atol": 1e-6,
    "patience": 50,
    "device": "auto",
    "compile": 50_000_000,
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
    atol: float = 2e-5,
    patience: int = 50,
    checkpoint_every: int = 100,
    device: torch.device | str | None = None,
    compile: int | bool = False,
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
        cliques,
        messages,
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
        # 1. BP forward
        with torch.no_grad():
            theta_bp = bp(list(theta))
            mu = [t.exp() for t in theta_bp]

        # 2. Loss + analytical gradient (no autograd)
        with torch.no_grad():
            loss, grads = loss_fn(mu)

        # 3. Assign gradients to theta (mirror descent: dL/dmu -> theta.grad)
        for t, g in zip(theta, grads):
            if t.grad is None:
                t.grad = g
            else:
                t.grad.copy_(g)

        return loss, mu

    total_params = sum(t.numel() for t in theta)
    do_compile = total_params >= compile if not isinstance(compile, bool) else compile
    if do_compile:
        logger.info("Compiling mirror descent compute graph...")
        compute_grad = torch.compile(compute_grad)

    logger.info(
        f"Mirror descent: {len(cliques)} cliques, {len(obs)} observations, "
        f"{total_params:_} params, "
        f"lr={lr}, device={device}, compile={do_compile}, triton={HAS_TRITON}, optim={optim}"
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
            diff_loss = best_loss - cur_loss
            if abs(diff_loss / best_loss) < ptol or abs(diff_loss) < atol:
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
    evidence_vars: set[str] | None = None,
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
        evidence_vars: Moral-graph nodes that will carry per-row evidence
            during sampling.  When given, all-pairs edges are added between
            them in a moral-graph copy before triangulation, forcing them
            into a single maximal clique.  That clique is then chosen as
            ``elim_root`` so forward sampling starts from the joint of all
            evidence variables — the only configuration where evidence
            propagates correctly through the rest of the tree.

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
        assert moral_graph is not None, "moral_graph is required for hugin tree modes"
        ev_present: list[str] = []
        if evidence_vars:
            ev_present = [v for v in evidence_vars if v in moral_graph]
            if len(ev_present) >= 2:
                from itertools import combinations as _combinations

                moral_graph = moral_graph.copy()
                for a, b in _combinations(ev_present, 2):
                    if not moral_graph.has_edge(a, b):
                        moral_graph.add_edge(a, b, evidence=True)
        if tree_mode != "hugin_comp":
            cap_heights(moral_graph, mode=tree_mode)
        elim_order, tri, _ = find_elim_order(
            moral_graph, attrs, elim_max_attempts, elim_factor_cost
        )
        junction = get_junction_tree(tri, attrs, compress=compress)

        # Stash the moral-graph elim order on the junction so the sampler can
        # follow it.  The first eliminated node (by design, a hist node with
        # ≥2 main neighbours) sits in a clique that captures its joint
        # structure with main vars — that's the natural root for evidence-
        # conditional sampling.  Without this, the sampler picks a max-domain
        # main-only clique and the hist evidence has no effect on what's
        # sampled there.
        import networkx as nx
        from .hugin import create_clique_meta as _ccm

        junction.graph["elim_order"] = list(elim_order)
        # Map each moral-graph node to its elim position.
        elim_pos = {n: i for i, n in enumerate(elim_order)}

        # For each maximal clique in the triangulated graph, compute its
        # CliqueMeta and the *earliest* elim position of any of its nodes —
        # that's when this clique started forming.  Multiple maximal cliques
        # can compress to the same CliqueMeta (when same-attr heights collapse);
        # we keep the minimum across them.
        clique_elim_idx: dict["CliqueMeta", int] = {}
        ev_set = set(ev_present)
        evidence_root_meta = None
        evidence_root_t = float("inf")
        for cl_nodes in nx.find_cliques(tri):
            cm = _ccm(cl_nodes, tri, attrs, compress=compress)
            if cm not in junction.nodes:
                continue
            t = min(elim_pos[n] for n in cl_nodes if n in elim_pos)
            cur = clique_elim_idx.get(cm)
            if cur is None or t < cur:
                clique_elim_idx[cm] = t
            if ev_set and ev_set.issubset(cl_nodes) and t < evidence_root_t:
                evidence_root_meta = cm
                evidence_root_t = t

        # Root = clique containing all evidence vars when present (forced by
        # the all-pairs edges added above; this is where evidence-conditional
        # forward sampling must start).  Otherwise pick the clique whose
        # creation time is earliest — by elim-order design that's the first
        # clique completed and tends to include hist nodes near the root.
        if evidence_root_meta is not None:
            elim_root_meta = evidence_root_meta
        else:
            elim_root_meta = (
                min(clique_elim_idx, key=clique_elim_idx.get)  # type: ignore[arg-type]
                if clique_elim_idx
                else None
            )
        junction.graph["elim_root"] = elim_root_meta
        junction.graph["clique_elim_idx"] = clique_elim_idx

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
    evidence_vars: set[str] | None = None,
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
        obs,
        attrs,
        tree_mode,
        compress,
        moral_graph,
        elim_max_attempts,
        elim_factor_cost,
        evidence_vars=evidence_vars,
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
