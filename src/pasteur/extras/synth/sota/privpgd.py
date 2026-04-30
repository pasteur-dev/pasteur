"""PrivPGD: Differentially Private Particle Gradient Descent.

Generates n_particles in the [0,1]^d hypercube (one dim per column) and
optimizes them via sliced 2-Wasserstein minimization against noisy k-way
marginals. Each marginal is first denoised by a sliced 1-Wasserstein
projection onto the probability simplex, then quantized into a target
particle cloud.

Reference: private-pgd-ref/examples/privpgd.py
"""

from __future__ import annotations

import itertools
import logging
import random as py_random
from math import sqrt
from typing import TYPE_CHECKING, Any, cast

from ....attribute import Attributes, DatasetAttributes
from ....marginal import MarginalOracle
from ....synth import Synth, make_deterministic
from ....utils import LazyFrame, data_to_tables, tables_to_data
from ....utils.progress import piter
from .common import (
    cdp_rho,
    _col_to_attr_sel,
    attr_domain_size,
    clique_domain_size,
    get_attr_names,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch

logger = logging.getLogger(__name__)


# ============================================================
# Hypercube embedding
# ============================================================
def _bin_centers(k: int, device: torch.device) -> torch.Tensor:
    """K bin centers in [0,1] for an attribute of domain size K."""
    import numpy as np
    import torch

    return torch.tensor(
        (np.arange(k) * 2 + 1) / (2 * k),
        dtype=torch.float32,
        device=device,
    )


def _clique_centers(
    cl: tuple[str, ...],
    col_dim: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """[n_cells, k] tensor of clique-cell center coordinates in [0,1]^k.

    Row i corresponds to cell index i in row-major (C) order over the
    per-column domain sizes [col_dim[c] for c in cl]."""
    import torch

    centers_per_col = [_bin_centers(col_dim[c], device) for c in cl]
    grids = torch.meshgrid(*centers_per_col, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1)


# ============================================================
# Sliced Wasserstein utilities
# ============================================================
def _random_directions(
    num_projections: int, k: int, device: torch.device
) -> torch.Tensor:
    import torch

    theta = torch.randn(
        num_projections, k, device=device, dtype=torch.float32
    )
    return theta / (theta.norm(dim=1, keepdim=True) + 1e-9)


def _sliced_w1_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    centers: torch.Tensor,
    num_projections: int,
) -> torch.Tensor:
    """Sliced 1-Wasserstein between weights u and v on shared support."""
    import torch

    n, k = centers.shape
    directions = _random_directions(num_projections, k, centers.device)
    proj = torch.mm(centers - 0.5, directions.t())
    proj_sorted, indices = torch.sort(proj, dim=0)
    u_sorted = u[indices]
    v_sorted = v[indices]
    distances = torch.diff(proj_sorted, dim=0)
    cu = torch.cumsum(u_sorted, dim=0)[:-1, :]
    cv = torch.cumsum(v_sorted, dim=0)[:-1, :]
    return (torch.abs(cu - cv) * distances).sum(dim=0).mean()


def _project_to_probability(
    y_norm: torch.Tensor,
    centers: torch.Tensor,
    iters: int,
    num_projections: int,
    lr: float,
    step: int,
    gamma: float,
) -> torch.Tensor:
    """Find probability vector closest to y_norm in sliced-1-Wasserstein.

    y_norm may be signed (a noisy estimate of a probability vector);
    the result is a non-negative vector summing to 1."""
    import torch

    init = torch.clamp(y_norm, min=0.0)
    s = init.sum()
    if s <= 0:
        init = torch.ones_like(y_norm) / y_norm.numel()
    else:
        init = init / s
    params = torch.log(init + 1e-9).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step, gamma=gamma
    )
    for _ in range(iters):
        optimizer.zero_grad()
        p = torch.softmax(params, dim=0)
        loss = _sliced_w1_loss(p, y_norm, centers, num_projections)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return torch.softmax(params, dim=0).detach()


def _quantize(
    centers: torch.Tensor,
    weights: torch.Tensor,
    n_particles: int,
) -> torch.Tensor:
    """Replicate clique-cell centers proportional to weights.

    Returns a [n_particles, k] tensor whose empirical distribution
    approximates the discrete distribution given by (centers, weights)."""
    import torch

    counts = (weights / (weights.sum() + 1e-12) * n_particles).long()
    counts = counts.clamp(min=0)
    out = centers.repeat_interleave(counts, dim=0)
    leftover = n_particles - out.shape[0]
    if leftover > 0:
        idx = torch.multinomial(
            weights.clamp(min=0) + 1e-12, leftover, replacement=True
        )
        out = torch.cat([out, centers[idx]], dim=0)
    elif leftover < 0:
        out = out[:n_particles]
    return out


def _sw2_squared_and_grad(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_projections: int,
) -> tuple[float, torch.Tensor]:
    """Sliced 2-Wasserstein squared and gradient w.r.t. X.

    Both X and Y are [n_particles, k]. Sorts projections along each
    random direction then scatters the gradient back to original X rows."""
    import torch

    n, k = X.shape
    directions = _random_directions(num_projections, k, X.device)
    pX = torch.mm(X, directions.t())
    pY = torch.mm(Y, directions.t())
    Xs, idx_X = torch.sort(pX, dim=0)
    Ys, _ = torch.sort(pY, dim=0)
    diff = Xs - Ys
    loss = (diff ** 2).mean()
    expanded_diff = diff.unsqueeze(2).expand(-1, -1, k)
    expanded_theta = directions.unsqueeze(0).expand(n, -1, -1)
    grads = 2 * expanded_diff * expanded_theta
    grad_back = torch.zeros_like(grads)
    for d in range(k):
        grad_back[:, :, d].scatter_(0, idx_X, grads[:, :, d])
    return float(loss.item()), grad_back.mean(dim=1) * 100


def _mask_grad(grad: torch.Tensor, p_mask: int) -> torch.Tensor:
    import torch

    if p_mask <= 0:
        return grad
    n_mask = int(grad.numel() * p_mask / 100)
    idx = torch.randperm(grad.numel(), device=grad.device)[:n_mask]
    grad.view(-1)[idx] = 0
    return grad


class PrivPGD(Synth):
    name = "privpgd"
    type = "idx"
    tabular = True
    multimodal = False
    timeseries = False
    parallel = True

    def __init__(
        self,
        e: float = 1.0,
        etotal: float | None = None,
        delta: float = 1e-9,
        degree: int = 2,
        max_cells: int = 10000,
        n_particles: int = 100000,
        iters: int = 1000,
        lr: float = 0.1,
        scheduler_step: int = 50,
        scheduler_gamma: float = 0.75,
        num_projections: int = 10,
        p_mask: int = 80,
        batch_size: int = 5,
        iters_proj: int = 1750,
        num_projections_proj: int = 200,
        scheduler_step_proj: int = 100,
        scheduler_gamma_proj: float = 0.8,
        marginal_mode: "MarginalOracle.MODES" = "out_of_core",
        marginal_worker_mult: int = 1,
        marginal_min_chunk: int = 100,
        seed: int | None = None,
        n: int | None = None,
        partitions: int | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.e = etotal if etotal is not None else e
        self.delta = delta
        self.degree = degree
        self.max_cells = max_cells
        self.n_particles = n_particles
        self.iters = iters
        self.lr = lr
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.num_projections = num_projections
        self.p_mask = p_mask
        self.batch_size = batch_size
        self.iters_proj = iters_proj
        self.num_projections_proj = num_projections_proj
        self.scheduler_step_proj = scheduler_step_proj
        self.scheduler_gamma_proj = scheduler_gamma_proj
        self.marginal_mode = marginal_mode
        self.marginal_worker_mult = marginal_worker_mult
        self.marginal_min_chunk = marginal_min_chunk
        self.seed = seed
        self.n = n
        self.partitions = partitions
        self.device = device
        self.kwargs = kwargs

    @make_deterministic
    def preprocess(self, meta: dict[str, Attributes], data: dict[str, LazyFrame]):
        self.table = next(iter(meta))
        self.attrs = meta
        self._n = data[self.table].shape[0]
        self._partitions = len(data[self.table])

    @make_deterministic
    def bake(self, data: dict[str, LazyFrame]):
        pass

    @make_deterministic
    def fit(self, data: dict[str, LazyFrame]):
        import numpy as np
        import torch

        ids, tables = data_to_tables(data)
        table = tables[self.table]
        self.partitions = self.partitions or len(table)
        self.n = self.n or (table.shape[0] // self.partitions)
        n = self._n

        table_attrs: DatasetAttributes = {None: self.attrs[self.table]}
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        all_attrs = get_attr_names(table_attrs)
        col_dim = {a: attr_domain_size(a, table_attrs) for a in all_attrs}
        col_idx = {a: i for i, a in enumerate(all_attrs)}
        d = len(all_attrs)

        cliques = list(itertools.combinations(all_attrs, self.degree))
        cliques = [
            cl
            for cl in cliques
            if clique_domain_size(cl, table_attrs) <= self.max_cells
        ]
        if not cliques:
            raise RuntimeError(
                f"PrivPGD: no cliques pass max_cells={self.max_cells} filter"
            )
        logger.info(
            f"PrivPGD: {len(cliques)} workload cliques (degree={self.degree})"
        )

        # Privacy: rho-CDP, distribute equally across queries.
        # sigma = sqrt(W / (2*rho)) matches the sensitivity-1 Gaussian
        # convention used by AIM/MST in pasteur (rho per query = 0.5/sigma^2).
        rho = cdp_rho(self.e, self.delta)
        sigma = sqrt(len(cliques) / (2 * rho))
        logger.info(f"PrivPGD: rho={rho:.4f}, sigma={sigma:.2f}")

        # ----- Phase 1: measure all cliques in one batched call -----
        with MarginalOracle(
            data,
            table_attrs,
            mode=self.marginal_mode,
            min_chunk_size=self.marginal_min_chunk,
            max_worker_mult=self.marginal_worker_mult,
        ) as oracle:
            requests = []
            for cl in cliques:
                req: dict[str, dict[str, int]] = {}
                for col_name in cl:
                    attr_name, sel = _col_to_attr_sel(col_name, table_attrs)
                    if attr_name in req:
                        req[attr_name].update(sel)
                    else:
                        req[attr_name] = dict(sel)
                requests.append(list(req.items()))
            results = oracle.process(requests, postprocess=None)

        # ----- Phase 2: project + quantize each marginal -> target cloud -----
        logger.info("PrivPGD: projection step (denoising via SW1)")
        targets: dict[tuple[str, ...], torch.Tensor] = {}
        clique_dims: dict[tuple[str, ...], torch.Tensor] = {}
        for cl, raw in zip(
            piter(cliques, desc="PrivPGD projection"), results
        ):
            req_dims = [col_dim[c] for c in cl]
            x = raw.ravel().astype(np.float64).reshape(req_dims)
            noise = np.random.normal(0, sigma, size=x.shape)
            y = (x + noise).reshape(-1) / max(n, 1)
            y_t = torch.tensor(y, dtype=torch.float32, device=device)

            centers = _clique_centers(cl, col_dim, device)
            yprob = _project_to_probability(
                y_t,
                centers,
                iters=self.iters_proj,
                num_projections=self.num_projections_proj,
                lr=0.1,
                step=self.scheduler_step_proj,
                gamma=self.scheduler_gamma_proj,
            )
            targets[cl] = _quantize(centers, yprob, self.n_particles)
            clique_dims[cl] = torch.tensor(
                [col_idx[c] for c in cl], dtype=torch.long, device=device
            )

        # ----- Phase 3: particle gradient descent (SW2 minimization) -----
        logger.info(
            f"PrivPGD: particle GD (n_particles={self.n_particles}, "
            f"d={d}, iters={self.iters})"
        )
        X = torch.rand(
            self.n_particles, d, dtype=torch.float32, device=device
        ).requires_grad_(True)

        if self.p_mask > 0:
            optimizer = torch.optim.SparseAdam([X], lr=self.lr)
        else:
            optimizer = torch.optim.Adam([X], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step,
            gamma=self.scheduler_gamma,
        )

        clique_keys = list(targets.keys())
        bs = (
            len(clique_keys) if self.batch_size == 0 else self.batch_size
        )
        order = list(range(len(clique_keys)))

        for epoch in piter(range(self.iters), desc="PrivPGD GD"):
            py_random.shuffle(order)
            total_loss = 0.0
            for s in range(0, len(clique_keys), bs):
                batch = [clique_keys[i] for i in order[s : s + bs]]
                grad_mat = torch.zeros_like(X)
                for cl in batch:
                    Y = targets[cl]
                    dims = clique_dims[cl]
                    X_sel = X.detach()[:, dims]
                    loss, gX = _sw2_squared_and_grad(
                        X_sel, Y, self.num_projections
                    )
                    grad_mat[:, dims] += gX.detach()
                    total_loss += loss

                if self.p_mask > 0:
                    X.grad = _mask_grad(grad_mat, self.p_mask).to_sparse()
                else:
                    X.grad = grad_mat

                optimizer.step()
                X.data.clamp_(0.0, 1.0)

            scheduler.step()
            log_every = max(1, self.iters // 10)
            if epoch == 0 or (epoch + 1) % log_every == 0:
                logger.info(
                    f"PrivPGD epoch {epoch + 1}/{self.iters}: "
                    f"loss={total_loss:.6f}"
                )

        self.X = X.detach()
        self.col_dim = col_dim
        self.col_idx = col_idx
        self.all_attrs = all_attrs
        self.table_attrs = table_attrs
        self._device = device

    def _discretize(self, X: torch.Tensor) -> dict[str, np.ndarray]:
        """Map [n, d] particles in [0,1] to integer bin indices per column."""
        import torch

        out = {}
        for col_name in self.all_attrs:
            j = self.col_idx[col_name]
            K = self.col_dim[col_name]
            v = X[:, j].clamp(0.0, 1.0)
            # Bin centers at (2k+1)/(2K). Nearest center index = floor(v*K)
            # clipped to [0, K-1] (ties at v=1 land in bin K-1).
            idx = torch.clamp((v * K).floor().long(), min=0, max=K - 1)
            out[col_name] = idx.cpu().numpy()
        return out

    @make_deterministic("i")
    def sample_partition(self, *, n: int, i: int = 0) -> dict[str, Any]:
        import pandas as pd
        import torch

        n = n or self.n
        idx = torch.randint(
            0, self.X.shape[0], (n,), device=self._device
        )
        Xs = self.X[idx]
        col_vals = self._discretize(Xs)

        all_attrs = cast(Attributes, self.table_attrs[None])
        out_cols: dict[str, np.ndarray] = {}
        for col_name, vals in col_vals.items():
            attr_name, sel = _col_to_attr_sel(col_name, self.table_attrs)
            attr = all_attrs[attr_name]
            if len(sel) == 1:
                vname = next(iter(sel))
                out_cols[vname] = vals
            else:
                from ....graph.sample import _decompose_dim

                decomposed = _decompose_dim(attr, sel, vals)
                for vname in attr.vals:
                    if vname in decomposed:
                        out_cols[vname] = decomposed[vname]

        df = pd.DataFrame(out_cols)
        return tables_to_data(
            {self.table: pd.DataFrame()},
            {self.table: df},
            partition=i if self.partitions > 1 else None,
        )
