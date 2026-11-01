"""Exact alternating Newton refit of the positive/noise objective under a constant Gaussian prior.

Per-occurrence L2 and Adam both give rare users/items nearly full-size vectors. Here each vector solves
    sum_pos w softplus(-s) + nu * expected noise softplus(s) + lam/2 ||x||^2
so its prior stays fixed while its data grows, and sparse vectors shrink as a posterior mode should.
"""

import time

import numpy as np
import torch

from .lib import NI, NU

DEV = torch.device("mps")


@torch.no_grad()
def side_step(X, F, o, row, col, w, refs, nu, ridge, step=1.0, chunk=200_000, max_norm=1.0):
    """One damped Newton step for all rows of X (n,p), with partner features F (m,p) and offsets o (m,)."""
    n, p = X.shape
    g = torch.zeros(n, p, device=DEV)
    H = torch.zeros(n, p, p, device=DEV)
    for i in range(0, len(row), chunk):
        r, c, ww = row[i : i + chunk], col[i : i + chunk], w[i : i + chunk]
        f = F[c]
        s = (X[r] * f).sum(1) + o[c]
        q = torch.sigmoid(-s)
        g.index_add_(0, r, -(ww * q)[:, None] * f)
        h = (ww * q * (1 - q))[:, None, None] * (f[:, :, None] * f[:, None, :])
        H.index_add_(0, r, h)
    Fr, orf = F[refs], o[refs]
    FF = (Fr[:, :, None] * Fr[:, None, :]).reshape(len(refs), -1)
    scale = (nu / len(refs))[:, None]
    for i in range(0, n, 8192):
        Xc = X[i : i + 8192]
        sn = Xc @ Fr.T + orf
        pn = torch.sigmoid(sn)
        g[i : i + 8192] += scale[i : i + 8192] * (pn @ Fr)
        H[i : i + 8192] += (scale[i : i + 8192] * ((pn * (1 - pn)) @ FF)).reshape(-1, p, p)
    g += ridge * X
    H += torch.diag(ridge)
    d = torch.linalg.solve(H.cpu().double(), g.cpu().double()).float().to(DEV)
    norm = d.norm(dim=1, keepdim=True).clamp(min=1e-9)
    d = d * (max_norm / norm).clamp(max=1.0)
    X -= step * d
    return float(d.norm(dim=1).mean())


def refit(U, V, b, prop, pu, pi, w, lam_u=0.05, lam_v=0.05, lam_b=0.01, sweeps=4, nu=1.0, m=1024, seed=0, log=False):
    """Warm-started alternating Newton. Returns numpy (U, V, b)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pu_t, pi_t = torch.as_tensor(pu.astype(np.int64), device=DEV), torch.as_tensor(pi.astype(np.int64), device=DEV)
    wt = torch.as_tensor(w, dtype=torch.float32, device=DEV)
    Ut = torch.as_tensor(U, dtype=torch.float32, device=DEV).clone()
    Vb = torch.as_tensor(np.c_[V, b], dtype=torch.float32, device=DEV).clone()
    k = U.shape[1]
    n_u = np.bincount(pu, weights=w, minlength=NU)
    total = float(n_u.sum())
    user_p = n_u / total
    ridge_u = torch.full((k,), float(lam_u), device=DEV)
    ridge_i = torch.tensor([float(lam_v)] * k + [float(lam_b)], device=DEV)
    nu_u = torch.as_tensor(n_u * nu, dtype=torch.float32, device=DEV)
    nu_i = torch.as_tensor(prop * total * nu, dtype=torch.float32, device=DEV)
    active = torch.as_tensor(n_u > 0, device=DEV)
    for s in range(sweeps):
        ref_u = torch.as_tensor(rng.choice(NU, m, p=user_p), device=DEV)
        ref_i = torch.as_tensor(rng.choice(NI, m, p=prop), device=DEV)
        F = torch.cat([Ut, torch.ones(NU, 1, device=DEV)], 1)
        di = side_step(Vb, F, torch.zeros(NU, device=DEV), pi_t, pu_t, wt, ref_u, nu_i, ridge_i)
        du = side_step(Ut, Vb[:, :k], Vb[:, k], pu_t, pi_t, wt, ref_i, nu_u, ridge_u)
        Ut[~active] = 0
        if log:
            print(f"  sweep {s}: item step {di:.4f} user step {du:.4f} {time.time() - t0:.0f}s", flush=True)
    r = Vb.cpu().numpy()
    return Ut.cpu().numpy(), r[:, :k], r[:, k]
