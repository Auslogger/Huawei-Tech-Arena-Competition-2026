"""Convex-ish shared bilinear choice head with an outside option, fit by L-BFGS on CPU."""
import time

import numpy as np
import torch

from .lib import THREADS

torch.set_num_threads(THREADS)


def choice_nll(z, y):
    """Multinomial NLL per candidate row; the extra 1 in the denominator is the no-click option."""
    log1p_sum = torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1))
    return (log1p_sum - (z * y).sum(1)).sum() / z.numel()


def marginal(z):
    """Marginal row probability under the outside-option model, as log-odds-free score in (0,1)."""
    return z - torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1))[:, None]


class Bilinear(torch.nn.Module):
    def __init__(self, a, b, c, blocks=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(a + 1, b + 1))
        # blocks=(k1,k2,...): keep linear terms and per-view interactions, drop cross-view interactions
        mask = torch.ones(a + 1, b + 1)
        if blocks is not None:
            mask[1:, 1:] = 0
            i = 1
            for k in blocks:
                mask[i:i + k, i:i + k] = 1
                i += k
        self.register_buffer("mask", mask)
        self.beta = torch.nn.Parameter(torch.zeros(c)) if c else None
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, Ua, Va, X=None):
        # Ua: (n,a+1), Va: (n,10,b+1), X: (n,10,c)
        z = ((Ua @ (self.W * self.mask.to(self.W.dtype)))[:, None, :] * Va).sum(2) + self.bias
        if self.beta is not None and X is not None:
            z = z + X @ self.beta
        return z


def augment(U):
    return torch.cat([torch.ones_like(U[..., :1]), U], -1)


def fit(U, V, cand, y, X=None, lam=.01, iters=300, verbose=False):
    """U (n,a), V (items,b), cand (n,10), y (n,10), X (n,10,c) -> trained module.

    Objective: choice NLL per row + .5 lam ||W, beta||^2 (bias is free), in float64 with a real convergence test.
    """
    Ua = augment(torch.as_tensor(U, dtype=torch.float64))
    Vt = augment(torch.as_tensor(V, dtype=torch.float64))
    c = torch.as_tensor(cand.astype(np.int64))
    Va = Vt[c]
    Y = torch.as_tensor(y, dtype=torch.float64)
    Xt = None if X is None else torch.as_tensor(X, dtype=torch.float64)
    m = Bilinear(Ua.shape[1] - 1, Vt.shape[1] - 1, 0 if X is None else X.shape[-1]).double()
    opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=iters, max_eval=iters * 2, history_size=30,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-12)
    t = time.time()

    def closure():
        opt.zero_grad()
        z = m(Ua, Va, Xt)
        reg = (m.W ** 2).sum() + (0 if m.beta is None else (m.beta ** 2).sum())
        loss = choice_nll(z, Y) + .5 * lam * reg
        loss.backward()
        return loss

    opt.step(closure)
    if verbose:
        st = opt.state_dict()["state"][0]
        print(f"fit obj {float(closure()):.6f} iters {st['n_iter']} evals {st['func_evals']} {time.time() - t:.1f}s",
              flush=True)
    return m


@torch.no_grad()
def score(m, U, V, cand, X=None):
    Ua = augment(torch.as_tensor(U, dtype=torch.float64))
    Vt = augment(torch.as_tensor(V, dtype=torch.float64))
    Va = Vt[torch.as_tensor(cand.astype(np.int64))]
    z = m(Ua, Va, None if X is None else torch.as_tensor(X, dtype=torch.float64))
    return marginal(z).numpy()


class Group:
    """Events sharing one item table. U (n,a), V (items,b), cand (n,10), y (n,10)|None, X (n,10,c), w (n,)|None."""

    def __init__(self, U, V, cand, y=None, X=None, w=None):
        self.U = augment(torch.as_tensor(np.asarray(U, np.float32)))
        self.V = augment(torch.as_tensor(np.asarray(V, np.float32)))
        self.c = torch.as_tensor(cand.astype(np.int64))
        self.y = None if y is None else torch.as_tensor(y.astype(np.float32))
        self.X = None if X is None else torch.as_tensor(np.asarray(X, np.float32))
        self.w = None if w is None else torch.as_tensor(np.asarray(w, np.float32))
        self.n = len(self.c)


def _chunk(g, i, j):
    return g.U[i:j].double(), g.V[g.c[i:j]].double(), None if g.X is None else g.X[i:j].double()


def fit_groups(groups, lam=.1, iters=300, chunk=40000, verbose=False, init=None, blocks=None, lam_b=None):
    a, b = groups[0].U.shape[1] - 1, groups[0].V.shape[1] - 1
    c = 0 if groups[0].X is None else groups[0].X.shape[-1]
    m = Bilinear(a, b, c, blocks=blocks).double() if init is None else init
    total = sum(float(g.n if g.w is None else g.w.sum()) for g in groups) * 10
    opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=iters, max_eval=iters * 2, history_size=30,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-12)
    t = time.time()

    def closure():
        opt.zero_grad()
        val = 0.
        for g in groups:
            for i in range(0, g.n, chunk):
                j = min(i + chunk, g.n)
                z = m(*_chunk(g, i, j))
                y = g.y[i:j].double()
                ll = torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1)) - (z * y).sum(1)
                ll = (ll if g.w is None else ll * g.w[i:j].double()).sum() / total
                ll.backward()
                val += float(ll.detach())
        reg = .5 * lam * ((m.W ** 2).sum() + (0 if m.beta is None else (m.beta ** 2).sum()))
        reg.backward()
        return val + float(reg.detach())

    opt.step(closure)
    if verbose:
        st = opt.state_dict()["state"][0]
        print(f"fit iters {st['n_iter']} evals {st['func_evals']} {time.time() - t:.0f}s", flush=True)
    return m


@torch.no_grad()
def score_group(m, g, chunk=40000):
    return np.concatenate([marginal(m(*_chunk(g, i, min(i + chunk, g.n)))).numpy() for i in range(0, g.n, chunk)])
