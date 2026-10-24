"""Convex-ish shared bilinear choice head with an outside option, fit by L-BFGS on CPU."""
import time

import numpy as np
import torch

torch.set_num_threads(8)


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
