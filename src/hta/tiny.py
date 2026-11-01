"""A few-parameter outside-option head on per-dimension dots: a fast, fair proxy for a table's value in the stack.

The fitted bilinear head is barely better than this, so a table can be judged in seconds without head and tree.
"""

import numpy as np
import torch

from . import pipe, states
from .lib import auc


def rows(d, ev, st, top=16, local=True, l2=0.02, repeat=True):
    """(n,10,c) features: one dot per leading singular direction of V, the remainder, the bias and the repeat scalars."""
    U, V, b, prop = st
    Vt = np.linalg.svd(V, full_matrices=False)[2]
    u = states.local_infer(U[d.user[ev]], V, b, prop, d.hist_item[ev], l2=l2) if local else U[d.user[ev]]
    c = d.cand[ev]
    Pu, Pv = u @ Vt[:top].T, V @ Vt[:top].T
    X = [Pu[:, j, None] * Pv[c, j] for j in range(Pu.shape[1])]
    if Vt.shape[0] > top:
        P = Vt[top:].T
        X.append(((u @ P)[:, None, :] * (V @ P)[c]).sum(2))
    X.append(b[c])
    if repeat:
        r = pipe.repeat_x(d, ev)
        X += [r[..., 0], r[..., 1], r[..., 2]]
    return np.stack(X, -1).astype(np.float64)


def fit(Xs, ys):
    Xt, Y = [torch.as_tensor(x) for x in Xs], [torch.as_tensor(y, dtype=torch.float64) for y in ys]
    beta = torch.zeros(Xs[0].shape[-1], dtype=torch.float64, requires_grad=True)
    c0 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, c0], lr=1, max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-7)
    n = sum(len(y) for y in ys)

    def closure():
        opt.zero_grad()
        total = 0.0
        for x, y in zip(Xt, Y):
            z = x @ beta + c0
            total = (
                total + (torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1)) - (z * y).sum(1)).sum() / n
            )
        total.backward()
        return total

    opt.step(closure)
    return beta.detach(), c0.detach()


def marginal(x, beta, c0):
    z = torch.as_tensor(x) @ beta + c0
    return (z - torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1))[:, None]).numpy()


def screen(d, table, cut, q, first=21, **kw):
    """All-delay AUC on q of tables `table(day)`: tiny head fit on days first..cut-1, each scored by its own day's table."""
    Xs, ys = [], []
    for day in range(first, cut):
        ev = pipe.days_of(d, day)
        Xs.append(rows(d, ev, table(day), **kw))
        ys.append(d.y[ev])
    beta, c0 = fit(Xs, ys)
    return auc(d.y[q], marginal(rows(d, q, table(cut), **kw), beta, c0)), beta.numpy()
