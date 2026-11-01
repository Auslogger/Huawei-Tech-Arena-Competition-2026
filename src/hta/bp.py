"""Belief propagation on the click graph, for unit preference vectors under an exponential click link.

Beliefs are von Mises-Fisher: a node's natural parameter eta is the sum of its incoming messages, so its mean direction is
eta / |eta| and its posterior mean has length A(|eta|). A click ties x and y through exp(kappa x.y). The message to one end
is kappa times the mean of the other end's cavity belief, which leaves out that same edge. So an edge never explains itself,
which is the self-reinforcement every point estimator here suffers from.
"""

import numpy as np
from scipy.special import ive

R = np.r_[np.linspace(0, 10, 201), np.linspace(10.25, 500, 1960)]


def _table(p):
    r = np.maximum(R, 1e-6)
    return ive(p / 2, r) / ive(p / 2 - 1, r)


class Mean:
    """Length of the mean of a vMF on S^(p-1) as a function of its concentration."""

    def __init__(self, p=8):
        self.t = _table(p)

    def __call__(self, r):
        return np.interp(r, R, self.t)


def run(pu, pi, w, y0, nu, ni, kappa=4.5, sweeps=6, damp=0.5, s0=20.0, white=True, log=None):
    """(eta_u, eta_i) from clicks (pu, pi, w) and start directions y0 (ni,p). Multiplicity w scales a click's message."""
    p = y0.shape[1]
    A = Mean(p)
    w = w.astype(np.float32)[:, None]
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    a = (kappa * A(s0) * unit(y0)[pi]).astype(np.float32)  # item to user
    c = np.zeros_like(a)  # user to item
    sum_by = lambda idx, v, n: np.stack([np.bincount(idx, weights=v[:, j], minlength=n) for j in range(p)], 1).astype(
        np.float32
    )
    for s in range(sweeps):
        eu = sum_by(pu, w * a, nu)
        cu = eu[pu] - w * a
        r = np.linalg.norm(cu, axis=1, keepdims=True)
        new = kappa * A(r) * cu / np.maximum(r, 1e-9)
        c = new if s == 0 else (1 - damp) * c + damp * new
        ei = sum_by(pi, w * c, ni)
        ci = ei[pi] - w * c
        if white:  # keep item directions isotropic (normally the partition function would take care of this)
            d_ = unit(ei)[np.bincount(pi, minlength=ni) > 0]
            d_ = d_ - d_.mean(0)
            S = d_.T @ d_ * p / len(d_)
            lam, Q = np.linalg.eigh(S)
            L = (Q * lam**-0.5) @ Q.T
            ci, ei = ci @ L, ei @ L
        r = np.linalg.norm(ci, axis=1, keepdims=True)
        new = kappa * A(r) * ci / np.maximum(r, 1e-9)
        a = (1 - damp) * a + damp * new
        if log:
            log(
                f"  sweep {s}: mean |eta_u| {np.linalg.norm(eu, axis=1).mean():.2f} mean |eta_i| {np.linalg.norm(ei, axis=1).mean():.2f}"
            )
    eu = sum_by(pu, w * a, nu)
    ei = sum_by(pi, w * c, ni)
    return eu, ei, a, c


def mean_of(eta, p=8):
    r = np.linalg.norm(eta, axis=1, keepdims=True)
    return Mean(p)(r) * eta / np.maximum(r, 1e-9)
