"""Shared data access, splits, metrics and fast linear algebra."""

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
T0 = 1737331200  # 2025-01-20 00:00 UTC
DAY = 86400
NU, NI = 54456, 50000
THREADS = int(
    os.environ.get("HTA_THREADS", 8)
)  # 8 is faster than 10 here because the efficiency cores stall the performance threads


def day_ts(jan):
    """Unix time of 00:00 UTC on the given January day."""
    return T0 + (jan - 20) * DAY


class D:
    """Lazy holder of parsed event arrays (mmap-free, loaded once)."""

    def __init__(self):
        z = np.load(CACHE / "events.npz")
        self.n_train = int(z["n_train"])
        for k in ["user", "ts", "bucket", "day_type", "cand", "y", "hist_item", "hist_ts", "hist_len"]:
            setattr(self, k, z[k])
        self.n = len(self.ts)


def first_events(d, start, end, base=None):
    """Index of each user's first event in [start,end), ties by export order (labelled rows only by default)."""
    base = np.arange(d.n_train) if base is None else base
    idx = base[(d.ts[base] >= start) & (d.ts[base] < end)]
    idx = idx[np.argsort(d.ts[idx], kind="stable")]
    _, first = np.unique(d.user[idx], return_index=True)
    return np.sort(idx[first])


def auc(y, s):
    y = np.asarray(y).ravel().astype(bool)
    r = rankdata(np.asarray(s).ravel())
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _wauc(y, w):
    """AUC of rows already sorted by score, with row weights w. Score ties are broken by sort order."""
    pos, neg = w * y, w * (1 - y)
    return (pos * np.cumsum(neg)).sum() / (pos.sum() * neg.sum())


def paired_gain(y, new, base, B=300, seed=0):
    """Global AUC gain of new over base with (q05, q95) from resampling whole events. y, new, base are (n,10)."""
    y, new, base = np.asarray(y), np.asarray(new), np.asarray(base)
    Y = y.ravel().astype(np.float64)
    a, b = np.argsort(new.ravel()), np.argsort(base.ravel())
    rng = np.random.default_rng(seed)
    g = np.zeros(B)
    for i in range(B):
        w = np.repeat(np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y)), y.shape[1]).astype(np.float64)
        g[i] = _wauc(Y[a], w[a]) - _wauc(Y[b], w[b])
    return auc(y, new) - auc(y, base), float(np.quantile(g, 0.05)), float(np.quantile(g, 0.95))


def check_query(d, ev, cut):
    """Leakage guard: a cut's query rows hold no click at or after the cutoff in their supplied histories."""
    assert d.hist_ts[ev].max() < day_ts(cut), f"cut {cut}: query history reaches the cutoff"


def leak_gap(name, y_fit, s_fit, y_next, s_next, tol=0.03):
    """Standing check for supervised stages: rows they fit on must not score far above rows they have not seen.

    A fitted table holding its training events' labels shows AUC .92 on them against .77 next day (attempts.md).
    """
    a, b = auc(y_fit, s_fit), auc(y_next, s_next)
    print(f"  leak check {name}: fit rows {a:.4f}, unseen rows {b:.4f}", flush=True)
    assert a - b < tol, f"{name}: {a:.4f} on its fit rows against {b:.4f} unseen, labels are in its inputs"


def rsvd(B, k, oversample=20, power=4, seed=0):
    """Randomized SVD of a sparse matrix. Returns U, s, Vt sorted by s, descending."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B.shape[1], k + oversample))
    for _ in range(power):
        Q, _ = np.linalg.qr(B @ Q)
        Q, _ = np.linalg.qr(B.T @ Q)
    Q, _ = np.linalg.qr(B @ Q)
    Ub, s, Vt = np.linalg.svd((B.T @ Q).T, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def norm_incidence(u, i, shape=(NU, NI), binary=True):
    C = sp.csr_matrix((np.ones(len(u), np.float64), (u, i)), shape=shape)
    C.sum_duplicates()
    if binary:
        C.data = np.minimum(C.data, 1)
    du = np.asarray(C.sum(1)).ravel()
    di = np.asarray(C.sum(0)).ravel()
    with np.errstate(divide="ignore"):
        Du = sp.diags(np.where(du > 0, du**-0.5, 0))
        Di = sp.diags(np.where(di > 0, di**-0.5, 0))
    return (Du @ C @ Di).tocsr(), du, di
