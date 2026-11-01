"""Semi-synthetic twin of the data: real users, slates and times, with clicks drawn from a known sphere model.

Truth is known, so estimators can be judged by their real error. Usage: python -m hta.synth kappa [seed]. Writes
cache/synth_k{kappa}_s{seed}.npz. `load` returns an object with the same fields as lib.D, plus X and Y, the true vectors.
"""

import sys

import numpy as np

from . import states
from .lib import CACHE, D, DAY, NI, NU, T0, day_ts


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def build(d, kappa=6.3, seed=0, dim=8, rate=0.629, tau=0.0, own=0):
    """tau: sd of a true user intercept on the utility. own=1 reproduces the first twin, with warm-up clicks from the user's own
    slates. own=0 draws them from slates of users with the same main exposure group, so they fall outside the user's offer list."""
    rng = np.random.default_rng(seed)
    X, Y = (
        unit(rng.standard_normal((NU, dim))).astype(np.float32),
        unit(rng.standard_normal((NI, dim))).astype(np.float32),
    )
    n = d.n_train
    cand, user = d.cand[:n], d.user[:n]
    zz = np.zeros((n, 10), np.float32)
    for i in range(0, n, 100000):
        zz[i : i + 100000] = (X[user[i : i + 100000]][:, None, :] * Y[cand[i : i + 100000]]).sum(2)
    zz *= kappa
    cu = rng.standard_normal(NU).astype(np.float32) * tau
    zz += cu[user][:, None]
    lo, hi = -15.0, 5.0  # intercept matching the event click rate
    for _ in range(40):
        c0 = (lo + hi) / 2
        p = 1 - 1 / (1 + np.exp(zz + c0).sum(1))
        lo, hi = (lo, c0) if p.mean() > rate else (c0, hi)
    e = np.exp(zz + c0)
    P = e / (1 + e.sum(1, keepdims=True))
    cum = np.cumsum(np.c_[P, 1 - P.sum(1)], 1)
    pick = (rng.random((n, 1)) > cum).sum(1)
    y = np.zeros((n, 10), np.int8)
    r = np.where(pick < 10)[0]
    y[r, pick[r]] = 1
    # warm-up clicks: same per-user count as the real data, each drawn from one of the user's own slates given a click
    pu, pi, pt, w = states.positives(d, day_ts(27))
    keep = pt < T0
    wu = np.repeat(pu[keep], w[keep].astype(np.int64))
    wt = np.repeat(pt[keep], w[keep].astype(np.int64)).astype(np.int64)
    if own:
        key, keyw = user, wu
        K = NU
    else:
        lab = np.load(CACHE / "clusters_c27_K16.npy")
        counts = np.zeros((NU, 16), np.int32)
        np.add.at(counts, (np.repeat(user, 10), lab[cand].ravel()), 1)
        group = counts.argmax(1)
        key, keyw, K = group[user], group[wu], 16
    order = np.argsort(key, kind="stable")
    off = np.r_[0, np.cumsum(np.bincount(key, minlength=K))]
    cnt = off[keyw + 1] - off[keyw]
    pickev = np.where(
        cnt > 0,
        order[off[keyw] + (rng.random(len(wu)) * cnt).astype(np.int64).clip(max=np.maximum(cnt - 1, 0))],
        rng.integers(n, size=len(wu)),
    )
    Cw = cand[pickev]
    zw = np.zeros(Cw.shape, np.float32)
    for i in range(0, len(wu), 100000):
        zw[i : i + 100000] = (X[wu[i : i + 100000]][:, None, :] * Y[Cw[i : i + 100000]]).sum(2)
    ew = np.exp(zw * kappa)
    pw = np.cumsum(ew / ew.sum(1, keepdims=True), 1)
    wi = Cw[np.arange(len(wu)), (rng.random((len(wu), 1)) > pw).sum(1).clip(max=9)]
    # clicks in time order, then each event's history is the user's latest 20 before it
    ru, rc = np.where(y == 1)
    all_u = np.r_[wu, user[ru]].astype(np.int64)
    all_t = np.r_[wt, d.ts[ru]].astype(np.int64)
    all_i = np.r_[wi, cand[ru, rc]]
    o = np.lexsort((all_t, all_u))
    all_u, all_t, all_i = all_u[o], all_t[o], all_i[o]
    comp = all_u * (1 << 32) + all_t
    hist, hts = np.full((d.n, 20), -1, np.int32), np.zeros((d.n, 20), np.int32)
    ue, te = d.user.astype(np.int64), d.ts.astype(np.int64).copy()
    te[n:] = np.minimum(
        te[n:], day_ts(27)
    )  # a query sees nothing after the cutoff, and here no click exists after it either
    stop = np.searchsorted(comp, ue * (1 << 32) + te, "left")
    start = np.searchsorted(comp, ue * (1 << 32), "left")
    L = np.minimum(stop - start, 20)
    for j in range(20):
        m = L > j
        pos = stop[m] - L[m] + j
        hist[m, j], hts[m, j] = all_i[pos], all_t[pos]
    return dict(
        n_train=n,
        user=d.user,
        ts=d.ts,
        bucket=d.bucket,
        day_type=d.day_type,
        cand=d.cand,
        y=np.r_[y, d.y[n:]],
        hist_item=hist,
        hist_ts=hts,
        hist_len=L.astype(np.int8),
        X=X,
        Y=Y,
        kappa=kappa,
        c0=c0,
        cu=cu,
    )


def isolate(name):
    """Point every disk cache at a twin's own folder, so its counters and states cannot mix with the real ones."""
    from . import features, pipe

    p = CACHE / f"twin_{name}"
    (p / "states").mkdir(parents=True, exist_ok=True)
    features.CACHE = pipe.CACHE = p
    return p


class Twin:
    def __init__(self, path):
        z = np.load(path)
        self.n_train = int(z["n_train"])
        for k in ["user", "ts", "bucket", "day_type", "cand", "y", "hist_item", "hist_ts", "hist_len", "X", "Y"]:
            setattr(self, k, z[k])
        self.n = len(self.ts)
        self.kappa, self.c0 = float(z["kappa"]), float(z["c0"])
        self.cu = z["cu"] if "cu" in z else np.zeros(self.X.shape[0], np.float32)


def path(kappa, seed, tag=""):
    return CACHE / f"synth_k{kappa}_s{seed}{tag}.npz"


def main():
    kappa, seed = float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    tag = sys.argv[4] if len(sys.argv) > 4 else "_g"
    z = build(D(), kappa, seed, tau=tau)
    np.savez(path(kappa, seed, tag), **z)
    print(
        "c0",
        z["c0"],
        "click rate",
        (z["y"][: z["n_train"]].sum(1) > 0).mean(),
        "median hist",
        np.median(z["hist_len"][: z["n_train"]]),
    )


if __name__ == "__main__":
    main()
