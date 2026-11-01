"""Optional per-candidate tree inputs. Each block returns (n,10,c) floats and is causal at its `day` cutoff."""

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from . import states
from .lib import CACHE, NI, NU, day_ts, norm_incidence, rsvd


def rival(x):
    """Candidate value relative to the other nine in its slate."""
    r = (x.sum(1, keepdims=True) - x) / 9
    return x - r


def pooled_sim(d, ev, V):
    """max/top3/mean/std of normalised similarity between each candidate and individual history items."""
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    Vn = Vn.astype(np.float32)
    out = np.zeros((len(ev), 10, 5), np.float32)
    for i in range(0, len(ev), 20000):
        e = ev[i : i + 20000]
        h = d.hist_item[e]
        m = h >= 0
        S = np.einsum("nij,nkj->nik", Vn[d.cand[e]], Vn[np.maximum(h, 0)], optimize=True)
        S = np.where(m[:, None, :], S, -2.0)
        srt = np.sort(S, 2)[:, :, ::-1]
        has = m.sum(1, keepdims=True) > 0
        cnt = np.maximum(m.sum(1, keepdims=True), 1)
        mean = (np.where(m[:, None, :], S, 0).sum(2)) / cnt
        top3 = np.where(srt[:, :, :3] > -2, srt[:, :, :3], 0).sum(2) / np.minimum(cnt, 3)
        out[i : i + len(e), :, 0] = np.where(has, srt[:, :, 0], 0)
        out[i : i + len(e), :, 1] = top3
        out[i : i + len(e), :, 2] = mean
        out[i : i + len(e), :, 3] = (S > 0.5).sum(2)
        out[i : i + len(e), :, 4] = rival(np.where(has, srt[:, :, 0], 0))
    return out


class CoClick:
    """Item-item cosine over users, from positives known before the cutoff. High rank, no training."""

    def __init__(self, pu, pi):
        X = sp.csr_matrix((np.ones(len(pu), np.float32), (pu, pi)), shape=(NU, NI))
        X.sum_duplicates()
        X.data[:] = 1
        n = np.asarray(X.sum(0)).ravel()
        C = (X.T @ X).tocoo()
        keep = C.row != C.col
        self.key = C.row[keep].astype(np.int64) * NI + C.col[keep]
        val = C.data[keep] / np.sqrt(n[C.row[keep]] * n[C.col[keep]])
        o = np.argsort(self.key)
        self.key, self.val, self.cnt = self.key[o], val[o].astype(np.float32), C.data[keep][o].astype(np.float32)

    def lookup(self, a, b):
        k = a.astype(np.int64) * NI + b
        o = np.argsort(k)
        p = np.searchsorted(self.key, k[o])
        p = np.minimum(p, len(self.key) - 1)
        hit = self.key[p] == k[o]
        val, cnt = np.zeros(len(k), np.float32), np.zeros(len(k), np.float32)
        val[o[hit]] = self.val[p[hit]]
        cnt[o[hit]] = self.cnt[p[hit]]
        return val, cnt

    def features(self, d, ev):
        c, h = d.cand[ev], d.hist_item[ev]
        a = np.repeat(c, 20, axis=1).ravel()
        b = np.tile(h, (1, 10)).reshape(len(ev), 10, 20).reshape(len(ev), -1).ravel()
        m = b >= 0
        v, n = np.zeros(len(a), np.float32), np.zeros(len(a), np.float32)
        v[m], n[m] = self.lookup(a[m], b[m])
        v, n = v.reshape(len(ev), 10, 20), n.reshape(len(ev), 10, 20)
        s, mx = v.sum(2), v.max(2)
        return np.stack([s, mx, n.sum(2), (v > 0).sum(2), rival(s), rival(mx)], -1).astype(np.float32)


def exposure_clusters(d, cut, K=16, dims=20):
    """Soft item groups from label-free offers before the cutoff."""
    p = CACHE / f"clusters_c{cut}_K{K}.npy"
    if p.exists():
        return np.load(p)
    u, i = states.exposures(d, day_ts(cut))
    B, _, _ = norm_incidence(u, i)
    _, _, Vt = rsvd(B, dims + 1, power=6)
    X = Vt.T[:, 1:]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    lab = KMeans(K, n_init=4, random_state=0).fit_predict(X).astype(np.int32)
    np.save(p, lab)
    return lab


def cluster_features(d, ev, lab, K=16):
    c, h = d.cand[ev], d.hist_item[ev]
    n = len(ev)
    m = h >= 0
    H = np.zeros((n, K), np.float32)
    np.add.at(H, (np.nonzero(m)[0], lab[h[m]]), 1)
    share = H / np.maximum(m.sum(1, keepdims=True), 1)
    S = np.zeros((n, K), np.float32)
    np.add.at(S, (np.repeat(np.arange(n), 10), lab[c].ravel()), 1)
    L = lab[c]
    a, b = np.take_along_axis(share, L, 1), np.take_along_axis(S, L, 1)
    return np.stack([a, rival(a), b, b / 10 - a], -1).astype(np.float32)


def rate_features(d, ev):
    """User click-rate evidence from the history's time span: warm-up clicks lack denominators, but span gives clicks
    per day and the labelled window gives events per day, so their ratio estimates propensity."""
    from .features import pair_counts
    from .lib import DAY, T0

    h, ht, t = d.hist_item[ev], d.hist_ts[ev], d.ts[ev]
    n = (h >= 0).sum(1)
    first = np.where(n > 0, ht[:, 0], t)
    last = np.where(n > 0, ht.max(1), t)
    span = np.maximum((last - first) / 86400.0, 1 / 24)
    click_rate = np.where(n >= 2, (n - 1) / span, 0)
    to_query = np.maximum((t - first) / 86400.0, 1 / 24)
    rate2 = np.where(n >= 1, n / to_query, 0)
    _, _, ue, uc = pair_counts(d)
    elapsed = np.maximum((t - T0) / 86400.0, 0.25)
    ev_rate = (ue[ev] + 0.5) / elapsed
    ratio = np.log1p(click_rate) - np.log(ev_rate)
    ratio2 = np.log1p(rate2) - np.log(ev_rate)
    x = np.stack([np.log1p(click_rate), np.log1p(rate2), np.log(ev_rate), ratio, ratio2, np.log1p(span)], -1)
    return np.repeat(x[:, None, :], 10, 1).astype(np.float32)


def slate_focus(d, ev, V, lab=None, K=16):
    """Event-level slate concentration, in vector space and optionally in cluster space, plus each candidate's fit."""
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    C = Vn[d.cand[ev]].astype(np.float32)
    mean = C.mean(1)
    res = np.linalg.norm(mean, axis=1)
    fit = (C * mean[:, None, :]).sum(2) / np.maximum(res, 1e-6)[:, None]
    out = [np.repeat(res[:, None], 10, 1), fit, rival(fit)]
    if lab is not None:
        L = lab[d.cand[ev]]
        n = len(ev)
        S = np.zeros((n, K), np.float32)
        np.add.at(S, (np.repeat(np.arange(n), 10), L.ravel()), 1)
        p = S / 10
        ent = -(p * np.log(np.maximum(p, 1e-9))).sum(1)
        mx = S.max(1)
        out += [np.repeat(ent[:, None], 10, 1), np.repeat(mx[:, None], 10, 1), np.take_along_axis(S, L, 1)]
    return np.stack(out, -1).astype(np.float32)


def known_clicks(d):
    """All clicks of each (user,item) known strictly before the row's event, beyond the 20-item window. Cached."""
    p = CACHE / "known_clicks.npy"
    if p.exists():
        return np.load(p)
    pu, pi, pt, w = states.positives(d, day_ts(28))
    key = (pu.astype(np.int64) * NI + pi) << 31
    comp = key + pt.astype(np.int64)
    o = np.argsort(comp)
    comp, cw = comp[o], np.r_[0, np.cumsum(w[o].astype(np.int64))]
    rk = (np.repeat(d.user, 10).astype(np.int64) * NI + d.cand.ravel()) << 31
    t = np.repeat(d.ts, 10).astype(np.int64)
    out = (cw[np.searchsorted(comp, rk + t, "left")] - cw[np.searchsorted(comp, rk, "left")]).reshape(-1, 10)
    np.save(p, out.astype(np.int16))
    return out.astype(np.int16)
