"""Exposure-corrected user affinity for item groups, the structure in the data that no preference table holds.

Offers show which groups a user is exposed to, clicks show which they take. A user's click share in a group means little
without their offer share in it, so the signal is the lift between the two. Offers are label-free and plentiful, and they are
stationary, so a user's offer share also corrects warm-up clicks, whose own offers were never logged.
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans

from . import extras, features, states
from .lib import CACHE, NI, NU, day_ts, norm_incidence, rsvd

K = 10  # offer factors have nine strong dimensions, and well-offered items sit on a ten-vertex simplex


def categories(d, cut, iters=12, alpha=2.0, evalrows=False):
    """Each item's category from label-free offers before the cut: spectral start, then hard EM on users' category mixes.

    An item's offers vote through the offered users' shares, so even a rarely offered item is placed by tens of informative
    votes. evalrows adds the evaluation slates, which are label-free too. Only use it at the final cut.
    """
    p = CACHE / f"cat{K}_c{cut}{'_e' if evalrows else ''}.npy"
    if p.exists():
        return np.load(p)
    eu, ei = states.exposures(d, day_ts(cut))
    if evalrows:
        e = np.arange(d.n_train, d.n)
        eu, ei = np.r_[eu, np.repeat(d.user[e], 10)], np.r_[ei, d.cand[e].ravel()]
    B, _, _ = norm_incidence(eu, ei)
    F = rsvd(B, K + 2, power=8)[2].T[:, 1:K]
    Y = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    off = np.bincount(ei, minlength=NI)
    km = KMeans(K, n_init=8, random_state=0).fit(Y[off >= 60])
    lab = (Y @ km.cluster_centers_.T).argmax(1)
    X = sp.csr_matrix((np.ones(len(eu), np.float32), (eu, ei)), shape=(NU, NI))
    X.sum_duplicates()
    for it in range(iters):
        Z = sp.csr_matrix((np.ones(NI, np.float32), (np.arange(NI), lab)), shape=(NI, K))
        H = np.asarray((X @ Z).todense(), np.float64)
        pi_ = (H.sum(0) + 1) / (H.sum() + K)
        th = (H + alpha * pi_) / (H.sum(1, keepdims=True) + alpha)
        size = np.bincount(lab, minlength=K) + 1.0
        ll = np.asarray(X.T @ (np.log(th / pi_) * 1.0), np.float64)  # per-item log likelihood ratio for each category
        new = (ll + np.log(size / size.sum())).argmax(1)
        changed = (new != lab).mean()
        lab = new
        if changed < 1e-4:
            break
    np.save(p, lab.astype(np.int16))
    return lab


class Lift:
    def __init__(self, d, cut, Ks=(16,), a=3.0, L=64, labs=None):
        self.d, self.cut, self.Ks, self.a, self.L = d, cut, Ks, a, L
        self.labs = labs if labs is not None else {K_: extras.exposure_clusters(d, cut, K_) for K_ in Ks}
        n = d.n_train
        self.tab = {}
        y = np.maximum(d.y, 0).astype(np.float64)
        known = np.r_[np.ones(n * 10), np.zeros((d.n - n) * 10)]
        t = np.repeat(d.ts, 10).astype(np.int64)
        vals = np.stack([known, y.ravel() * known], 1)
        for K, lab in self.labs.items():
            keys = np.repeat(d.user, 10).astype(np.int64) * K + lab[d.cand].ravel()
            self.tab[K] = features.prior_sum(keys, t, vals).reshape(-1, 10, 2)
        self.u = features.prior_sum(
            d.user.astype(np.int64), d.ts.astype(np.int64), np.stack([(d.y[:, 0] >= 0) * 10.0, y.sum(1)], 1)
        )

    def __call__(self, day, ev, _=None, hist=None):
        """(n,10,c) per K: lift of all known clicks against offer share, labelled lift, log offer share, log offers, log clicks."""
        d = self.d
        hist = states.known_history(d, ev, day_ts(day), self.L) if hist is None else hist
        m = hist >= 0
        n_h = m.sum(1, keepdims=True).astype(np.float64)
        out = []
        for K, lab in self.labs.items():
            N, C = self.tab[K][ev, :, 0], self.tab[K][ev, :, 1]
            Nu, Cu = self.u[ev, 0, None], self.u[ev, 1, None]
            share = (N + 1) / (Nu + K)
            H = np.zeros((len(ev), K))
            np.add.at(H, (np.nonzero(m)[0], lab[hist[m]]), 1)
            c_h = np.take_along_axis(H, lab[d.cand[ev]], 1)
            full = np.log((c_h + self.a * share) / (n_h * share + self.a * share))
            base = (Cu + 1) / (Nu + 16)
            lab_lift = np.log((C + 8 * base) / (N * base + 8 * base))
            out += [full, lab_lift, np.log(share), np.log1p(N), np.log1p(C)]
        return np.stack(out, -1).astype(np.float32)


class Cats:
    """Candidate inputs from the ten categories: one-hot, offer share, click lifts, counts, same-category rivals.

    rich adds the user's mix with the event's own slate counted, its concentration, the share relative to the slate, and the
    item's offer share within its category. All are label-free: only offers, which are far more plentiful than clicks.
    """

    def __init__(self, d, cut, onehot=True, rich=0):
        self.d, self.lab, self.onehot, self.rich = d, categories(d, cut), onehot, rich
        self.lift = Lift(d, cut, labs={K: self.lab})
        self.mix = Mix(d, cut) if rich > 1 else None
        if rich > 2:
            self.leaf = leaves(d, cut)
            self.nl = int(self.leaf.max()) + 1
            g = self.leaf[d.cand]
            self.lown = np.zeros((d.n, self.nl))
            np.add.at(self.lown, (np.repeat(np.arange(d.n), 10), g.ravel()), 1)
            known = self.lown * (np.arange(d.n) < d.n_train)[:, None]
            self.lpast = features.prior_sum(d.user.astype(np.int64), d.ts.astype(np.int64), known)
            before = d.ts < day_ts(min(cut, 27))
            self.lpi = (known[before].sum(0) + 1) / (known[before].sum() + self.nl)
        if rich:
            n = d.n_train
            g = self.lab[d.cand]
            self.own = np.zeros((d.n, K), np.float32)
            np.add.at(self.own, (np.repeat(np.arange(d.n), 10), g.ravel()), 1)
            known = self.own * (np.arange(d.n) < n)[:, None]
            self.past = features.prior_sum(d.user.astype(np.int64), d.ts.astype(np.int64), known.astype(np.float64))
            self.rows = np.cumsum(features.item_day_counts(d)[0], 0).astype(np.float64)

    def __call__(self, day, ev, _=None):
        d = self.d
        hist = states.known_history(d, ev, day_ts(day), self.lift.L)
        G = self.lift(day, ev, hist=hist)
        g = self.lab[d.cand[ev]]
        same = (g[:, :, None] == g[:, None, :]).sum(2)[..., None].astype(np.float32) - 1
        parts = [G, same]
        if self.onehot:
            parts.append(np.eye(K, dtype=np.float32)[g])
        if self.rich:
            c = self.past[ev] + self.own[ev]
            th = (c + 1) / (c.sum(1, keepdims=True) + K)
            ls = np.log(np.take_along_axis(th, g, 1))
            ent = -(th * np.log(th)).sum(1, keepdims=True)
            n_i = self.rows[max(day - 21, 0)]
            tot = np.bincount(self.lab, weights=n_i, minlength=K)
            pop = np.log((n_i[d.cand[ev]] + 1) / (tot[g] + K))
            rp = lambda a: np.repeat(a, 10, 1)
            parts.append(
                np.stack(
                    [
                        ls,
                        ls - ls.mean(1, keepdims=True),
                        rp(ent),
                        rp(th.max(1, keepdims=True)),
                        pop,
                        pop - pop.mean(1, keepdims=True),
                    ],
                    -1,
                ).astype(np.float32)
            )
        if self.mix is not None:
            a, b = self.mix.scores(ev, self.mix.clicks(day, ev, hist), 2.0, 0.55)
            parts.append(np.stack([a, a - a.mean(1, keepdims=True), b], -1).astype(np.float32))
        if self.rich > 2:
            m = hist >= 0
            C = np.zeros((len(ev), self.nl))
            np.add.at(C, (np.nonzero(m)[0], self.leaf[hist[m]]), 1)
            c = self.lpast[ev] + self.lown[ev] + 2 * C
            th = (c + 10 * self.lpi) / (c.sum(1, keepdims=True) + 10)
            ls = np.log(np.take_along_axis(th, self.leaf[d.cand[ev]], 1))
            parts.append(np.stack([ls, ls - ls.mean(1, keepdims=True)], -1).astype(np.float32))
        return np.concatenate(parts, -1)


def noise_tables(d, cut, power=0.75, alpha=2.0, tilt=1.0):
    """(theta (users,K), labels, item mass) for the exposure model: a category from the user's mix, then an item by offers.

    tilt=1+beta draws categories as clicks do, not as offers do, so the table need not absorb the category term at all.
    """
    lab = categories(d, cut)
    eu, ei = states.exposures(d, day_ts(cut))
    H = np.zeros((NU, K))
    np.add.at(H, (eu, lab[ei]), 1)
    pi_ = (H.sum(0) + 1) / (H.sum() + K)
    th = (H + alpha * pi_) / (H.sum(1, keepdims=True) + alpha)
    th = th**tilt
    th = th / th.sum(1, keepdims=True)
    return th.astype(np.float32), lab, np.bincount(ei, minlength=NI).astype(np.float64) ** power


def refs(d, day, ev, R=64, seed=104842, **kw):
    """R noise references per event from the same exposure model, so query inference matches how the table was trained."""
    th, lab, mass = noise_tables(d, day, **kw)
    rng = np.random.default_rng(seed)
    t = th[d.user[ev]]
    cum = np.cumsum(t, 1)
    cc = (rng.random((len(ev), R, 1)) > cum[:, None, :]).sum(2).clip(max=K - 1)
    out = np.zeros(cc.shape, np.int64)
    for c in range(K):
        items = np.where(lab == c)[0]
        pool = rng.choice(items, 2_000_000, p=mass[items] / mass[items].sum())
        m = cc == c
        out[m] = pool[rng.integers(len(pool), size=m.sum())]
    return out


def dirichlet(H, iters=200):
    """Dirichlet prior on users' mixes from category counts (users,K), by Minka's fixed point on the multinomial marginal."""
    H = H[H.sum(1) > 0]
    n = H.sum(1)
    a = (H.sum(0) + 1) / (H.sum() + K)
    for _ in range(iters):
        a0 = a.sum()
        a = a * (digamma(H + a) - digamma(a)).sum(0) / (digamma(n + a0) - digamma(a0)).sum()
    return a


class Mix:
    """Posterior of a user's category mix: Dirichlet prior, offers before the event, the event's slate, and known clicks.

    A click is stronger evidence than an offer, so it counts gamma times. The click law has a term beta log theta, so the score a
    row needs is log E[theta^beta], which is closed form for a Dirichlet and keeps a user's uncertainty in the estimate.
    """

    def __init__(self, d, cut, L=64):
        self.d, self.cut, self.L, self.lab = d, cut, L, categories(d, cut)
        g = self.lab[d.cand]
        self.own = np.stack([(g == k).sum(1) for k in range(K)], 1).astype(np.float64)
        known = self.own * (np.arange(d.n) < d.n_train)[:, None]
        self.past = features.prior_sum(d.user.astype(np.int64), d.ts.astype(np.int64), known)
        cut_rows = d.ts[: d.n_train] < day_ts(min(cut, 27))
        Hc = np.stack(
            [
                np.bincount(d.user[: d.n_train][cut_rows], weights=known[: d.n_train, k][cut_rows], minlength=NU)
                for k in range(K)
            ],
            1,
        )
        self.alpha = dirichlet(Hc)

    def clicks(self, day, ev, hist=None):
        hist = states.known_history(self.d, ev, day_ts(day), self.L) if hist is None else hist
        m = hist >= 0
        C = np.zeros((len(ev), K))
        np.add.at(C, (np.nonzero(m)[0], self.lab[hist[m]]), 1)
        return C

    def scores(self, ev, C, gamma=1.0, beta=0.55):
        """(n,10) log E[theta_c^beta] and E[log theta_c] for each candidate's category."""
        A = self.alpha + self.past[ev] + self.own[ev] + gamma * C
        g = self.lab[self.d.cand[ev]]
        a = np.take_along_axis(A, g, 1)
        t = A.sum(1, keepdims=True)
        return gammaln(a + beta) - gammaln(a) + gammaln(t) - gammaln(t + beta), digamma(a) - digamma(t)


def em(eu, ei, lab, K_, iters=12, alpha=2.0):
    """Hard EM on offers: each item goes where its offered users' mixes make it likeliest."""
    X = sp.csr_matrix((np.ones(len(eu), np.float32), (eu, ei)), shape=(NU, NI))
    X.sum_duplicates()
    for it in range(iters):
        Z = sp.csr_matrix((np.ones(NI, np.float32), (np.arange(NI), lab)), shape=(NI, K_))
        H = np.asarray((X @ Z).todense(), np.float64)
        pi_ = (H.sum(0) + 1) / (H.sum() + K_)
        th = (H + alpha * pi_) / (H.sum(1, keepdims=True) + alpha)
        size = np.bincount(lab, minlength=K_) + 1.0
        ll = np.asarray(X.T @ (np.log(th / pi_) * 1.0), np.float64)
        new = (ll + np.log(size / size.sum())).argmax(1)
        changed = (new != lab).mean()
        lab = new
        if changed < 1e-4:
            break
    return lab


def leaves(d, cut, kmax=6, dims=4, sil=0.3, log=None):
    """Second level of the taxonomy: split each category by its own offer factors where a clear split exists, then refine by EM."""
    p = CACHE / f"leaf_c{cut}.npy"
    if p.exists():
        return np.load(p)
    from sklearn.metrics import silhouette_score

    lab = categories(d, cut)
    eu, ei = states.exposures(d, day_ts(cut))
    off = np.bincount(ei, minlength=NI)
    out = np.zeros(NI, np.int64)
    n_leaf = 0
    for c in range(K):
        m = lab[ei] == c
        B, _, _ = norm_incidence(eu[m], ei[m])
        Vc = rsvd(B, dims + 3, power=8, seed=c)[2].T[:, 1 : dims + 1]
        items = np.where((lab == c) & (off >= max(50, np.quantile(off[lab == c], 0.3))))[0]
        Y = Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), 1e-12)
        best, score, cen = 1, sil, None
        sub = np.random.default_rng(c).choice(len(items), min(2500, len(items)), replace=False)
        for k in range(2, kmax + 1):
            km = KMeans(k, n_init=4, random_state=0).fit(Y[items])
            a = silhouette_score(Y[items][sub], km.labels_[sub])
            if a > score:
                best, score, cen = k, a, km.cluster_centers_
        here = np.where(lab == c)[0]
        out[here] = n_leaf + (0 if cen is None else (Y[here] @ cen.T).argmax(1))
        if log:
            log(f"  category {c}: items {len(here)}, {best} leaves, silhouette {score:.3f}")
        n_leaf += best
    out = em(eu, ei, out, n_leaf)
    np.save(p, out.astype(np.int16))
    return out
