"""Joint posterior of unit user and item vectors under the full slate likelihood, on Metal.

Labelled events go through the outside-option choice likelihood, warm-up clicks through a density term, so rejected offers
count. Users and items are drawn in turn: a von Mises-Fisher proposal from the click evidence, then a Metropolis correction by
the slate likelihood. Point estimates of this likelihood memorise (attempts.md), so scoring averages the choice probability
over samples. A chain at T sees events strictly before T and, for query users, their supplied histories.
"""

import math
import time

import numpy as np
import torch

from . import states
from .lib import NI, NU, T0, day_ts, norm_incidence, rsvd

P = 8


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class Vmf:
    """Exact von Mises-Fisher draws by Ulrich's rejection method. Beta draws come from a pool, the rest runs on the device."""

    def __init__(self, dev, seed=0, pool=1 << 22):
        g = torch.Generator().manual_seed(seed)
        a, c = (
            torch._standard_gamma(torch.full((pool,), (P - 1) / 2.0), generator=g),
            torch._standard_gamma(torch.full((pool,), (P - 1) / 2.0), generator=g),
        )
        self.beta, self.dev, self.pool = (a / (a + c)).to(dev), dev, pool

    def __call__(self, eta):
        k = eta.norm(dim=-1)
        mu = eta / k.clamp(min=1e-9)[:, None]
        m = P - 1
        root = (4 * k * k + m * m).sqrt()
        b = m / (2 * k + root)
        x0 = (1 - b) / (1 + b)
        c = k * x0 + m * torch.log((1 - x0 * x0).clamp(min=1e-12))
        w = torch.zeros_like(k)
        todo = torch.arange(len(k), device=self.dev)
        while len(todo):
            z = self.beta[torch.randint(self.pool, (len(todo),), device=self.dev)]
            bb, xx = b[todo], x0[todo]
            om = (1 - (1 + bb) * z) / (1 - (1 - bb) * z)
            ok = (
                k[todo] * om + m * torch.log((1 - xx * om).clamp(min=1e-12)) - c[todo]
                >= torch.rand(len(todo), device=self.dev).log()
            )
            w[todo[ok]] = om[ok]
            todo = todo[~ok]
        v = torch.randn(len(k), P, device=self.dev)
        v = unit(v - (v * mu).sum(1, keepdim=True) * mu)
        return w[:, None] * mu + (1 - w * w).clamp(min=0).sqrt()[:, None] * v


def add(n, idx, val):
    """Sum of val by idx, into n rows."""
    return torch.zeros((n, *val.shape[1:]), device=val.device, dtype=val.dtype).index_add_(0, idx, val)


class Chain:
    """off: (E,10) fixed utility terms of the labelled events. cat: item categories, each with a fitted intercept.

    kp, kd: strength of the labelled and warm-up click evidence in the proposal. With link=0, kd is also the warm-up density
    term. link=1 makes the warm-up term the chance a click lands on this item against a mean-field slate, which saturates as
    MNL does. tau: prior sd of the user intercept. item_sd: prior sd of an item intercept. em: fit kappa, beta scale and
    category intercepts during burn-in, with intercept variances held fixed, since EM recovers very slowly from the first
    M-steps. theta (NU,K): users' mixes, for the warm-up clicks' exposure mean and offset. qev: query events, whose histories
    count as warm-up evidence. A user with no event before T has no other evidence, and the histories are label-free.
    ufeat (NU,F), ifeat (NI,F): covariates of the prior means of the user and item intercepts, fitted during burn-in.
    """

    def __init__(
        self,
        d,
        T,
        X0,
        Y0,
        kappa=6.3,
        c0=-3.4,
        tau=0.0,
        off=None,
        cat=None,
        kp=4.0,
        kd=4.0,
        link=0,
        item_sd=0.0,
        em=0,
        theta=None,
        qev=None,
        coin=1.0,
        dev="mps",
        seed=0,
        ufeat=None,
        ifeat=None,
        bt0=None,
        rfix=0.0,
        ws=1.0,
        warm_off=None,
        leaves=None,
        pois=0,
    ):
        torch.manual_seed(seed)
        self.dev, self.tau, self.item_sd, self.em, self.link, self.coin = (
            torch.device(dev),
            tau,
            item_sd,
            em,
            link,
            coin,
        )
        self.k, self.c0, self.kp, self.kd = kappa, c0, kp, kd
        self.vmf = Vmf(self.dev, seed)
        ten = lambda a, t=torch.float32: torch.as_tensor(a, dtype=t, device=self.dev)
        n = d.n_train
        ev = np.flatnonzero(d.ts[:n] < T)
        self.E = len(ev)
        self.eu, self.c = ten(d.user[ev].astype(np.int64), torch.long), ten(d.cand[ev].astype(np.int64), torch.long)
        yy = d.y[ev]
        self.has = ten(yy.any(1))
        self.cl = ten(yy.argmax(1).astype(np.int64), torch.long)[:, None]
        self.rows = self.c.ravel()
        self.isc = ten(yy.astype(np.float32))
        cu, cc = np.nonzero(yy)
        self.lu, self.li = (
            ten(d.user[ev[cu]].astype(np.int64), torch.long),
            ten(d.cand[ev[cu], cc].astype(np.int64), torch.long),
        )
        self.wl_i = add(NI, self.li, torch.ones(len(self.li), device=self.dev))[:, None] / len(self.li)
        self.wl_u = add(NU, self.lu, torch.ones(len(self.lu), device=self.dev))[:, None] / len(self.lu)
        pu, pi, pt, w = states.positives(d, T, query=qev)
        m = pt < T0
        self.wu, self.wi, self.ww = (
            ten(pu[m].astype(np.int64), torch.long),
            ten(pi[m].astype(np.int64), torch.long),
            ten(w[m]) * ws,
        )
        self.rfix, self.pois = rfix, pois
        self.nw_u, self.nw_i = add(NU, self.wu, self.ww), add(NI, self.wi, self.ww)
        self.ww_i, self.ww_u = (self.nw_i / self.nw_i.sum())[:, None], (self.nw_u / self.nw_u.sum())[:, None]
        off = np.zeros((self.E, 10, 1), np.float32) if off is None else off
        self.off = ten(
            off if off.ndim == 3 else off[..., None]
        )  # (E,10,F): one channel per fixed term, each with its own fitted scale
        self.bt = torch.ones(self.off.shape[-1], device=self.dev) if bt0 is None else ten(bt0[: self.off.shape[-1]])
        self.cat = None if cat is None else ten(cat.astype(np.int64), torch.long)
        self.K = 0 if cat is None else int(cat.max()) + 1
        self.ci = (
            None if cat is None else cat[d.cand[ev]].ravel()
        )  # small bin counts are slow on Metal, so keep these on the CPU
        self.a = torch.zeros(max(self.K, 1), device=self.dev)
        self.leaves = (
            None if leaves is None else ten(leaves.astype(np.int64), torch.long)
        )  # one intercept per leaf, nested in its category
        self.li_ = None if leaves is None else leaves[d.cand[ev]].ravel()
        self.nl = 0 if leaves is None else int(leaves.max()) + 1
        self.al = torch.zeros(max(self.nl, 1), device=self.dev)
        self.theta = None if theta is None else ten(theta)
        self.owi, self.owm = None, None
        if theta is not None and link:
            self.owi = ten(np.log(np.maximum(theta[pu[m], cat[pi[m]]], 1e-6)))
        if warm_off is not None:  # (W,F): labelled channels at each warm-up pair, so both terms read the same utility
            self.owm = ten(warm_off(pu[m], pi[m], pt[m]))
        self.X, self.Y = unit(ten(X0)), unit(ten(Y0))
        self.b, self.d = torch.zeros(NU, device=self.dev), torch.zeros(NI, device=self.dev)
        self.mb, self.md = torch.zeros(NU, device=self.dev), torch.zeros(NI, device=self.dev)
        self.ufeat, self.ifeat, self.seen = ufeat, ifeat, np.bincount(d.user[ev], minlength=NU) > 0
        self.sb, self.sd = torch.full((NU,), 0.2, device=self.dev), torch.full((NI,), 0.1, device=self.dev)
        self.offers = add(NI, self.rows, torch.ones_like(self.rows, dtype=torch.float32))
        self.ctr_mass = None
        self.refix()
        self.Z = self.dot(self.X, self.Y)
        self.rbar, self.ks, self.au, self.ai, self.sweeps, self.pclick = 0.1, 3.6, 0.0, 0.0, 0, 0.4
        self.rq = torch.full((16,), 0.9, device=self.dev)

    def refix(self):
        """The part of each row's utility that neither the vectors nor the user intercept touch."""
        fix = self.c0 + self.off @ self.bt + self.d[self.c]
        self.fix = (
            fix + (self.a[self.cat[self.c]] if self.K else 0.0) + (self.al[self.leaves[self.c]] if self.nl else 0.0)
        )

    def dot(self, X, Y):
        return (X[self.eu].unsqueeze(1) * Y[self.c]).sum(2)

    def util(self, Z):
        return self.k * Z + self.fix + self.b[self.eu].unsqueeze(1)

    def ll(self, U):
        return self.has * U.gather(1, self.cl)[:, 0] - torch.log1p(U.exp().sum(1))

    def means(self):
        """(item mean under a user's exposure, user mean under an item's exposure), the partition term of the warm-up density."""
        if self.theta is None or not self.K:
            yb, xb = (self.ww_i * self.Y).sum(0), (self.ww_u * self.X).sum(0)
            return yb, xb
        Yc = (
            add(self.K, self.cat, self.offers[:, None] * self.Y)
            / add(self.K, self.cat, self.offers).clamp(min=1)[:, None]
        )
        Xc = (self.theta.T @ (self.nw_u[:, None] * self.X)) / (self.theta.T @ self.nw_u)[:, None]
        return self.theta @ Yc, Xc[self.cat]

    def warm(self, x, y):
        """Warm-up term of each pair: the chance a click lands on this item. link 1 uses one rival mass, link 2 averages over the
        rival masses of the labelled slates, which is the exact marginal over an unseen slate up to independence of its rows."""
        U = self.k * (x * y).sum(1) + self.fw + self.b[self.wu]
        eb = self.b[self.wu].exp()
        if self.link == 1:
            return self.ww * (U - torch.logaddexp(torch.log1p(self.rw * eb), U))
        den = 1 + self.rq.unsqueeze(0) * eb.unsqueeze(1) + U.exp().unsqueeze(1)
        return self.ww * (U + (1 / den).mean(1).log())

    def warm_fix(self):
        o = (
            self.c0
            + self.d[self.wi]
            + (self.a[self.cat[self.wi]] if self.K else 0.0)
            + (self.al[self.leaves[self.wi]] if self.nl else 0.0)
        )
        self.fw = (
            o + self.owm @ self.bt if self.owm is not None else o + self.bt[0] * self.owi if self.owi is not None else o
        )
        self.rw = 9 * self.rbar

    @torch.no_grad()
    def sweep(self):
        dev, X, Y = self.dev, self.X, self.Y
        ybar, xbar = self.means()
        if self.link:
            self.warm_fix()
        U = self.util(self.Z)
        l0 = self.ll(U)
        el = self.kp * add(NU, self.lu, Y[self.li] - (self.wl_i * Y).sum(0))
        ewp = self.ww[:, None] * (Y[self.wi] - (ybar[self.wu] if ybar.dim() > 1 else ybar))
        ew = self.kd * add(NU, self.wu, ewp)
        Xn = self.vmf(el + ew)
        Zn = self.dot(Xn, Y)
        dx = Xn - X
        du = add(NU, self.eu, self.ll(self.util(Zn)) - l0) - (dx * el).sum(1)
        if self.link:
            mean_term = (self.nw_u[:, None] * (ybar if ybar.dim() > 1 else ybar[None])).mul(dx).sum(1)
            du = (
                du
                + add(NU, self.wu, self.warm(Xn[self.wu], Y[self.wi]) - self.warm(X[self.wu], Y[self.wi]))
                - self.ks * mean_term
                - (dx * ew).sum(1)
            )
        a = torch.rand(NU, device=dev).log() < du
        self.X = X = torch.where(a[:, None], Xn, X)
        self.Z = Z = torch.where(a[self.eu].unsqueeze(1), Zn, self.Z)
        self.au = float(a.float().mean())
        U = self.util(Z)
        S = U.exp().sum(1)
        if self.tau > 0:
            nb = self.b + self.sb * torch.randn(NU, device=dev)
            de = (nb - self.b)[self.eu]
            db = add(NU, self.eu, self.has * de - torch.log1p(S * de.exp()) + torch.log1p(S))
            db = db - 0.5 * ((nb - self.mb) ** 2 - (self.b - self.mb) ** 2) / self.tau**2
            ab = torch.rand(NU, device=dev).log() < db
            self.b = torch.where(ab, nb, self.b)
            self.sb = (self.sb * torch.where(ab, 1.02, 0.99)).clamp(0.02, 2.0)
            U = self.util(Z)
            S = U.exp().sum(1)
        if self.link:
            self.warm_fix()
        el = self.kp * add(NI, self.li, X[self.lu] - (self.wl_u * X).sum(0))
        ewp = self.ww[:, None] * (X[self.wu] - (xbar[self.wi] if xbar.dim() > 1 else xbar))
        ew = self.kd * add(NI, self.wi, ewp)
        Yn = self.vmf(el + ew)
        Zn = self.dot(X, Yn)
        dU = self.k * (Zn - Z)
        eU, Dn = U.exp(), (1 + S)[:, None]
        Dp = (Dn - eU + (U + dU).exp()).clamp(min=1e-6)
        dy = Yn - Y
        di = add(NI, self.rows, (self.isc * dU - Dp.log() + Dn.log()).ravel()) - (dy * el).sum(1)
        if self.link:
            mean_term = (self.nw_i[:, None] * (xbar if xbar.dim() > 1 else xbar[None])).mul(dy).sum(1)
            di = (
                di
                + add(NI, self.wi, self.warm(X[self.wu], Yn[self.wi]) - self.warm(X[self.wu], Y[self.wi]))
                - self.ks * mean_term
                - (dy * ew).sum(1)
            )
        a = torch.rand(NI, device=dev).log() < di
        if self.coin < 1:
            a = a & (torch.rand(NI, device=dev) < self.coin)
        self.Y = Y = torch.where(a[:, None], Yn, Y)
        self.Z = Z = torch.where(a[self.c].unsqueeze(-1), Zn.unsqueeze(-1), Z.unsqueeze(-1)).squeeze(-1)
        self.ai = float(a.float().mean()) / self.coin
        if self.item_sd > 0:
            U = self.util(Z)
            eU, Dn = U.exp(), 1 + U.exp().sum(1, keepdim=True)
            nd = self.d + self.sd * torch.randn(NI, device=dev)
            de = nd - self.d
            dU = de[self.c]
            Dp = (Dn - eU + eU * dU.exp()).clamp(min=1e-6)
            dd = add(NI, self.rows, (self.isc * dU - Dp.log() + Dn.log()).ravel())
            mw = self.offers
            if (
                self.pois
            ):  # expected warm-up clicks from offers and the item's group rate, so the count doesn't re-learn the group
                mw = mw * (self.a[self.cat] + (self.al[self.leaves] if self.nl else 0.0)).exp()
            mw = self.nw_i.sum() * mw / mw.sum()
            dd = dd + self.nw_i * de - mw * (nd.exp() - self.d.exp())
            dd = dd - 0.5 * ((nd - self.md) ** 2 - (self.d - self.md) ** 2) / self.item_sd**2
            ad = (torch.rand(NI, device=dev).log() < dd) & (torch.rand(NI, device=dev) < 0.5)
            self.d = torch.where(ad, nd, self.d)
            self.sd = (self.sd * torch.where(ad, 1.04, 0.98)).clamp(0.01, 1.0)
            self.refix()
        self.sweeps += 1

    @torch.no_grad()
    def fit_hyper(self, rate=0.5):
        """One damped Newton step of kappa and the offset scale on the current sample, then category intercepts by ratio."""
        U = self.util(self.Z)
        e = U.exp()
        p = e / (1 + e.sum(1, keepdim=True))
        pc = p.gather(1, self.cl)[:, 0]
        self.pclick = float((self.has * pc).sum() / self.has.sum())
        if self.rfix:
            self.rbar = self.rfix / 9
        elif self.link == 1:
            self.fit_rival()
        elif self.link == 2:
            R = (0.9 * (e.sum(1) * (-self.b[self.eu]).exp())).cpu().numpy()
            self.rq = torch.as_tensor(np.quantile(R, (np.arange(16) + 0.5) / 16), dtype=torch.float32, device=self.dev)
            self.rbar = float(self.rq.mean()) / 9
        self.ks = P * float(((self.has > 0) * self.Z.gather(1, self.cl)[:, 0]).sum() / self.has.sum())
        if self.em:
            step = lambda f: (
                lambda m: (
                    float((self.has * f.gather(1, self.cl)[:, 0] - m).sum()) / float(((p * f * f).sum(1) - m * m).sum())
                )
            )((p * f).sum(1))
            self.k += rate * step(self.Z)
            for j in range(self.off.shape[-1]):
                f = self.off[..., j]
                if f.abs().sum() > 0:
                    self.bt[j] += rate * step(f)
        if self.ufeat is not None and self.tau > 0:
            self.mb = self.prior_mean(self.b, self.ufeat, self.seen)
        if self.ifeat is not None and self.item_sd > 0:
            self.md = self.prior_mean(self.d, self.ifeat)
        if self.K:
            obs = np.bincount(self.ci, weights=self.isc.ravel().cpu().numpy(), minlength=self.K)
            exp = np.bincount(self.ci, weights=p.ravel().cpu().numpy(), minlength=self.K)
            self.a = self.a + 0.7 * torch.as_tensor(np.log(obs / exp), dtype=torch.float32, device=self.dev)
            if self.nl:
                obs = np.bincount(self.li_, weights=self.isc.ravel().cpu().numpy(), minlength=self.nl) + 5
                exp = np.bincount(self.li_, weights=p.ravel().cpu().numpy(), minlength=self.nl) + 5
                self.al = self.al + 0.7 * torch.as_tensor(np.log(obs / exp), dtype=torch.float32, device=self.dev)
        else:
            self.c0 = self.c0 + 0.7 * math.log(float(self.isc.sum()) / float(p.sum()))
        self.refix()

    def fit_rival(self):
        """The rival mass at which warm-up clicks have the choice probability labelled clicks have. A mean over rows would overstate
        it: one strong rival dominates that mean, and clicks fall in slates without one."""
        self.warm_fix()
        U = self.k * (self.X[self.wu] * self.Y[self.wi]).sum(1) + self.fw + self.b[self.wu]
        eb, eU, n = self.b[self.wu].exp(), U.exp(), self.ww.sum()
        lo, hi = -6.0, 6.0
        for _ in range(14):
            r = (lo + hi) / 2
            q = float((self.ww * eU / (1 + math.exp(r) * eb + eU)).sum() / n)
            lo, hi = (r, hi) if q > self.pclick else (lo, r)
        self.rbar = math.exp((lo + hi) / 2) / 9

    def prior_mean(self, v, f, on=None):
        """Least-squares prior mean of this sample's intercepts on covariates, fitted on entities that have evidence: an entity with
        none would only echo the mean it was given."""
        x = v.cpu().numpy().astype(np.float64)
        fit = (f, x) if on is None else (f[on], x[on])
        return torch.as_tensor(f @ np.linalg.lstsq(*fit, rcond=None)[0], dtype=torch.float32, device=self.dev)

    def event_ll(self):
        return float(self.ll(self.util(self.Z)).mean())

    def burn_clicks(self, n, seed=0):
        """Click-only Gibbs sweeps: a label-only start for the slate sweeps. Whitening stops the collapse onto the degree direction."""
        X, Y = self.X, self.Y
        has = self.offers > 0
        pair = (
            torch.cat([self.wu, self.lu]),
            torch.cat([self.wi, self.li]),
            torch.cat([self.ww, torch.ones_like(self.li, dtype=torch.float32)]),
        )
        nu, ni = add(NU, pair[0], pair[2]), add(NI, pair[1], pair[2])
        for _ in range(n):
            ybar = (Y * (ni / ni.sum())[:, None]).sum(0)
            X = self.vmf(self.kp * add(NU, pair[0], pair[2][:, None] * (Y[pair[1]] - ybar)))
            xbar = (X * (nu / nu.sum())[:, None]).sum(0)
            Y = self.vmf(self.kp * add(NI, pair[1], pair[2][:, None] * (X[pair[0]] - xbar)))
            c = Y[has] - Y[has].mean(0)
            lam, Q = torch.linalg.eigh((c.T @ c * P / len(c)).cpu().double())
            Y = unit(Y @ ((Q * lam**-0.5) @ Q.T).float().to(self.dev))
        self.X, self.Y = X, Y
        self.Z = self.dot(X, Y)


class Scorer:
    """Running posterior predictive of rows: mean probability, mean and variance of the dot, mean user intercept.

    A row whose user has earlier rows in the set has seen outcomes the chain has not, so each sample is reweighted by how well it
    explains those earlier outcomes. A chain at T can then score a whole day, and a first event after T has weight one.
    """

    def __init__(self, ch, d, rows, off=None):
        ten = lambda a, t=torch.float32: torch.as_tensor(a, dtype=t, device=ch.dev)
        self.ch, self.rows = ch, rows
        self.u, self.c = ten(d.user[rows].astype(np.int64), torch.long), ten(d.cand[rows].astype(np.int64), torch.long)
        off = np.zeros((*d.cand[rows].shape, 1), np.float32) if off is None else off
        self.off = ten(off if off.ndim == 3 else off[..., None])
        yy = np.maximum(d.y[rows], 0)
        self.has, self.cl = ten(yy.any(1)), ten(yy.argmax(1).astype(np.int64), torch.long)[:, None]
        us = d.user[rows]
        self.o = np.lexsort((np.arange(len(rows)), us))
        s = us[self.o]
        first = np.r_[True, s[1:] != s[:-1]]
        self.gs = np.maximum.accumulate(np.where(first, np.arange(len(s)), 0))
        self.multi = not first.all()
        self.reset()

    def reset(self):
        n, inf = len(self.rows), float("-inf")
        new = lambda *sh: torch.full(sh, inf, device=self.ch.dev)
        self.lp, self.lt, self.lt2, self.lb, self.lw, self.n = new(n, 10), new(n, 10), new(n, 10), new(n), new(n), 0

    @torch.no_grad()
    def add(self):
        ch = self.ch
        t = (ch.X[self.u].unsqueeze(1) * ch.Y[self.c]).sum(2)
        b = ch.b[self.u]
        z = ch.k * t + ch.c0 + self.off @ ch.bt + ch.d[self.c] + b.unsqueeze(1)
        if ch.K:
            z = z + ch.a[ch.cat[self.c]]
        if ch.nl:
            z = z + ch.al[ch.leaves[self.c]]
        den = torch.log1p(z.exp().sum(1))
        w = torch.zeros(len(self.rows), device=ch.dev)
        if self.multi:
            ll = (self.has * z.gather(1, self.cl)[:, 0] - den).cpu().numpy().astype(np.float64)
            cs = np.r_[0, np.cumsum(ll[self.o])]
            out = np.zeros(len(ll))
            out[self.o] = cs[:-1] - cs[self.gs]
            w = torch.as_tensor(out.astype(np.float32), device=ch.dev)
        lt = torch.log1p(t.clamp(min=-0.9999))
        self.lp = torch.logaddexp(self.lp, w[:, None] + z - den[:, None])
        self.lt = torch.logaddexp(self.lt, w[:, None] + lt)
        self.lt2 = torch.logaddexp(self.lt2, w[:, None] + 2 * lt)
        self.lb = torch.logaddexp(self.lb, w + torch.log(b + 20))
        self.lw = torch.logaddexp(self.lw, w)
        self.n += 1

    def out(self):
        lw = self.lw.unsqueeze(1)
        p = (self.lp - lw).exp().cpu().numpy()
        t1, t2 = (self.lt - lw).exp(), (self.lt2 - lw).exp()
        return dict(
            p=p,
            t=(t1 - 1).cpu().numpy(),
            v=(t2 - t1 * t1).cpu().numpy(),
            b=((self.lb - self.lw).exp() - 20).cpu().numpy(),
            n=self.n,
            rows=self.rows,
        )


def spectral(d, T, seed=0, qev=None):
    """Item start from the click graph: directions two to nine of the normalised incidence matrix."""
    pu, pi, _, _ = states.positives(d, T, query=qev)
    B, _, _ = norm_incidence(pu, pi)
    Vt = rsvd(B, P + 1, seed=seed)[2]
    Y = Vt.T[:, 1 : P + 1]
    return Y + 1e-3 * np.random.default_rng(seed).standard_normal(Y.shape)


def run(ch, scorer, burn, samples, thin=1, log=print, tick=None, hyper_every=10):
    """Burn in, fitting hyperparameters, then average the predictive. tick(n) reports progress along the way."""
    t0 = time.time()
    for i in range(burn):
        if i % hyper_every == 0:
            ch.fit_hyper(0.5 if i < burn * 0.6 else 0.2)
        ch.sweep()
        if i % 100 == 0:
            log(
                f" burn {i} accept {ch.au:.3f} {ch.ai:.3f} kappa {ch.k:.3f} bt {ch.bt.cpu().numpy().round(3)} tau {ch.tau:.3f} isd {ch.item_sd:.3f} "
                f"ks {ch.ks:.2f} pc {ch.pclick:.3f} rival {9 * ch.rbar:.2f} ll {ch.event_ll():.4f} {(time.time() - t0) / (i + 1):.3f}s/sweep"
            )
    ch.fit_hyper(0.2)
    for s in range(1, samples + 1):
        for _ in range(thin):
            ch.sweep()
        scorer.add()
        if tick and s % tick[0] == 0:
            tick[1](s)
    return scorer.out()


def mix_terms(d, cut, ev, rows, mode="causal", beta=0.5, A0=17.6):
    """Offsets of chain events and of scored rows, and users' pooled mixes. causal: offers before the row plus its own slate and
    its history's clicks, so no term reads a later offer. global: one mix per user from all offers before the cut."""
    from . import groups

    lab = groups.categories(d, cut).astype(np.int64)
    H = np.zeros((NU, 10))
    np.add.at(H, (np.repeat(d.user[ev].astype(np.int64), 10), lab[d.cand[ev]].ravel()), 1)
    g = H.sum(0) / H.sum()
    th = (H + A0 * g) / (H.sum(1, keepdims=True) + A0)
    if mode == "global":
        own = np.zeros((len(rows), 10))
        np.add.at(own, (np.repeat(np.arange(len(rows)), 10), lab[d.cand[rows]].ravel()), 1)
        Hq = H[d.user[rows].astype(np.int64)] + own
        thq = (Hq + A0 * g) / (Hq.sum(1, keepdims=True) + A0)
        off_e = beta * np.log(np.take_along_axis(th[d.user[ev].astype(np.int64)], lab[d.cand[ev]], 1))
        off_r = beta * np.log(np.take_along_axis(thq, lab[d.cand[rows]], 1))
    else:
        m = groups.Mix(d, cut)
        off_e = m.scores(ev, m.clicks(cut, ev, d.hist_item[ev]), 2.0, beta)[0]
        off_r = m.scores(rows, m.clicks(cut, rows, d.hist_item[rows]), 2.0, beta)[0]
    return lab, th.astype(np.float32), off_e.astype(np.float32), off_r.astype(np.float32)


class Leaf:
    """Log share of each row's leaf inside its category: offers before the row, its own slate, and its history's clicks twice."""

    def __init__(self, d, cut, conc=8.0, at=None):
        from . import features, groups

        self.d, self.conc, at = d, conc, at or cut
        self.leaf, cat = groups.leaves(d, at).astype(np.int64), groups.categories(d, at).astype(np.int64)
        nl = int(self.leaf.max()) + 1
        self.own = np.zeros((d.n, nl))
        np.add.at(self.own, (np.repeat(np.arange(d.n), 10), self.leaf[d.cand].ravel()), 1)
        known = self.own * (np.arange(d.n) < d.n_train)[:, None]
        self.past = features.prior_sum(d.user.astype(np.int64), d.ts.astype(np.int64), known)
        before = d.ts < day_ts(min(cut, 27))
        tot = known[before].sum(0) + 1
        self.lc = np.zeros(nl, np.int64)
        np.add.at(self.lc, self.leaf, 0)
        self.lc[self.leaf] = cat
        self.g = tot / np.bincount(self.lc, weights=tot, minlength=10)[self.lc]

    def __call__(self, ev):
        d = self.d
        h = d.hist_item[ev]
        m = h >= 0
        C = np.zeros((len(ev), len(self.g)))
        np.add.at(C, (np.nonzero(m)[0], self.leaf[h[m]]), 1)
        c = self.past[ev] + self.own[ev] + 2 * C + self.conc * self.g
        tot = np.zeros((len(ev), 10))
        np.add.at(
            tot, (np.arange(len(ev))[:, None].repeat(c.shape[1], 1).ravel(), np.tile(self.lc, len(ev))), c.ravel()
        )
        lf = self.leaf[d.cand[ev]]
        return np.log(np.take_along_axis(c, lf, 1) / np.take_along_axis(tot, self.lc[lf], 1)).astype(np.float32)


def click_share(d, ev, leaf, rho=0.9, norm=True):
    """Share of the row's leaf in the user's recent clicks, newest first at weight 1, falling by rho a click as the offer echo does.
    norm=False gives the decayed count, an interest state that does not depend on history length."""
    h = d.hist_item[ev]
    m = h >= 0
    w = np.where(m, rho ** (m.sum(1, keepdims=True) - 1 - (np.cumsum(m, 1) - 1)), 0.0)
    C = np.zeros((len(ev), int(leaf.max()) + 1))
    np.add.at(C, (np.nonzero(m)[0], leaf[h[m]]), w[m])
    t = np.maximum(w.sum(1, keepdims=True), 1e-9) if norm else 10.0
    return (np.take_along_axis(C, leaf[d.cand[ev]], 1) / t).astype(np.float32)


def pair_share(u, i, t, leaf, rho=0.9, L=20):
    """click_share for a list of clicks: each click against the same user's earlier clicks in the list."""
    o = np.lexsort((t, u))
    us, ls = u[o], leaf[i[o]]
    num, den = np.zeros(len(o)), np.zeros(len(o))
    for k in range(1, L + 1):
        ok = us[k:] == us[:-k]
        w = rho ** (k - 1)
        num[k:] += w * (ok & (ls[k:] == ls[:-k]))
        den[k:] += w * ok
    out = np.zeros(len(o))
    out[o] = num / np.maximum(den, 1e-9)
    return out


def covariates(d, cut, ev, qev, cells=False):
    """Prior covariates of the intercepts, known at the cut. Users: warm-up clicks, events, and warm-up click rate, which over the
    event rate reads the user's click rate. Items: log offers. Their warm-up clicks are already in the likelihood."""
    T = day_ts(cut)
    pu, pi, pt, w = states.positives(d, T, query=qev)
    m = pt < T0
    nw = np.bincount(pu[m], weights=w[m], minlength=NU)
    ne = np.bincount(d.user[ev], minlength=NU).astype(np.float64)
    lo, hi = np.full(NU, 2e9), np.zeros(NU)
    np.minimum.at(lo, pu[m], pt[m])
    np.maximum.at(hi, pu[m], pt[m])
    span = np.maximum((hi - lo) / 86400.0, 0)
    rate = np.where((nw > 1) & (span > 0), np.log((nw - 1) / np.maximum(span, 0.05)), 0.0)
    uf = np.stack([np.ones(NU), np.log1p(nw), np.log1p(ne), rate, (nw > 1) * 1.0, (nw >= 20) * 1.0], 1)
    if cells:  # conditional mean of the intercept over warm-up clicks by events, no assumed form
        bin_w = np.digitize(nw, [1, 2, 4, 8, 16, 20])
        bin_e = np.digitize(np.maximum(ne, 1), [1, 2, 4, 8, 16, 32])  # users with no events share the one-event cells
        rb = np.digitize(rate, [-1.5, -0.5, 0.0, 0.5, 1.0])
        one = np.zeros((NU, 49 + 6))
        one[np.arange(NU), bin_w * 7 + bin_e] = 1
        one[np.arange(NU), 49 + rb] = 1
        uf = np.c_[one[:, one.sum(0) > 0], rate]
    off = np.bincount(d.cand[ev].ravel(), minlength=NI).astype(np.float64)
    fi = np.stack([np.ones(NI), np.log1p(off)], 1)
    return uf, fi


def rows_of(d, cut):
    """Rows a chain at this cut scores: that day's events, for training later stages, and the cut's query rows."""
    from .lib import first_events

    if cut >= 27:
        return np.arange(d.n_train, d.n), np.arange(d.n_train, d.n)
    q = first_events(d, day_ts(cut), day_ts(27))
    t = d.ts[: d.n_train]
    day = np.flatnonzero((t >= day_ts(cut)) & (t < day_ts(cut + 1)))
    return np.union1d(q, day), q


def fit(
    d,
    cut,
    seed=113,
    mix="causal",
    beta=0.5,
    burn=400,
    samples=2000,
    thin=1,
    click=100,
    qh=1,
    cov=0,
    leaf=0,
    poly=0,
    wo=0,
    lint=0,
    clk=0,
    wck=0,
    tick=None,
    log=print,
    **cfg,
):
    """One chain at this cut. Returns the scored rows' predictive and the chain."""
    T = day_ts(cut)
    ev = np.flatnonzero(d.ts[: d.n_train] < T)
    rows, q = rows_of(d, cut)
    qev = q if qh else None
    lab, th, off_e, off_r = (None, None, None, None) if mix == "none" else mix_terms(d, cut, ev, rows, mix, beta)
    uf, fi = covariates(d, cut, ev, qev, cov == 3) if cov else (None, None)
    fi = fi if cov == 2 else None
    if leaf:
        lf = Leaf(
            d, cut, at=27 if leaf > 1 else None
        )  # 2: the T27 taxonomy. Its finer leaves are an item property, not an outcome
        off_e, off_r = np.stack([off_e, lf(ev)], -1), np.stack([off_r, lf(rows)], -1)
        if poly:  # click rate isn't log-linear in the user's share: flat at low shares, roughly linear above
            mu = off_e.mean((0, 1))
            more = lambda o: np.stack(
                [(o[..., 0] - mu[0]) ** 2, (o[..., 0] - mu[0]) * (o[..., 1] - mu[1]), (o[..., 1] - mu[1]) ** 2], -1
            )
            lin_ = lambda o: np.stack([np.exp(2 * o[..., 0]), np.exp(2 * o[..., 0] + o[..., 1])], -1)  # the raw shares
            ext = more if poly == 1 else lin_
            off_e, off_r = np.concatenate([off_e, ext(off_e)], -1), np.concatenate([off_r, ext(off_r)], -1)
    wf = None
    if clk:
        ce, cr = click_share(d, ev, lf.leaf, norm=clk != 3), click_share(d, rows, lf.leaf, norm=clk != 3)
        cm = ce.mean()
        off_e, off_r = np.concatenate([off_e, ce[..., None] - cm], -1), np.concatenate([off_r, cr[..., None] - cm], -1)
        if clk == 2:  # share of the item's category, and whether the item itself is among recent clicks
            ce2, cr2 = click_share(d, ev, lab), click_share(d, rows, lab)
            off_e, off_r = (
                np.concatenate([off_e, ce2[..., None] - ce2.mean()], -1),
                np.concatenate([off_r, cr2[..., None] - ce2.mean()], -1),
            )
    if wo:
        hl = np.zeros((NU, len(lf.g)))
        np.add.at(hl, (np.repeat(d.user[ev].astype(np.int64), 10), lf.leaf[d.cand[ev]].ravel()), 1)
        hc = np.zeros((NU, 10))
        np.add.at(hc, (np.arange(NU)[:, None].repeat(hl.shape[1], 1).ravel(), np.tile(lf.lc, NU)), hl.ravel())

        def wf(u, i, t):
            a_ = beta * np.log(np.maximum(th[u, lab[i]], 1e-6))
            l_ = np.log((hl[u, lf.leaf[i]] + lf.conc * lf.g[lf.leaf[i]]) / (hc[u, lf.lc[lf.leaf[i]]] + lf.conc))
            o = np.stack([a_, l_], -1)
            o = np.concatenate([o, more(o)], -1) if poly == 1 else o
            if clk and wck:
                return np.concatenate([o, pair_share(u, i, t, lf.leaf)[:, None] - cm], -1).astype(np.float32)
            return (np.concatenate([o, np.zeros((len(o), 2 if clk == 2 else 1))], -1) if clk else o).astype(np.float32)

    ch = Chain(
        d,
        T,
        np.ones((NU, P)),
        spectral(d, T, seed, qev),
        off=off_e,
        cat=lab,
        theta=th,
        qev=qev,
        seed=seed,
        ufeat=uf,
        ifeat=fi,
        warm_off=wf,
        leaves=lf.leaf if lint else None,
        **cfg,
    )
    ch.burn_clicks(click)
    sc = Scorer(ch, d, rows, off_r)
    r = run(ch, sc, burn, samples, thin, log=log, tick=None if tick is None else (tick[0], lambda n: tick[1](n, sc)))
    return r, ch


class Inputs:
    """model.run_seed extra_head: chain outputs at each row's own cut, averaged over the named chains of this seed."""

    def __init__(self, names, seed):
        from .lib import RUNS

        self.paths, self.seed, self.mem = [RUNS / "slate" / n for n in names.split("+")], seed, {}

    def table(self, day):
        if day not in self.mem:
            zs = [np.load(p.parent / f"{p.name}_c{day}_s{self.seed}.npz") for p in self.paths]
            self.mem[day] = zs[0]["rows"], *[sum(z[k] for z in zs) / len(zs) for k in ("p", "t", "v", "b")]
        return self.mem[day]

    def p(self, day, ev):
        rows, p = self.table(day)[:2]
        return np.clip(p[np.searchsorted(rows, ev)], 1e-7, 1 - 1e-7)

    def __call__(self, day, ev, _=None):
        rows, p, t, v, b = self.table(day)
        i = np.searchsorted(rows, ev)
        assert (rows[i] == ev).all(), f"chain at {day} did not score these rows"
        p, t, v, b = np.clip(p[i], 1e-7, 1 - 1e-7), t[i], v[i], b[i]
        l = np.log(p / (1 - p))
        e = np.clip(p.sum(1, keepdims=True), 1e-4, 1 - 1e-4)
        rp = lambda a: np.repeat(a, 10, 1)
        return np.stack([l, l - l.mean(1, keepdims=True), t, v, rp(b[:, None]), rp(np.log(e / (1 - e)))], -1).astype(
            np.float32
        )


def member(d, cut, ev, seed, names):
    """Portfolio member: the chain's predictive alone, as log-odds."""
    p = Inputs(names, seed).p(cut, ev)
    return np.log(p / (1 - p))
