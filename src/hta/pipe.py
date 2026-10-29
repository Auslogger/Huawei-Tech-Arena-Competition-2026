"""Cached state producers and per-event inputs. A state is fully keyed by its cutoff, seed and config."""
import json
import sys
import time

import numpy as np

from . import states
from .lib import CACHE, D, DAY, NI, NU, day_ts

DEFAULT = dict(k=64, l2=.02, wd=.3, epochs=80)


def key(T, seed, cfg):
    c = {**DEFAULT, **cfg}
    return f"nce_T{(T - day_ts(20)) // DAY + 20}_s{seed}_" + "_".join(f"{k}{c[k]}" for k in sorted(c))


def state(d, T, seed=113, **cfg):
    """(U, V, b, prop) fit on positives known strictly before T; cached on disk.

    nlu/nlv/ns request an exact constant-prior Newton refit warm-started from the Adam fit of the same config.
    """
    p = CACHE / "states" / f"{key(T, seed, cfg)}.npz"
    if p.exists():
        z = np.load(p)
        return z["U"], z["V"], z["b"], z["prop"]
    t = time.time()
    if "nlu" in cfg:
        from . import newton
        nw = {k: cfg[k] for k in ("nlu", "nlv", "ns")}
        U0, V0, b0, prop = state(d, T, seed, **{k: v for k, v in cfg.items() if k not in nw})
        pu, pi, _, w = states.positives(d, T)
        U, V, b = newton.refit(U0, V0, b0, prop, pu, pi, w, lam_u=nw["nlu"], lam_v=nw["nlv"], sweeps=nw["ns"])
        p.parent.mkdir(exist_ok=True)
        np.savez(p, U=U, V=V, b=b, prop=prop)
        print(f"  newton {p.stem} {time.time() - t:.0f}s", flush=True)
        return U, V, b, prop
    if cfg.get("slate"):  # each click trains one likelihood; see choice.py
        from . import choice
        c = {**DEFAULT, **cfg}
        U, V, b, prop = choice.fit(d, T, seed=seed, **{k: v for k, v in c.items() if k not in ("slate", "fism")})
        p.parent.mkdir(exist_ok=True)
        np.savez(p, U=U, V=V, b=b, prop=prop)
        print(f"  fit {p.stem} {time.time() - t:.0f}s", flush=True)
        return U, V, b, prop
    pu, pi, _, w = states.positives(d, T)
    c = {**DEFAULT, **cfg}
    if c.pop("fism", 0):
        from . import fism
        c = {k: c[k] for k in c if k in ("k", "l2", "wd", "epochs", "alpha", "scale", "L", "power")}
        U, V, b, prop = fism.fit(pu, pi, w, seed=seed, **c)
        p.parent.mkdir(exist_ok=True)
        np.savez(p, U=U, V=V, b=b, prop=prop)
        print(f"  fit {p.stem} {time.time() - t:.0f}s", flush=True)
        return U, V, b, prop
    if c.pop("expo_view", 0):  # label-free view: offers play the role of positives
        pu, pi = states.exposures(d, T)
        w = np.ones(len(pu), np.float32)
    if c.pop("offer_prop", 0):  # noise follows how often each item is offered, an item-level exposure correction
        c["noise_mass"] = np.bincount(states.exposures(d, T)[1], minlength=NI).astype(np.float64)
    if c.pop("bag", 0):  # Bayesian bootstrap: exponential weights average data noise, not just optimiser noise
        w = w * np.random.default_rng(seed * 1009 + T % 100003).exponential(1., len(w)).astype(np.float32)
    if c.get("own_noise"):
        c["expo"] = states.exposure_index(*states.exposures(d, T))
    if c.get("iota"):
        eu, ei = states.exposures(d, T)
        c["iexpo"] = states.exposure_index(ei, eu, NI)
    if c.get("beta"):
        c["init_expo"] = states.exposures(d, T)
    if c.get("gamma"):
        ev = np.where(d.ts[:d.n_train] < T)[0]
        c["choice"] = (d.user[ev], d.cand[ev], d.y[ev])
    c.pop("own_noise", None)
    U, V, b, prop = states.fit_nce(pu, pi, w, seed=seed, device="mps", **c)
    p.parent.mkdir(exist_ok=True)
    np.savez(p, U=U, V=V, b=b, prop=prop)
    print(f"  fit {p.stem} {time.time() - t:.0f}s", flush=True)
    return U, V, b, prop


def repeat_x(d, ev):
    c, h, ht, t = d.cand[ev], d.hist_item[ev], d.hist_ts[ev], d.ts[ev]
    eq = c[:, :, None] == h[:, None, :]
    cnt = eq.sum(2)
    age = np.where(eq, (t[:, None, None] - ht[:, None, :]) / 86400., 1e9).min(2)
    return np.stack([cnt > 0, np.log1p(cnt), np.where(cnt > 0, np.log1p(np.minimum(age, 60)), 0)], -1).astype(np.float32)


def density_x(u, V, b, c):
    s = (u[:, None, :] * V[c]).sum(2) + b[c]
    mean = (s.sum(1, keepdims=True) - s) / 9
    srt = np.sort(s, 1)
    rmax = np.where(s == srt[:, -1:], srt[:, -2:-1], srt[:, -1:])
    return np.stack([s, b[c], s - mean, s - rmax, np.repeat(s.std(1, keepdims=True), 10, 1)], -1).astype(np.float32)


def standardize(tr, *others, axis=0):
    m, s = tr.mean(axis), tr.std(axis) + 1e-9
    return [np.clip((a - m) / s, -8, 8) for a in (tr, *others)]


CFG = dict(l2=.02, wd=.3, epochs=80)


LOCAL_KEYS = ("il", "il_lr", "il_rounds", "il_users")


def item_local(d, day, U, V, b, prop, cfg):
    """Entity-mean refit of item vectors against their clickers; optional alternation with users' full positives."""
    pu, pi, _, w = states.positives(d, day_ts(day))
    l2, steps, lr = cfg.get("l2", DEFAULT["l2"]), cfg["il"], cfg.get("il_lr", .5)
    n_u = np.bincount(pu, weights=w, minlength=NU)
    rng = np.random.default_rng(104842)
    refs_u = rng.choice(NU, 256, p=n_u / n_u.sum())
    refs_i = rng.choice(NI, 256, p=prop)
    zero_u = np.zeros(NU, np.float32)
    zero_n = np.zeros(256, np.float32)
    for _ in range(cfg.get("il_rounds", 1)):
        if cfg.get("il_users"):
            U = states.local_pairs(U, V, zero_u, pu, pi, w, V[refs_i], b[refs_i], steps=steps, lr=lr, l2=l2)
        V = states.local_pairs(V, U, b, pi, pu, w, U[refs_u], zero_n, steps=steps, lr=lr, l2=l2)
    return U, V


def rejected_refs(d, day, ev, prop, R=64, seed=104842):
    """R noise references per event from its user's offers rejected before the day; users with none use the proposal."""
    e = np.where(d.ts[:d.n_train] < day_ts(day))[0]
    r, c = np.where(d.y[e] == 0)
    off, lst = states.exposure_index(d.user[e[r]], d.cand[e[r], c])
    u, rng = d.user[ev], np.random.default_rng(seed)
    n = off[u + 1] - off[u]
    pick = (rng.random((len(ev), R)) * n[:, None]).astype(np.int64)
    out = lst[np.minimum(off[u][:, None] + pick, len(lst) - 1)]
    out[n == 0] = rng.choice(NI, ((n == 0).sum(), R), p=prop)
    return out


class Producer:
    """States of one seed at successive daily cutoffs, rotated into the first cutoff's coordinates."""

    def __init__(self, d, first, last, seed=113, **cfg):
        self.d, self.seed, self.cfg, self.st, self.first = d, seed, cfg, {}, first
        ref = None
        fit_cfg = {k: v for k, v in cfg.items() if k not in ("local_steps", "unit_u", "unit_v", "slate_refs",
                                                             "slate_as_hist", "own_w", "fism_local")}
        for day in range(first, last + 1):
            U, V, b, prop = state(d, day_ts(day), seed, **{k: v for k, v in fit_cfg.items() if k not in LOCAL_KEYS})
            if cfg.get("il"):
                U, V = item_local(d, day, U, V, b, prop, cfg)
            if ref is None:
                ref = V
            else:
                R = states.procrustes(V, ref)
                U, V = U @ R, V @ R
            self.st[day] = (U, V, b, prop)

    def inputs(self, day, ev):
        """Query-local user vectors, the item table and density scalars of events scored by the `day` producer."""
        U, V, b, prop = self.st[day]
        if self.cfg.get("fism"):
            from . import fism
            c = self.cfg
            u = fism.pooled_user(U, self.d.hist_item[ev], c.get("alpha", 1.), c.get("scale", 3.))
            if c.get("fism_local"):  # optional refinement from the pooled start
                u = states.local_infer(u, V, b, prop, self.d.hist_item[ev], steps=c["fism_local"],
                                       l2={**DEFAULT, **c}["l2"])
            return u, V, density_x(u, V, b, self.d.cand[ev])
        steps = self.cfg.get("local_steps", 15)
        own = self.d.cand[ev] if self.cfg.get("slate_refs") else None
        own_w = self.cfg.get("own_w", 1 - self.cfg.get("mix", 0.))
        if self.cfg.get("rej_w"):
            own, own_w = rejected_refs(self.d, day, ev, prop), self.cfg["rej_w"]
        hist = self.d.cand[ev] if self.cfg.get("slate_as_hist") else self.d.hist_item[ev]
        u = states.local_infer(U[self.d.user[ev]], V, b, prop, hist, steps=steps,
                               l2={**DEFAULT, **self.cfg}["l2"], own=own, own_w=own_w)
        if self.cfg.get("unit_u"):
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6) * self.cfg["unit_u"]
        if self.cfg.get("unit_v"):
            V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-6) * self.cfg["unit_v"]
        return u, V, density_x(u, V, b, self.d.cand[ev])


def days_of(d, day):
    """Labelled event indices on the given January day."""
    t = d.ts[:d.n_train]
    return np.where((t >= day_ts(day)) & (t < day_ts(day + 1)))[0]


def main():
    """Fit states ahead of a run, so GPU and CPU stages can be scheduled apart: python -m hta.pipe CFG_JSON T,T seed,seed."""
    cfg, days, seeds = json.loads(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(",")
    d = D()
    for seed in seeds:
        for day in days:
            state(d, day_ts(int(day)), int(seed), **cfg)


if __name__ == "__main__":
    main()
