"""Cached state producers and per-event inputs. A state is fully keyed by its cutoff, seed and config."""
import time

import numpy as np

from . import states
from .lib import CACHE, DAY, NI, NU, day_ts

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
    if cfg.get("slate"):  # item table from shown slates, warm-started from the frozen NCE table of this cutoff
        from . import choice
        c = {**DEFAULT, **cfg}
        U, V, b, prop = choice.fit(d, T, *state(d, T, seed, **CFG)[1:3], seed=seed,
                                   **{k: c[k] for k in ("l2", "wd", "epochs", "alpha", "scale", "lr", "batch") if k in c})
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
