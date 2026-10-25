"""End-to-end candidate: causal daily states -> shared bilinear choice head -> LightGBM residual.

Every training day uses the producer fitted strictly before that day; query events use the producer at the cutoff.
Labels never enter a state of the event they train on. Scores are returned as log-odds, one array per seed.
"""
import json
import time

import lightgbm as lgb
import numpy as np

from . import features, head, pipe
from .lib import NI, RUNS, day_ts

LGB = dict(objective="binary", learning_rate=.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=.5,
           bagging_fraction=.8, bagging_freq=1, lambda_l2=30., max_bin=63, verbose=-1, num_threads=8)


def logit_of(lp):
    p = np.exp(np.minimum(lp, -1e-6))
    return np.log(p / (1 - p))


def table(lp, D, X, S):
    """Tree inputs: marginal log-odds, event click log-odds, and every per-candidate scalar."""
    l = logit_of(lp)
    e = np.minimum(np.exp(lp).sum(1), .999)
    ev = np.repeat(np.log(e / (1 - e)), 10)
    F = np.concatenate([l.reshape(-1, 1), ev[:, None], D.reshape(-1, D.shape[-1]), X.reshape(-1, X.shape[-1]),
                        S.reshape(-1, S.shape[-1])], 1)
    return F.astype(np.float32), l.ravel()


def user_day_weight(d, ev, alpha):
    """Event weight 1/n^alpha, n = that user's events in this block; the hidden set counts every user once."""
    if alpha == 0:
        return None
    _, inv, cnt = np.unique(d.user[ev], return_inverse=True, return_counts=True)
    w = cnt[inv].astype(np.float32) ** -alpha
    return w * len(w) / w.sum()


def run_seed(d, cut, test_ev, seed, cfg=pipe.CFG, lam=.1, trees=300, extra=None, log=print, alpha=0., tree_alpha=None,
             views=None, blocks=False, params=None, train_out=None, first_day=21, day_std=0, oof=0, support=0):
    """Fit on labelled days [21, cut) and return log-odds (n_test,10) for test_ev. alpha reweights heavy users.

    views: list of state configs, concatenated per day. blocks=True drops cross-view head interactions.
    day_std: standardise each table by its own moments, 1 items, 2 items and users (default: day-21 items, pooled users).
    oof: the tree learns from head logits of a head fit on the other days. support: adds features.support tree inputs.
    """
    t0 = time.time()
    views = [cfg] if views is None else views
    Ps = [pipe.Producer(d, first_day, cut, seed=seed, **c) for c in views]

    def inputs(day, ev):
        parts = [p.inputs(day, ev) for p in Ps]
        return (np.concatenate([x[0] for x in parts], 1), np.concatenate([x[1] for x in parts], 1),
                np.concatenate([x[2] for x in parts], -1))

    P = Ps[0]
    days = list(range(first_day, cut))
    data = []
    for day in days:
        ev = pipe.days_of(d, day)
        assert d.ts[ev].min() >= day_ts(day), "training rows precede their producer"
        u, V, D = inputs(day, ev)
        X = pipe.repeat_x(d, ev)
        S = features.scalars(d, ev)
        if support:
            S = np.concatenate([S, features.support(d, day, ev, D)], -1)
        if extra is not None:
            S = np.concatenate([S, extra(day, ev, P)], -1)
        data.append((ev, u, V, D, X, S))
    uq, Vq, Dq = inputs(cut, test_ev)
    Xq = pipe.repeat_x(d, test_ev)
    Sq = features.scalars(d, test_ev, cut_day=cut)
    if support:
        Sq = np.concatenate([Sq, features.support(d, cut, test_ev, Dq)], -1)
    if extra is not None:
        Sq = np.concatenate([Sq, extra(cut, test_ev, P)], -1)

    Ucat = np.concatenate([x[1] for x in data])
    Dcat = np.concatenate([np.concatenate([x[4], x[3]], -1) for x in data])
    moments = lambda a: (a.mean(0), a.std(0) + 1e-9)
    scale = lambda a, ms: np.clip((a - ms[0]) / ms[1], -8, 8)
    mu, mv = moments(Ucat), moments(data[0][2])
    xm, xs = Dcat.mean((0, 1)), Dcat.std((0, 1)) + 1e-9
    su = lambda a: scale(a, moments(a) if day_std > 1 else mu)
    sv = lambda a: scale(a, moments(a) if day_std else mv)
    sx = lambda a: np.clip((a - xm) / xs, -8, 8)
    groups = [head.Group(su(u), sv(V), d.cand[ev], d.y[ev], sx(np.concatenate([X, D], -1)),
                         user_day_weight(d, ev, alpha)) for ev, u, V, D, X, S in data]
    blk = [p.st[cut][1].shape[1] for p in Ps] if blocks and len(Ps) > 1 else None
    m = head.fit_groups(groups, lam=lam, iters=300, blocks=blk)
    gq = head.Group(su(uq), sv(Vq), d.cand[test_ev], None, sx(np.concatenate([Xq, Dq], -1)))
    lpq = head.score_group(m, gq)

    heads = [head.fit_groups(groups[:i] + groups[i + 1:], lam=lam, iters=300, blocks=blk)
             if oof and len(groups) > 1 else m for i in range(len(groups))]
    tabs = [table(head.score_group(h, g), x[3], x[4], x[5]) for h, g, x in zip(heads, groups, data)]
    F, base = np.concatenate([a for a, _ in tabs]), np.concatenate([b for _, b in tabs])
    lab = np.concatenate([d.y[x[0]].ravel() for x in data])
    ta = alpha if tree_alpha is None else tree_alpha
    wt = None if ta == 0 else np.concatenate([np.repeat(user_day_weight(d, x[0], ta), 10) for x in data])
    ds = lgb.Dataset(F, label=lab, init_score=base, weight=wt, free_raw_data=True)
    b = lgb.train({**LGB, "seed": seed, **(params or {})}, ds, num_boost_round=trees)
    Fq, lq = table(lpq, Dq, Xq, Sq)
    out = (lq + b.predict(Fq, raw_score=True)).reshape(-1, 10)
    if train_out is not None:  # stacked log-odds of the training rows, for a later residual stage
        train_out.update(ev=np.concatenate([x[0] for x in data]),
                         logit=(base + b.predict(F, raw_score=True)).reshape(-1, 10))
    log(f"  seed {seed} cut {cut}: head+tree done in {time.time() - t0:.0f}s")
    return out, lq.reshape(-1, 10), b


def ledger(name, **metrics):
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "at": time.strftime("%Y-%m-%d %H:%M"), **metrics}) + "\n")
