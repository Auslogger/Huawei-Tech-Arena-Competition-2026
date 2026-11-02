"""End-to-end candidate: causal daily states -> shared bilinear choice head -> LightGBM residual.

Every training day uses the producer fitted strictly before that day, and query events use the producer at the cutoff.
Labels never enter a state of the event they train on. Scores are returned as log-odds, one array per seed.
"""

import json
import time

import lightgbm as lgb
import numpy as np

from . import features, head, pipe
from .lib import NI, RUNS, THREADS, day_ts, first_events

LGB = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=200,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=30.0,
    max_bin=63,
    verbose=-1,
    num_threads=THREADS,
)


def logit_of(lp):
    p = np.exp(np.minimum(lp, -1e-6))
    return np.log(p / (1 - p))


def table(lp, D, X, S):
    """Tree inputs: marginal log-odds, event click log-odds, and every per-candidate scalar."""
    l = logit_of(lp)
    e = np.minimum(np.exp(lp).sum(1), 0.999)
    ev = np.repeat(np.log(e / (1 - e)), 10)
    F = np.concatenate(
        [
            l.reshape(-1, 1),
            ev[:, None],
            D.reshape(-1, D.shape[-1]),
            X.reshape(-1, X.shape[-1]),
            S.reshape(-1, S.shape[-1]),
        ],
        1,
    )
    return F.astype(np.float32), l.ravel()


def utility(l):
    """Outside-option utilities (n,10) from marginal log-odds (n,10)."""
    lp = -np.log1p(np.exp(-l))
    return lp - np.log(np.maximum(1 - np.exp(lp).sum(1, keepdims=True), 1e-9))


def marginal_logit(z):
    """Marginal log-odds (n,10) from outside-option utilities."""
    lp = z - np.log1p(np.exp(z).sum(1, keepdims=True))
    return lp - np.log(np.maximum(1 - np.exp(lp), 1e-12))


def event_objective(y):
    """LightGBM objective: outside-option NLL of each event. y (n,10). Scores are utilities, so a slate's rows compete."""

    def f(preds, _):
        s = preds.reshape(-1, 10)
        top = np.maximum(s.max(1, keepdims=True), 0.0)
        e = np.exp(s - top)
        p = e / (np.exp(-top) + e.sum(1, keepdims=True))
        return (p - y).ravel(), np.maximum(p * (1 - p), 1e-6).ravel()

    return f


def user_day_weight(d, ev, alpha):
    """Event weight 1/n^alpha, n = that user's events in this block. The hidden set counts every user once."""
    if alpha == 0:
        return None
    _, inv, cnt = np.unique(d.user[ev], return_inverse=True, return_counts=True)
    w = cnt[inv].astype(np.float32) ** -alpha
    return w * len(w) / w.sum()


def run_seed(
    d,
    cut,
    test_ev,
    seed,
    cfg=pipe.CFG,
    lam=0.1,
    trees=300,
    extra=None,
    log=print,
    alpha=0.0,
    tree_alpha=None,
    views=None,
    blocks=False,
    params=None,
    train_out=None,
    first_day=21,
    day_std=0,
    oof=0,
    support=0,
    extra_head=None,
    pop=0,
    cats=0,
    head_u=0,
    lam_b=None,
    prx=0,
    te=0,
    slate=None,
    slate_base=0,
):
    """Fit on labelled days [21, cut) and return log-odds (n_test,10) for test_ev. alpha reweights heavy users.

    views: list of state configs, concatenated per day. blocks=True drops cross-view head interactions.
    day_std: standardise each table by its own moments, 1 items, 2 items and users (default: day-21 items, pooled users).
    oof: the tree learns from head logits of a head fit on the other days. support: adds features.support tree inputs.
    extra: extra(day, ev, producer) (n,10,c) goes to the tree only. extra_head: same signature, goes to head and tree.
    pop: training rows of each day's table. 0 all that day's events, 1 each user's first event that day, 2 each user's first
    event from that day to the cut, the hidden set's own construction.
    cats: adds the exposure model's inputs to head and tree (groups.Cats, rich=cats). head_u: also gives the head the user
    scalars, so a user's propensity enters the choice model and not only the tree. prx: adds the density scalars of the table's
    top prx directions to the tree. te: the tree learns the event's outside-option likelihood from the head's utilities, not
    row log-loss. slate: names of slate chains whose outputs go to head and tree (slate.Inputs). slate_base: the chain's
    predictive replaces the head, so the tree starts from it.
    """
    t0 = time.time()
    if cats:
        from . import groups

        extra_head = groups.Cats(d, cut, rich=cats)
    if head_u:
        inner = extra_head

        def extra_head(day, ev, P):
            S = features.scalars(d, ev, cut_day=None if day < cut else cut)[..., [5, 6, 7, 8, 9]]
            return S if inner is None else np.concatenate([inner(day, ev, P), S], -1)

    if slate:
        from .slate import Inputs

        chains, inner2 = Inputs(slate, seed), extra_head

        def extra_head(day, ev, P):
            S = chains(day, ev)
            return S if inner2 is None else np.concatenate([inner2(day, ev, P), S], -1)

    views = [cfg] if views is None else views
    Ps = [pipe.Producer(d, first_day, cut, seed=seed, **c) for c in views]
    if prx:

        def extra(day, ev, P):
            U, V, b, prop = P.st[day]
            Q = np.linalg.svd(V, full_matrices=False)[2][:prx].T
            Up, Vp = (U @ Q).astype(np.float32), (V @ Q).astype(np.float32)
            u = pipe.states.local_infer(Up[d.user[ev]], Vp, b, prop, d.hist_item[ev], l2=0.02)
            return pipe.density_x(u, Vp, b, d.cand[ev])[..., [0, 2, 3, 4]]

    def inputs(day, ev):
        parts = [p.inputs(day, ev) for p in Ps]
        return (
            np.concatenate([x[0] for x in parts], 1),
            np.concatenate([x[1] for x in parts], 1),
            np.concatenate([x[2] for x in parts], -1),
        )

    P = Ps[0]
    days = list(range(first_day, cut))
    data = []
    for day in days:
        ev = pipe.days_of(d, day) if not pop else first_events(d, day_ts(day), day_ts(day + 1 if pop == 1 else cut))
        assert d.ts[ev].min() >= day_ts(day) and d.ts[ev].max() < day_ts(cut), "training rows outside their window"
        u, V, D = inputs(day, ev)
        X = pipe.repeat_x(d, ev)
        if extra_head is not None:
            X = np.concatenate([X, extra_head(day, ev, P)], -1)
        S = features.scalars(d, ev, cut_day=day if pop else None)
        if support:
            S = np.concatenate([S, features.support(d, day, ev, D)], -1)
        if extra is not None:
            S = np.concatenate([S, extra(day, ev, P)], -1)
        data.append((ev, u, V, D, X, S))
    uq, Vq, Dq = inputs(cut, test_ev)
    Xq = pipe.repeat_x(d, test_ev)
    if extra_head is not None:
        Xq = np.concatenate([Xq, extra_head(cut, test_ev, P)], -1)
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
    groups = [
        head.Group(su(u), sv(V), d.cand[ev], d.y[ev], sx(np.concatenate([X, D], -1)), user_day_weight(d, ev, alpha))
        for ev, u, V, D, X, S in data
    ]
    blk = [p.st[cut][1].shape[1] for p in Ps] if blocks and len(Ps) > 1 else None
    if slate_base:
        lpq = np.log(chains.p(cut, test_ev))
        tabs = [table(np.log(chains.p(day, x[0])), x[3], x[4], x[5]) for day, x in zip(days, data)]
    else:
        m = head.fit_groups(groups, lam=lam, iters=300, blocks=blk, lam_b=lam_b)
        gq = head.Group(su(uq), sv(Vq), d.cand[test_ev], None, sx(np.concatenate([Xq, Dq], -1)))
        lpq = head.score_group(m, gq)
        heads = [
            head.fit_groups(groups[:i] + groups[i + 1 :], lam=lam, iters=300, blocks=blk, lam_b=lam_b)
            if oof and len(groups) > 1
            else m
            for i in range(len(groups))
        ]
        tabs = [table(head.score_group(h, g), x[3], x[4], x[5]) for h, g, x in zip(heads, groups, data)]
    F, base = np.concatenate([a for a, _ in tabs]), np.concatenate([b for _, b in tabs])
    lab = np.concatenate([d.y[x[0]].ravel() for x in data])
    ta = alpha if tree_alpha is None else tree_alpha
    wt = None if ta == 0 else np.concatenate([np.repeat(user_day_weight(d, x[0], ta), 10) for x in data])
    Fq, lq = table(lpq, Dq, Xq, Sq)
    if te:
        z0 = utility(base.reshape(-1, 10))
        ds = lgb.Dataset(F, label=lab, init_score=z0.ravel(), free_raw_data=False)
        ob = event_objective(lab.reshape(-1, 10).astype(np.float64))
        b = lgb.train({**LGB, "seed": seed, **(params or {}), "objective": ob}, ds, num_boost_round=trees)
        out = marginal_logit(utility(lq.reshape(-1, 10)) + b.predict(Fq, raw_score=True).reshape(-1, 10))
    else:
        ds = lgb.Dataset(F, label=lab, init_score=base, weight=wt, free_raw_data=True)
        b = lgb.train({**LGB, "seed": seed, **(params or {})}, ds, num_boost_round=trees)
        out = (lq + b.predict(Fq, raw_score=True)).reshape(-1, 10)
    if train_out is not None and train_out.get("keep"):  # keep the tree's own tables around for another stacker
        train_out.update(
            F=F.reshape(-1, 10, F.shape[-1]), Fq=Fq.reshape(-1, 10, Fq.shape[-1]), base=base.reshape(-1, 10)
        )
    if train_out is not None:  # stacked log-odds of the training rows, used by a later residual stage
        raw = b.predict(F, raw_score=True).reshape(-1, 10)
        train_out.update(
            ev=np.concatenate([x[0] for x in data]),
            head=base.reshape(-1, 10),
            logit=marginal_logit(utility(base.reshape(-1, 10)) + raw) if te else base.reshape(-1, 10) + raw,
            su=np.concatenate([su(x[1]) for x in data]),
            suq=su(uq),
        )
    log(f"  seed {seed} cut {cut}: head+tree done in {time.time() - t0:.0f}s")
    return out, lq.reshape(-1, 10), b


def ledger(name, **metrics):
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "at": time.strftime("%Y-%m-%d %H:%M"), **metrics}) + "\n")
