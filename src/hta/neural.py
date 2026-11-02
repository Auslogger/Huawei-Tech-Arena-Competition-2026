"""Shared-table choice model trained on the shown slates, with no free user parameters (PLAN.md Step 1).

A producer N_T fits on labelled events strictly before T, days 20 to T-1, and scores day-T rows, so a row's label never
reaches the model that scores it. One item table serves history and candidate roles, and a user is a pooled history. Its
start table comes from warm-up clicks only, so the start holds no labelled outcome. Selection uses a user-level holdout
inside the producer's own window, never its next day.
"""

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import features, head, model, pipe, states
from .lib import CACHE, D, DAY, NI, T0, THREADS, auc, day_ts

DEFAULT = dict(
    k=16,
    lr=0.005,
    batch=1024,
    wd=0.5,
    hdrop=0.2,
    mu=0.0,
    init="warm",
    attn=0,
    width=32,
    epochs=0,
    l2=0.0,
    anchor=0.0,
    freeze=0,
    bfree=0,
    sub=1.0,
)
GRID = (1, 2, 3, 4)  # with epochs=0, pick one of these per producer by holdout AUC
KEEP = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]  # scalars that don't grow with the calendar
PATH = CACHE / "neural"


class Net(nn.Module):
    def __init__(self, E0, b0, nx, attn, width):
        super().__init__()
        k = E0.shape[1]
        self.k, self.attn = k, attn
        self.E = nn.Parameter(torch.as_tensor(E0, dtype=torch.float32))
        self.b = nn.Parameter(torch.as_tensor(b0, dtype=torch.float32))
        self.Wh = nn.Parameter(torch.eye(k) * 3.0)
        self.c0 = nn.Parameter(torch.tensor(-1.8))
        self.f = nn.Sequential(nn.Linear(k + 2 + nx, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        if attn:
            self.Wq, self.Wk = nn.Parameter(torch.eye(k)), nn.Parameter(torch.eye(k))

    def pool(self, h):
        """(history vectors in the user role, mean user). An empty history pools to zero."""
        m = h >= 0
        Hr = self.E[h.clamp(min=0)]
        Hv = Hr @ self.Wh
        return Hr, Hv, m, (Hv * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    def forward(self, h, c, x):
        Hr, Hv, m, um = self.pool(h)
        C = self.E[c]
        u = um[:, None, :]
        if self.attn:
            a = (C @ self.Wq) @ (Hr @ self.Wk).transpose(1, 2) / self.k**0.5
            a = a.masked_fill(~m[:, None, :], -1e4).softmax(2) * m[:, None, :]
            u = 0.5 * (u + a @ Hv)
        dot = (u * C).sum(2)
        bias = self.b[c]
        inp = torch.cat([um[:, None, :] * C, dot[..., None], bias[..., None], x], -1)
        return dot + bias + self.c0 + self.f(inp)[..., 0]


def key(T, seed, cfg):
    c = {**DEFAULT, **cfg}
    return f"N_T{(T - day_ts(20)) // DAY + 20}_s{seed}_" + "_".join(f"{k}{c[k]}" for k in sorted(c))


def nll(z, y):
    """Outside-option NLL per event."""
    return (torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1)) - (z * y).sum(1)).mean()


def warm_table(d, T, seed, k):
    """Item table and bias from warm-up clicks known before T: positive/noise, label-free."""
    p = PATH / f"warm_T{(T - day_ts(20)) // DAY + 20}_s{seed}_k{k}.npz"
    if p.exists():
        z = np.load(p)
        return z["V"], z["b"], z["prop"]
    pu, pi, pt, w = states.positives(d, T)
    keep = pt < T0
    assert pt[keep].max() < T0
    _, V, b, prop = states.fit_nce(
        pu[keep], pi[keep], w[keep], k=k, l2=0.02, wd=0.3, epochs=80, seed=seed, device="mps"
    )
    PATH.mkdir(exist_ok=True)
    np.savez(p, V=V, b=b, prop=prop)
    return V, b, prop


_tabs = {}


def inputs(d, ev=None, cut=None):
    """Standard per-candidate scalars. ev=None gives all labelled events, each by its own day. cut pins item statistics."""
    if ev is None:
        if "all" not in _tabs:
            _tabs["all"] = inputs(d, np.arange(d.n_train))
        return _tabs["all"]
    return np.concatenate([features.scalars(d, ev, cut_day=cut)[..., KEEP], pipe.repeat_x(d, ev)], -1)


def fit(d, T, seed=113, device="mps", hold=0.0, log=print, run=0, **cfg):
    """Train on events before T. hold>0 keeps that share of users out and reports their NLL per epoch.

    run reseeds only the optimiser path, and with sub<1 draws that share of training events, so runs share table and holdout.
    """
    c = {**DEFAULT, **cfg}
    torch.manual_seed(seed + 1000 * run)
    dev = torch.device(device)
    t = d.ts[: d.n_train]
    ev = np.where(t < T)[0]
    assert d.ts[ev].max() < T and (d.hist_ts[ev].max(1) <= d.ts[ev]).all()
    held = np.random.default_rng(seed + 7).random(d.user.max() + 1) < hold
    tr, ho = ev[~held[d.user[ev]]], ev[held[d.user[ev]]]
    if c["sub"] < 1:
        tr = np.sort(np.random.default_rng(seed + 1000 * run + 1).choice(tr, int(len(tr) * c["sub"]), replace=False))
    nxt = pipe.days_of(d, (T - day_ts(20)) // DAY + 20) if T < day_ts(27) else np.array([], int)
    A = inputs(d)
    mx, sx = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-9
    tens = lambda x, ty=None: torch.as_tensor(x if ty is None else x.astype(ty), device=dev)
    std = lambda a: np.clip((a - mx) / sx, -8, 8).astype(np.float32)
    H, C, Y, X = (
        tens(d.hist_item[tr], np.int64),
        tens(d.cand[tr], np.int64),
        tens(d.y[tr], np.float32),
        tens(std(A[tr])),
    )
    if c["init"] == "warm":
        V0, b0, prop = warm_table(d, T, seed, c["k"])
    else:
        rng = np.random.default_rng(seed)
        V0, b0 = rng.standard_normal((NI, c["k"])).astype(np.float32) * 0.1, np.zeros(NI, np.float32)
        prop = warm_table(d, T, seed, c["k"])[2]
    m = Net(V0, b0, X.shape[-1], c["attn"], c["width"]).to(dev)
    E0 = m.E.detach().clone()
    m.E.requires_grad_(not c["freeze"])
    anchored = bool(
        c["anchor"]
    )  # decay toward the warm-up table (a Gaussian prior centred on it) rather than toward zero
    decay = ([] if anchored or c["freeze"] else [m.E]) + [m.Wh] + ([m.Wq, m.Wk] if c["attn"] else [])
    m.b.requires_grad_(bool(c["bfree"]))  # item CTR variance is close to binomial noise, so a free bias just memorises
    rest = [m.c0, *m.f.parameters()] + ([m.E] if anchored else []) + ([m.b] if c["bfree"] else [])
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": c["wd"]}, {"params": rest, "weight_decay": 0.0}], lr=c["lr"]
    )
    per = max(len(tr) // c["batch"], 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["epochs"] * per, eta_min=0.05 * c["lr"])
    pool = torch.multinomial(torch.as_tensor(prop, dtype=torch.float32), 20_000_000, True).to(dev) if c["mu"] else None
    rows = {}
    t0 = time.time()
    for ep in range(c["epochs"]):
        perm = torch.randperm(len(tr), device=dev)
        tot = torch.zeros((), device=dev)
        for s in range(per):
            j = perm[s * c["batch"] : (s + 1) * c["batch"]]
            h = H[j]
            if c["hdrop"]:
                h = h.masked_fill(torch.rand(h.shape, device=dev) < c["hdrop"], -1)
            z = m(h, C[j], X[j])
            loss = nll(z, Y[j])
            tot = tot + loss.detach()
            if c["l2"]:
                loss = loss + c["l2"] * (m.E[C[j]].pow(2).sum(2).mean() + m.E[H[j].clamp(min=0)].pow(2).sum(2).mean())
            if c["mu"]:
                loss = loss + c["mu"] * leave_one_out(m, H[j], pool)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if anchored:
                m.E.data.sub_(sched.get_last_lr()[0] * c["anchor"] * (m.E.data - E0))
            sched.step()
        msg = f"ep {ep} train {float(tot) / per:.4f} moved {float((m.E.detach() - E0).norm() / E0.norm()):.3f}"
        if len(ho):
            st = state_of(m, mx, sx)
            s_h = score(st, d, ho, std(A[ho]))
            rows[ep] = (
                float(nll(torch.as_tensor(s_h), torch.as_tensor(d.y[ho], dtype=torch.float32))),
                auc(d.y[ho], s_h),
            )
            msg += f" hold {rows[ep][0]:.4f} auc {rows[ep][1]:.4f}"
        if len(nxt) and c.get("show_next"):
            st = state_of(m, mx, sx)
            s_n = score(st, d, nxt, std(A[nxt]))
            msg += f" next {float(nll(torch.as_tensor(s_n), torch.as_tensor(d.y[nxt], dtype=torch.float32))):.4f} auc {auc(d.y[nxt], s_n):.4f}"
        log(f"  {msg} {time.time() - t0:.0f}s")
    return state_of(m, mx, sx), rows


def leave_one_out(m, h, pool, neg=16):
    """Predict one history item from the others, against popularity noise, sharing the item table."""
    n, L = h.shape
    mask = h >= 0
    has = mask.sum(1) >= 2
    if not has.any():
        return torch.zeros((), device=h.device)
    h, mask = h[has], mask[has]
    pick = (torch.rand(mask.shape, device=h.device) * mask).argmax(1)
    tgt = h.gather(1, pick[:, None])[:, 0]
    rest = h.scatter(1, pick[:, None], -1)
    _, _, _, um = m.pool(rest)
    nn_ = pool[torch.randint(len(pool), (len(h) * neg,), device=h.device)].view(len(h), neg)
    pos = (um * m.E[tgt]).sum(1) + m.b[tgt]
    noise = (m.E[nn_] * um[:, None, :]).sum(2) + m.b[nn_]
    return 0.5 * F.softplus(-pos).mean() + 0.5 * F.softplus(noise).mean()


def state_of(m, mx, sx):
    st = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    st["_mx"], st["_sx"], st["_attn"] = mx, sx, m.attn
    return st


def rebuild(st):
    """A CPU copy of the model, for scoring."""
    keys = {k: v for k, v in st.items() if not k.startswith("_")}
    nx = keys["f.0.weight"].shape[1] - keys["E"].shape[1] - 2
    m = Net(keys["E"].numpy(), keys["b"].numpy(), nx, st["_attn"], keys["f.0.weight"].shape[0])
    m.load_state_dict(keys)
    return m.eval()


@torch.no_grad()
def score(st, d, ev, x, chunk=20000):
    """Utilities (n,10) of events. x holds the standardised scalars."""
    m = rebuild(st)
    h, c = torch.as_tensor(d.hist_item[ev].astype(np.int64)), torch.as_tensor(d.cand[ev].astype(np.int64))
    x = torch.as_tensor(x)
    return torch.cat(
        [m(h[i : i + chunk], c[i : i + chunk], x[i : i + chunk]) for i in range(0, len(ev), chunk)]
    ).numpy()


def save(st, p):
    PATH.mkdir(exist_ok=True)
    torch.save(st, p)


def load(p):
    return torch.load(p, weights_only=False)


def producer(d, T, seed=113, **cfg):
    """The producer N_T: fit once, cached by cutoff, seed and config."""
    p = PATH / f"{key(T, seed, cfg)}.pt"
    if p.exists():
        return load(p)
    t = time.time()
    c = dict(cfg)
    if not c.get(
        "epochs", DEFAULT["epochs"]
    ):  # a shorter cosine schedule takes a different path, so every length is a separate run
        hold = {e: fit(d, T, seed, hold=0.1, log=lambda s: None, **{**c, "epochs": e})[1][e - 1] for e in GRID}
        c["epochs"] = max(hold, key=lambda e: hold[e][1])
        print(f"  {p.stem}: holdout " + " ".join(f"e{e} {v[0]:.4f}/{v[1]:.4f}" for e, v in hold.items()), flush=True)
    st, _ = fit(d, T, seed, log=lambda s: None, **c)
    st["_epochs"] = c["epochs"]
    save(st, p)
    print(f"  fit {p.stem} epochs {c['epochs']} {time.time() - t:.0f}s", flush=True)
    return load(p)


def outputs(st, d, ev, cut=None):
    """(n,10) log-marginal and (n,10,c) slate scalars of events, the same contrasts as pipe.density_x on utility z."""
    x = np.clip((inputs(d, ev, cut) - st["_mx"]) / st["_sx"], -8, 8).astype(np.float32)
    z = score(st, d, ev, x)
    lp = z - np.logaddexp(0, np.logaddexp.reduce(z, 1, keepdims=True))
    mean = (z.sum(1, keepdims=True) - z) / 9
    srt = np.sort(z, 1)
    rmax = np.where(z == srt[:, -1:], srt[:, -2:-1], srt[:, -1:])
    return lp, np.stack([z, z - mean, z - rmax, np.repeat(z.std(1, keepdims=True), 10, 1)], -1).astype(np.float32)


def batch_std(a):
    """Label-free scale of one scored batch, as model.run_seed's day_std does for coordinates."""
    return (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-9)


def extras(d, cut, seed=113, **cfg):
    """model.run_seed extra_head: N's scalars for the rows of `day`, standardised per scored batch."""

    def f(day, ev, _):
        lp, D_ = outputs(producer(d, day_ts(day), seed, **cfg), d, ev, cut if day == cut else None)
        return batch_std(np.concatenate([lp[..., None], D_], -1)).astype(np.float32)

    return f


def member(d, cut, q, seed=113, first_day=21, trees=300, log=print, train_out=None, cats=0, **cfg):
    """N as its own member: LightGBM over N's log-odds with N's scalars and counters, and no NCE density input."""
    t0 = time.time()
    tabs = []
    if cats:
        from . import groups

        C = groups.Cats(d, cut, rich=cats)
    for day in range(first_day, cut):
        ev = pipe.days_of(d, day)
        lp, D_ = outputs(producer(d, day_ts(day), seed, **cfg), d, ev)
        S = np.concatenate(
            [features.scalars(d, ev), features.support(d, day, ev, D_)] + ([C(day, ev)] if cats else []), -1
        )
        tabs.append(model.table(lp, D_, pipe.repeat_x(d, ev), S))
    lpq, Dq = outputs(producer(d, day_ts(cut), seed, **cfg), d, q, cut)
    Sq = np.concatenate(
        [features.scalars(d, q, cut_day=cut), features.support(d, cut, q, Dq)] + ([C(cut, q)] if cats else []), -1
    )
    F_, base = np.concatenate([a for a, _ in tabs]), np.concatenate([b for _, b in tabs])
    lab = np.concatenate([d.y[pipe.days_of(d, x)].ravel() for x in range(first_day, cut)])
    b = lgb.train({**model.LGB, "seed": seed}, lgb.Dataset(F_, label=lab, init_score=base, free_raw_data=True), trees)
    Fq, lq = model.table(lpq, Dq, pipe.repeat_x(d, q), Sq)
    out = (lq + b.predict(Fq, raw_score=True)).reshape(-1, 10)
    if train_out is not None:
        train_out.update(lq=lq.reshape(-1, 10))
    log(f"  N member seed {seed} cut {cut}: {time.time() - t0:.0f}s")
    return out


def main():
    """Fit producers: python -m hta.neural CFG_JSON T,T seed,seed."""
    cfg, days, seeds = json.loads(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(",")
    d = D()
    for s in seeds:
        for day in days:
            producer(d, day_ts(int(day)), int(s), **cfg)


if __name__ == "__main__":
    main()
