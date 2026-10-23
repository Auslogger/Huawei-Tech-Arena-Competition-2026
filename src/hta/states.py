"""Log-bilinear (NCE) user/item states and query-local user inference."""
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .lib import NI, NU, norm_incidence, rsvd


def positives(d, T, start=0):
    """Deduplicated positive (user,item,time,count) known strictly before T, from snapshots and labelled clicks.

    A click repeated by later snapshots counts once: count = max(max snapshot multiplicity, labelled clicks at key).
    """
    ev = np.where((d.ts[:d.n_train] < T) & (d.ts[:d.n_train] >= start))[0]
    m = d.hist_item[ev] >= 0
    n = m.sum(1)
    snap = (pl.DataFrame({"e": np.repeat(ev, n), "u": np.repeat(d.user[ev], n), "i": d.hist_item[ev][m],
                          "t": d.hist_ts[ev][m]})
            .group_by(["e", "u", "i", "t"]).len().group_by(["u", "i", "t"]).agg(pl.col("len").max().alias("c"))
            .with_columns(pl.col("c").cast(pl.Int32)))
    r, c = np.where(d.y[ev] == 1)
    lab = (pl.DataFrame({"u": d.user[ev[r]], "i": d.cand[ev[r], c], "t": d.ts[ev[r]]})
           .group_by(["u", "i", "t"]).len().rename({"len": "l"}).with_columns(pl.col("l").cast(pl.Int32)))
    j = snap.join(lab, on=["u", "i", "t"], how="full", coalesce=True).fill_null(0)
    j = j.with_columns(pl.max_horizontal("c", "l").alias("w"))
    assert j["t"].max() < T, "positive at or after the cutoff"
    return j["u"].to_numpy(), j["i"].to_numpy(), j["t"].to_numpy(), j["w"].to_numpy().astype(np.float32)


def exposures(d, T, start=0):
    """(user,item) rows of all offers strictly before T."""
    ev = np.where((d.ts[:d.n_train] < T) & (d.ts[:d.n_train] >= start))[0]
    return np.repeat(d.user[ev], 10), d.cand[ev].ravel()


def exposure_index(eu, ei, n=NU):
    """Per-user exposure lists as (offsets, items), label-free draws from that user's offer distribution."""
    order = np.argsort(eu, kind="stable")
    counts = np.bincount(eu, minlength=n)
    return np.r_[0, np.cumsum(counts)], ei[order]


def fit_nce(pu, pi, w=None, k=64, epochs=20, batch=8192, neg=16, lr=.005, wd=.01, l2=.005, seed=113, init="svd",
            device="cpu", power=.75, log=False, expo=None, mix=0., init_expo=None, beta=0., choice=None, gamma=0.,
            noise_mass=None, iexpo=None, iota=0., sphere=0., drop=0.):
    """Positive/noise logistic factorization. Noise is proposal-based, never an observed negative.

    expo=(offsets, items) draws noise from the user's own offer list, so the logit targets preference rather than
    exposure x preference; users without offers, and a `mix` share of noise, fall back to the global proposal.
    init_expo=(users, items) adds the offer graph with weight beta to the spectral start, for rare-item vectors.
    choice=(user, cand, y) adds gamma * slate-choice NLL that updates V only: U is detached, so users cannot memorise.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = np.ones(len(pu), np.float32) if w is None else w
    mass = np.bincount(pi, weights=w, minlength=NI).astype(np.float64) if noise_mass is None else noise_mass
    prop = mass ** power
    prop = prop / prop.sum()
    if init == "svd":
        B, _, _ = norm_incidence(pu, pi)
        if init_expo is not None and beta > 0:
            Bx, _, _ = norm_incidence(*init_expo)
            B = sp.vstack([B, np.sqrt(beta) * Bx]).tocsr()
        _, s, Vt = rsvd(B, k, seed=seed)
        V0 = Vt.T * np.sqrt(NI) * .3
        cnt = np.maximum(np.bincount(pu, weights=w, minlength=NU), 1)
        U0 = np.zeros((NU, k))
        np.add.at(U0, pu, V0[pi] * w[:, None])
        U0 = U0 / cnt[:, None]
    else:
        rng = np.random.default_rng(seed)
        V0 = rng.standard_normal((NI, k)) * .1
        U0 = rng.standard_normal((NU, k)) * .1
    dev = torch.device(device)
    U = torch.nn.Parameter(torch.tensor(U0, dtype=torch.float32, device=dev))
    V = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    b = torch.nn.Parameter(torch.zeros(NI, device=dev))
    groups = [{"params": [U, V, b], "weight_decay": wd}]
    if choice is not None:
        A = torch.nn.Parameter(torch.eye(k, device=dev) * .5)
        cb = torch.nn.Parameter(torch.zeros(NI, device=dev))
        c0 = torch.nn.Parameter(torch.tensor(-2.5, device=dev))
        groups.append({"params": [A, cb, c0], "weight_decay": .01})
        cu, cc, cy = [torch.as_tensor(a, device=dev) for a in (choice[0].astype(np.int64), choice[1].astype(np.int64),
                                                               choice[2].astype(np.float32))]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * max(len(pu) // batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.1 * lr)
    pt = torch.as_tensor(prop, dtype=torch.float32)
    # alias-free noise: pre-draw a large pool and index it
    pool = torch.multinomial(pt, min(50_000_000, max(len(pu) * neg, 1_000_000)), replacement=True, generator=g).to(dev)
    pu_t, pi_t = torch.as_tensor(pu.astype(np.int64), device=dev), torch.as_tensor(pi.astype(np.int64), device=dev)
    wt = torch.as_tensor(w, device=dev)
    if iexpo is not None:  # offered-user lists per item, the mirror of `expo`
        ioff = torch.as_tensor(iexpo[0].astype(np.int64), device=dev)
        ilst = torch.as_tensor(iexpo[1].astype(np.int64), device=dev)
    if expo is not None:
        off = torch.as_tensor(expo[0].astype(np.int64), device=dev)
        lst = torch.as_tensor(expo[1].astype(np.int64), device=dev)
    t0, step = time.time(), 0
    for ep in range(epochs):
        order = torch.randperm(len(pu), generator=g).to(dev)
        tot = 0.
        for s in range(0, len(pu) - batch + 1, batch):
            idx = order[s:s + batch]
            uu, ii, ww = pu_t[idx], pi_t[idx], wt[idx]
            nn = pool[torch.randint(len(pool), (len(idx) * neg,), generator=g).to(dev)].view(len(idx), neg)
            if expo is not None:
                lo, n = off[uu][:, None], (off[uu + 1] - off[uu])[:, None]
                pick = (torch.rand(len(idx), neg, generator=g).to(dev) * n).long()
                own = lst[(lo + pick).clamp(max=len(lst) - 1)]
                use = (n > 0) & (torch.rand(len(idx), neg, generator=g).to(dev) >= mix)
                nn = torch.where(use, own, nn)
            ue, ve, ne = U[uu], V[ii], V[nn]
            if sphere:  # unit vectors with shared temperature: direction only, and tau alone bounds capacity
                ue, ve, ne = [F.normalize(x, dim=-1) * sphere ** .5 for x in (ue, ve, ne)]
            if drop:
                ue = F.dropout(ue, drop)
            pos = (ue * ve).sum(1) + b[ii]
            nsc = (ne * ue[:, None, :]).sum(2) + b[nn]
            noise = F.softplus(nsc).mean(1)
            if iexpo is not None:  # noise users: others who were offered this same item
                lo, n = ioff[ii][:, None], (ioff[ii + 1] - ioff[ii])[:, None]
                pick = (torch.rand(len(idx), neg, generator=g).to(dev) * n).long()
                nu = ilst[(lo + pick).clamp(max=len(ilst) - 1)]
                sn2 = (U[nu] * ve[:, None, :]).sum(2) + b[ii][:, None]
                noise = (1 - iota) * noise + iota * F.softplus(sn2).mean(1)
            loss = (ww * (.5 * F.softplus(-pos) + .5 * noise)).sum() / ww.sum()
            reg = (ue.pow(2).sum(1).mean() + ve.pow(2).sum(1).mean() + ne.pow(2).sum(2).mean())
            total = loss + l2 * reg
            if choice is not None:
                e = torch.randint(len(cu), (2048,), generator=g).to(dev)
                zz = ((U[cu[e]].detach() @ A)[:, None, :] * V[cc[e]]).sum(2) + cb[cc[e]] + c0
                ll = torch.logaddexp(torch.zeros_like(zz[:, 0]), torch.logsumexp(zz, 1)) - (zz * cy[e]).sum(1)
                total = total + gamma * ll.mean()
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([U, V, b], 5.)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            step += 1
        if log:
            print(f"ep {ep} loss {tot / max(len(pu) // batch, 1):.4f} {time.time() - t0:.0f}s", flush=True)
    if sphere:
        U.data, V.data = F.normalize(U.data, dim=-1) * sphere ** .5, F.normalize(V.data, dim=-1) * sphere ** .5
    return U.detach().cpu().numpy(), V.detach().cpu().numpy(), b.detach().cpu().numpy(), prop


@torch.no_grad()
def local_infer(U0, V, b, prop, hist, steps=15, lr=.5, l2=.005, seed=104842, n_refs=256, kappa=0., anchor=False,
                own=None, own_w=.5):
    """Own-history density update from prior U0 (n,k); hist (n,L) int items with -1 padding. Duplicates are kept.

    kappa scales the data term by n/(n+kappa), so short histories move less; anchor pulls towards U0 instead of 0.
    own (n,R) adds event-specific noise references, e.g. the slate, for states trained against the user's offers.
    """
    rng = np.random.default_rng(seed)
    refs = rng.choice(NI, n_refs, p=prop)
    Vt, bt = torch.as_tensor(V), torch.as_tensor(b)
    Vr, br = Vt[refs], bt[refs]
    h = torch.as_tensor(hist.astype(np.int64))
    m = (h >= 0).float()
    cnt = m.sum(1, keepdim=True)
    h = h.clamp(min=0)
    Hv, Hb = Vt[h], bt[h]  # (n,L,k), (n,L)
    u0 = torch.as_tensor(U0)
    u = u0.clone()
    has = (cnt[:, 0] > 0)
    if own is not None:
        o = torch.as_tensor(own.astype(np.int64))
        Vo, bo = Vt[o], bt[o]
    conf = cnt / (cnt + kappa) if kappa > 0 else 1.
    for _ in range(steps):
        sp = (Hv * u[:, None, :]).sum(2) + Hb
        gp = -((torch.sigmoid(-sp) * m)[..., None] * Hv).sum(1) / cnt.clamp(min=1)
        sn = u @ Vr.T + br
        gn = torch.sigmoid(sn) @ Vr / n_refs
        if own is not None:
            so = (Vo * u[:, None, :]).sum(2) + bo
            gn = (1 - own_w) * gn + own_w * (torch.sigmoid(so)[..., None] * Vo).mean(1)
        g = conf * (.5 * gp + .5 * gn) + 2 * l2 * (u - u0 if anchor else u)
        u[has] -= lr * g[has]
    return u.numpy()


def procrustes(V, Vref):
    """Orthogonal map R with V @ R close to Vref on row-normalised item vectors."""
    A = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    Bn = Vref / np.maximum(np.linalg.norm(Vref, axis=1, keepdims=True), 1e-9)
    Ua, _, Vb = np.linalg.svd(A.T @ Bn, full_matrices=False)
    return Ua @ Vb


@torch.no_grad()
def local_pairs(X0, F, off, row, col, w, F_noise, off_noise, steps=15, lr=.5, l2=.02):
    """Entity-mean update of every row of X0 from its positive partners, the item-side twin of local_infer.

    The data term is a per-entity mean, so each entity moves the same distance whatever its count. off is per row.
    """
    X = torch.as_tensor(X0).clone()
    F, off, F_noise, off_noise = [torch.as_tensor(a) for a in (F, off, F_noise, off_noise)]
    row, col = torch.as_tensor(row.astype(np.int64)), torch.as_tensor(col.astype(np.int64))
    w = torch.as_tensor(w, dtype=torch.float32)
    cnt = torch.zeros(len(X)).index_add_(0, row, w)
    has = cnt > 0
    Fc = F[col]
    for _ in range(steps):
        s = (X[row] * Fc).sum(1) + off[row]
        gp = torch.zeros_like(X).index_add_(0, row, -(w * torch.sigmoid(-s))[:, None] * Fc) / cnt.clamp(min=1)[:, None]
        sn = X @ F_noise.T + off[:, None] + off_noise
        gn = torch.sigmoid(sn) @ F_noise / len(F_noise)
        g = .5 * gp + .5 * gn + 2 * l2 * X
        X[has] -= lr * g[has]
    return X.numpy()
