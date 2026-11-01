"""Trained graph table: vectors smoothed over the positive graph, under the positive/noise objective of states.fit_nce.

Base vectors are free and final vectors are the mean over propagation layers (LightGCN). Each batch's own edges are masked
out of the propagation, so a target pair never explains itself. Final vectors are returned, so pipe.Producer treats the
table as any NCE state, and layers=0 is the matched control.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .lib import NI, NU, norm_incidence, rsvd


def propagate(Eu, Ei, eu, ei, a, layers):
    """Mean of the base vectors and each hop. a (E,) holds the edge weights, zero for masked edges."""
    Uo, Vo, Us, Vs = Eu, Ei, Eu, Ei
    for _ in range(layers):
        Un = torch.zeros_like(Eu).index_add(0, eu, a[:, None] * Vo[ei])
        Vn = torch.zeros_like(Ei).index_add(0, ei, a[:, None] * Uo[eu])
        Uo, Vo, Us, Vs = Un, Vn, Us + Un, Vs + Vn
    return Us / (layers + 1), Vs / (layers + 1)


def fit(
    pu,
    pi,
    w,
    layers=2,
    k=64,
    epochs=160,
    batch=32768,
    neg=16,
    lr=0.005,
    wd=0.3,
    l2=0.02,
    power=0.75,
    seed=113,
    device="mps",
    every=40,
):
    """(U, V, b, prop) from positives (user, item, weight). One edge per pair, weighted by its summed positives."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    mass = np.bincount(pi, weights=w, minlength=NI).astype(np.float64)
    prop = mass**power / (mass**power).sum()
    B, _, _ = norm_incidence(pu, pi)
    V0 = rsvd(B, k, seed=seed)[2].T * np.sqrt(NI) * 0.3
    cnt = np.maximum(np.bincount(pu, weights=w, minlength=NU), 1)
    U0 = np.zeros((NU, k))
    np.add.at(U0, pu, V0[pi] * w[:, None])
    U0 = U0 / cnt[:, None]
    key, inv = np.unique(pu.astype(np.int64) * NI + pi, return_inverse=True)
    eu_, ei_ = key // NI, key % NI
    ew = np.bincount(inv, weights=w).astype(np.float32)
    du, di = np.bincount(eu_, minlength=NU).astype(np.float64), np.bincount(ei_, minlength=NI).astype(np.float64)
    a = torch.as_tensor((du[eu_] * di[ei_]) ** -0.5, dtype=torch.float32, device=dev)
    eu, ei = torch.as_tensor(eu_, device=dev), torch.as_tensor(ei_, device=dev)
    wt = torch.as_tensor(ew, device=dev)
    Eu = torch.nn.Parameter(torch.tensor(U0, dtype=torch.float32, device=dev))
    Ei = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    b = torch.nn.Parameter(torch.zeros(NI, device=dev))
    opt = torch.optim.AdamW([Eu, Ei, b], lr=lr, weight_decay=wd)
    pool = torch.multinomial(torch.as_tensor(prop, dtype=torch.float32), 50_000_000, True).to(dev)
    per = max(len(key) // batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * per, eta_min=0.1 * lr)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(key), device=dev)
        tot = torch.zeros((), device=dev)
        for s in range(per):
            idx = perm[s * batch : (s + 1) * batch]
            U, V = propagate(Eu, Ei, eu, ei, a.index_fill(0, idx, 0.0), layers)
            uu, ii, ww = eu[idx], ei[idx], wt[idx]
            nn = pool[torch.randint(len(pool), (len(idx) * neg,), device=dev)].view(len(idx), neg)
            ue, ve, ne = U[uu], V[ii], V[nn]
            sp = (ue * ve).sum(1) + b[ii]
            sn = (ne * ue[:, None, :]).sum(2) + b[nn]
            loss = (ww * (0.5 * F.softplus(-sp) + 0.5 * F.softplus(sn).mean(1))).sum() / ww.sum()
            reg = ue.pow(2).sum(1).mean() + ve.pow(2).sum(1).mean() + ne.pow(2).sum(2).mean()
            opt.zero_grad(set_to_none=True)
            (loss + l2 * reg).backward()
            torch.nn.utils.clip_grad_norm_([Eu, Ei, b], 5.0)
            opt.step()
            sched.step()
            tot = tot + loss.detach()
        if (ep + 1) % every == 0 or ep == epochs - 1:
            print(f"  ep {ep} loss {float(tot) / per:.4f} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        U, V = propagate(Eu, Ei, eu, ei, a, layers)
        return U.cpu().numpy(), V.cpu().numpy(), b.cpu().numpy(), prop
