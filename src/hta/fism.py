"""Item-to-item log-bilinear model: a user is the pooled representation of their other clicks.

There are no free user parameters, so users cannot be memorised, and the fit target is the query-time task: predict a
held-out click from the rest of that user's positives. P pools history, Q scores candidates.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .lib import NI, NU, norm_incidence, rsvd


def fit(pu, pi, w, k=64, epochs=40, batch=8192, neg=16, lr=.005, wd=.3, l2=.02, L=20, power=.75, seed=113, alpha=1.,
        scale=3., device="mps", log=False):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(device)
    mass = np.bincount(pi, weights=w, minlength=NI).astype(np.float64)
    prop = mass ** power
    prop = prop / prop.sum()
    B, _, _ = norm_incidence(pu, pi)
    _, _, Vt = rsvd(B, k, seed=seed)
    V0 = Vt.T * np.sqrt(NI) * .3
    P = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    Q = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    b = torch.nn.Parameter(torch.zeros(NI, device=dev))
    opt = torch.optim.AdamW([P, Q, b], lr=lr, weight_decay=wd)
    steps = epochs * max(len(pu) // batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.1 * lr)
    order = np.argsort(pu, kind="stable")
    off = torch.as_tensor(np.r_[0, np.cumsum(np.bincount(pu, minlength=NU))].astype(np.int64), device=dev)
    lst = torch.as_tensor(pi[order].astype(np.int64), device=dev)
    pool = torch.multinomial(torch.as_tensor(prop, dtype=torch.float32), max(len(pu) * neg, 10_000_000), True,
                             generator=g).to(dev)
    uu_, ii_ = torch.as_tensor(pu.astype(np.int64), device=dev), torch.as_tensor(pi.astype(np.int64), device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(pu), generator=g).to(dev)
        tot = 0.
        for s in range(0, len(pu) - batch + 1, batch):
            idx = perm[s:s + batch]
            uu, ii, ww = uu_[idx], ii_[idx], wt[idx]
            lo, n = off[uu][:, None], (off[uu + 1] - off[uu])[:, None]
            pick = (torch.rand(len(idx), L, generator=g).to(dev) * n).long()
            h = lst[(lo + pick).clamp(max=len(lst) - 1)]
            m = (h != ii[:, None]).float()  # leave the target click out of its own history
            pooled = (P[h] * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1) ** alpha * scale
            nn = pool[torch.randint(len(pool), (len(idx) * neg,), generator=g).to(dev)].view(len(idx), neg)
            q, nq = Q[ii], Q[nn]
            sp = (pooled * q).sum(1) + b[ii]
            sn = (nq * pooled[:, None, :]).sum(2) + b[nn]
            loss = (ww * (.5 * F.softplus(-sp) + .5 * F.softplus(sn).mean(1))).sum() / ww.sum()
            reg = pooled.pow(2).sum(1).mean() + q.pow(2).sum(1).mean() + nq.pow(2).sum(2).mean()
            opt.zero_grad(set_to_none=True)
            (loss + l2 * reg).backward()
            torch.nn.utils.clip_grad_norm_([P, Q, b], 5.)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if log:
            print(f"  ep {ep} loss {tot / max(len(pu) // batch, 1):.4f} {time.time() - t0:.0f}s", flush=True)
    return P.detach().cpu().numpy(), Q.detach().cpu().numpy(), b.detach().cpu().numpy(), prop


def pooled_user(Pm, hist, alpha=1., scale=3.):
    """Query-time user vector from supplied history (n,L), -1 padded; duplicates count."""
    m = (hist >= 0)
    v = (Pm[np.maximum(hist, 0)] * m[..., None]).sum(1)
    return (v / np.maximum(m.sum(1, keepdims=True), 1) ** alpha * scale).astype(np.float32)
