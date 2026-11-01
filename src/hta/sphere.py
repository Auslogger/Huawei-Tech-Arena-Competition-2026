"""Log-bilinear table with the simulator's geometry: a unit preference block at a fixed temperature, plus a free block.

Well-supported items are uniform on the 8-dim unit sphere, and kappa near 6.3 on that sphere reproduces every repeat-CTR
moment. So s(u,i) = tau x_u.y_i + p_u.q_i + b_i, with x, y unit vectors and (p, q) a small free block for exposure.
A unit vector has no norm to regularise, so only the free block carries a penalty.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .lib import NI, NU, norm_incidence, rsvd


def fit(
    pu,
    pi,
    w,
    tau=6.3,
    k1=8,
    k2=8,
    epochs=80,
    batch=8192,
    neg=16,
    lr=0.005,
    wd=0.3,
    l2=0.02,
    power=0.75,
    seed=113,
    device="mps",
    every=20,
    expo=None,
    mix=1.0,
    init=None,
    log=print,
):
    """(X, Y, P, Q, b, prop). X, Y unit (n,k1). expo=(offsets, items): a 1-mix share of noise comes from the user's own offers."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    mass = np.bincount(pi, weights=w, minlength=NI).astype(np.float64)
    prop = mass**power / (mass**power).sum()
    if init is None:
        B, _, _ = norm_incidence(pu, pi)
        V0 = rsvd(B, k1 + k2 + 1, seed=seed)[2].T[:, 1:] * np.sqrt(NI) * 0.3  # first direction is the degree
        cnt = np.maximum(np.bincount(pu, weights=w, minlength=NU), 1)
        U0 = np.zeros((NU, k1 + k2))
        np.add.at(U0, pu, V0[pi] * w[:, None])
        U0 = U0 / cnt[:, None] * 3
    else:
        U0, V0 = init
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    X = torch.nn.Parameter(t(U0[:, :k1]))
    Y = torch.nn.Parameter(t(V0[:, :k1]))
    P = torch.nn.Parameter(t(U0[:, k1:]))
    Q = torch.nn.Parameter(t(V0[:, k1:]))
    b = torch.nn.Parameter(torch.zeros(NI, device=dev))
    opt = torch.optim.AdamW([{"params": [X, Y], "weight_decay": 0.0}, {"params": [P, Q, b], "weight_decay": wd}], lr=lr)
    per = max(len(pu) // batch, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * per, eta_min=0.1 * lr)
    pool = torch.multinomial(
        torch.as_tensor(prop, dtype=torch.float32), min(50_000_000, max(len(pu) * neg, 1_000_000)), True
    ).to(dev)
    uu_, ii_ = [torch.as_tensor(a.astype(np.int64), device=dev) for a in (pu, pi)]
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    if expo is not None:
        off, lst = [torch.as_tensor(a.astype(np.int64), device=dev) for a in expo]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(pu), device=dev)
        tot = torch.zeros((), device=dev)
        for s in range(per):
            idx = perm[s * batch : (s + 1) * batch]
            uu, ii, ww = uu_[idx], ii_[idx], wt[idx]
            nn = pool[torch.randint(len(pool), (len(idx) * neg,), device=dev)].view(len(idx), neg)
            if expo is not None:
                lo, n = off[uu][:, None], (off[uu + 1] - off[uu])[:, None]
                own = lst[(lo + (torch.rand(len(idx), neg, device=dev) * n).long()).clamp(max=len(lst) - 1)]
                nn = torch.where((n > 0) & (torch.rand(len(idx), neg, device=dev) >= mix), own, nn)
            xe, ye, ne = F.normalize(X[uu], dim=-1), F.normalize(Y[ii], dim=-1), F.normalize(Y[nn], dim=-1)
            pe, qe, nq = P[uu], Q[ii], Q[nn]
            sp = tau * (xe * ye).sum(1) + (pe * qe).sum(1) + b[ii]
            sn = tau * (ne * xe[:, None, :]).sum(2) + (nq * pe[:, None, :]).sum(2) + b[nn]
            loss = (ww * (0.5 * F.softplus(-sp) + 0.5 * F.softplus(sn).mean(1))).sum() / ww.sum()
            reg = pe.pow(2).sum(1).mean() + qe.pow(2).sum(1).mean() + nq.pow(2).sum(2).mean()
            opt.zero_grad(set_to_none=True)
            (loss + l2 * reg).backward()
            torch.nn.utils.clip_grad_norm_([X, Y, P, Q, b], 5.0)
            opt.step()
            sched.step()
            tot = tot + loss.detach()
        if every and ((ep + 1) % every == 0 or ep == epochs - 1):
            log(f"  ep {ep} loss {float(tot) / per:.4f} {time.time() - t0:.0f}s")
    with torch.no_grad():
        return (
            F.normalize(X, dim=-1).cpu().numpy(),
            F.normalize(Y, dim=-1).cpu().numpy(),
            P.cpu().numpy(),
            Q.cpu().numpy(),
            b.cpu().numpy(),
            prop,
        )


@torch.no_grad()
def user_map(
    X0,
    Y,
    hist,
    tau,
    weight=20.0,
):
    """Direction of a user from their supplied history and their fitted vector: the exponential-link sufficient statistic."""
    h = torch.as_tensor(hist.astype(np.int64))
    m = (h >= 0).float()
    s = (torch.as_tensor(Y)[h.clamp(min=0)] * m[..., None]).sum(1) + weight * torch.as_tensor(X0)
    return F.normalize(s, dim=-1).numpy()
