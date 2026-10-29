"""Item table learned with the right likelihood for each click, each click used exactly once.

Warm-up clicks have no slate, so they train through positive/noise. Labelled events train only through the outside-option
slate likelihood, no-click events included. Users are pooled histories that strictly precede the event, so there are no
free user parameters. gamma=0 keeps all positives in the density term, the matched control. Same shape as fism.fit, and
scored by the same Producer branch.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import head, states
from .lib import DAY, NI, NU, T0, auc, leak_gap, norm_incidence, rsvd


def pooled(P, h, alpha, scale):
    m = (h >= 0).float()
    return (P[h.clamp(min=0)] * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1) ** alpha * scale


@torch.no_grad()
def score(params, h, cand, alpha, scale, chunk=20000):
    P, Q, c, a, c0 = params
    return torch.cat([a * (pooled(P, h[i:i + chunk], alpha, scale)[:, None, :] * Q[cand[i:i + chunk]]).sum(2)
                      + c[cand[i:i + chunk]] + c0 for i in range(0, len(h), chunk)])


def fit(d, T, seed=113, gamma=1., k=64, epochs=40, batch=8192, neg=16, lr=.005, wd=.3, l2=.02, L=20, power=.75, alpha=1.,
        scale=3., device="mps", every=10, hdrop=0.):
    """(P, Q, bias, prop) from what is known strictly before T. hdrop randomly hides history items from the choice term."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    pu, pi, pt, w = states.positives(d, T)
    if gamma:  # the choice term owns every labelled click
        keep = pt < T0
        pu, pi, w = pu[keep], pi[keep], w[keep]
        assert pt[keep].max() < T0 and len(pu), "labelled click in the density term"
    mass = np.bincount(pi, weights=w, minlength=NI).astype(np.float64)
    prop = mass ** power / (mass ** power).sum()
    B, _, _ = norm_incidence(pu, pi)
    V0 = rsvd(B, k, seed=seed)[2].T * np.sqrt(NI) * .3
    P = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    Q = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    b = torch.nn.Parameter(torch.zeros(NI, device=dev))
    c = torch.nn.Parameter(torch.zeros(NI, device=dev))
    a = torch.nn.Parameter(torch.ones((), device=dev))
    c0 = torch.nn.Parameter(torch.tensor(-3., device=dev))
    opt = torch.optim.AdamW([{"params": [P, Q, b, c], "weight_decay": wd}, {"params": [a, c0], "weight_decay": 0.}], lr=lr)

    order = np.argsort(pu, kind="stable")
    off = torch.as_tensor(np.r_[0, np.cumsum(np.bincount(pu, minlength=NU))].astype(np.int64), device=dev)
    lst = torch.as_tensor(pi[order].astype(np.int64), device=dev)
    uu_, ii_ = torch.as_tensor(pu.astype(np.int64), device=dev), torch.as_tensor(pi.astype(np.int64), device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    pool = torch.multinomial(torch.as_tensor(prop, dtype=torch.float32), max(len(pu) * neg, 10_000_000), True).to(dev)

    t = d.ts[:d.n_train]
    ev, nxt = np.where(t < T)[0], np.where((t >= T) & (t < T + DAY))[0]
    tens = lambda e, x, ty=np.int64: torch.as_tensor(x[e].astype(ty), device=dev)
    H, C, Y = tens(ev, d.hist_item), tens(ev, d.cand), tens(ev, d.y, np.float32)
    Hn, Cn, Yn = tens(nxt, d.hist_item), tens(nxt, d.cand), tens(nxt, d.y, np.float32)
    params = (P, Q, c if gamma else b, a, c0)
    if gamma and len(nxt):  # the table must not yet hold the labels of the events it is about to train on
        sub = torch.as_tensor(np.random.default_rng(seed).choice(len(ev), min(len(ev), 98000), replace=False), device=dev)
        leak_gap("start table", d.y[ev[sub.cpu().numpy()]], score(params, H[sub], C[sub], alpha, scale).cpu().numpy(),
                 d.y[nxt], score(params, Hn, Cn, alpha, scale).cpu().numpy(), tol=.02)

    per = max(len(pu) // batch, 1)
    ebatch = max(round(batch * len(ev) / len(pu)), 64)  # one pass of events per pass of warm-up clicks
    nw, ne = len(pu), len(ev) if gamma else 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * per, eta_min=.1 * lr)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(pu), device=dev)
        eperm = torch.randperm(len(ev), device=dev) if gamma else None
        dtot, ctot = torch.zeros((), device=dev), torch.zeros((), device=dev)
        for s in range(per):
            idx = perm[s * batch:(s + 1) * batch]
            uu, ii, ww = uu_[idx], ii_[idx], wt[idx]
            lo, n = off[uu][:, None], (off[uu + 1] - off[uu])[:, None]
            h = lst[(lo + (torch.rand(len(idx), L, device=dev) * n).long()).clamp(max=len(lst) - 1)]
            h = h.masked_fill(h == ii[:, None], -1)  # leave the target click out of its own history
            u = pooled(P, h, alpha, scale)
            nn = pool[torch.randint(len(pool), (len(idx) * neg,), device=dev)].view(len(idx), neg)
            q, nq = Q[ii], Q[nn]
            sp = (u * q).sum(1) + b[ii]
            sn = (nq * u[:, None, :]).sum(2) + b[nn]
            dens = (ww * (.5 * F.softplus(-sp) + .5 * F.softplus(sn).mean(1))).sum() / ww.sum()
            loss = nw * (dens + l2 * (u.pow(2).sum(1).mean() + q.pow(2).sum(1).mean() + nq.pow(2).sum(2).mean()))
            if gamma:
                j = eperm[s * ebatch:(s + 1) * ebatch]
                hj = H[j].masked_fill(torch.rand(H[j].shape, device=dev) < hdrop, -1) if hdrop else H[j]
                ue, qe = pooled(P, hj, alpha, scale), Q[C[j]]
                z = a * (ue[:, None, :] * qe).sum(2) + c[C[j]] + c0
                nll = 10 * head.choice_nll(z, Y[j])
                loss = loss + gamma * ne * (nll + l2 * (ue.pow(2).sum(1).mean() + qe.pow(2).sum(2).mean()))
                ctot = ctot + nll.detach()
            opt.zero_grad(set_to_none=True)
            (loss / (nw + gamma * ne)).backward()
            torch.nn.utils.clip_grad_norm_([P, Q, b, c], 5.)
            opt.step()
            sched.step()
            dtot = dtot + dens.detach()
        if (ep + 1) % every == 0 or ep == epochs - 1:
            msg = f"ep {ep} dens {float(dtot) / per:.4f} choice {float(ctot) / per:.4f}"
            if len(nxt):
                z = score(params, Hn, Cn, alpha, scale)
                msg += f" next-day nll {10 * float(head.choice_nll(z, Yn)):.4f} auc {auc(d.y[nxt], z.cpu().numpy()):.4f}"
            print(f"  {msg} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        return (P * a).cpu().numpy(), Q.cpu().numpy(), (c if gamma else b).cpu().numpy(), prop
