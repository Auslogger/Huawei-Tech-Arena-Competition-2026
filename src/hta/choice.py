"""Item table learned from shown slates and their outcomes: the outside-option likelihood on every labelled event.

There are no free user parameters: a user is the pooled representation of a history that strictly precedes the event, so
labels reach item parameters only. The result has the shape of fism.fit and is scored by the same Producer branch.
"""
import time

import numpy as np
import torch

from . import head
from .lib import DAY, NI, auc


def pooled(P, h, alpha, scale):
    m = (h >= 0).float()
    return (P[h.clamp(min=0)] * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1) ** alpha * scale


def utility(P, Q, c, c0, A, h, cand, alpha, scale):
    return ((pooled(P, h, alpha, scale) @ A)[:, None, :] * Q[cand]).sum(2) + c[cand] + c0


@torch.no_grad()
def holdout(d, ev, params, alpha, scale, dev, chunk=20000):
    """Event NLL and raw row AUC of the next day's events, a memorisation diagnostic that never feeds the fit."""
    h, cand = [torch.as_tensor(a.astype(np.int64), device=dev) for a in (d.hist_item[ev], d.cand[ev])]
    z = torch.cat([utility(*params, h[i:i + chunk], cand[i:i + chunk], alpha, scale) for i in range(0, len(ev), chunk)])
    y = torch.as_tensor(d.y[ev].astype(np.float32), device=dev)
    return 10 * float(head.choice_nll(z, y)), auc(d.y[ev], z.cpu().numpy())


def fit(d, T, V0, b0, seed=113, epochs=4, batch=4096, lr=.002, wd=.1, l2=.02, alpha=.5, scale=3., device="mps"):
    """(P, Q, c, prop) from labelled events strictly before T, warm-started from an NCE item table of the same cutoff."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(device)
    t = d.ts[:d.n_train]
    ev, nxt = np.where(t < T)[0], np.where((t >= T) & (t < T + DAY))[0]
    h = torch.as_tensor(d.hist_item[ev].astype(np.int64), device=dev)
    cand = torch.as_tensor(d.cand[ev].astype(np.int64), device=dev)
    y = torch.as_tensor(d.y[ev].astype(np.float32), device=dev)
    P = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    Q = torch.nn.Parameter(torch.tensor(V0, dtype=torch.float32, device=dev))
    c = torch.nn.Parameter(torch.tensor(b0, dtype=torch.float32, device=dev))
    A = torch.nn.Parameter(torch.eye(V0.shape[1], device=dev) * .5)
    with torch.no_grad():  # start at the observed click mass: at this lr the offset could not get there itself
        z0 = utility(P, Q, c, 0., A, h[:20000], cand[:20000], alpha, scale)
    c0 = torch.nn.Parameter(-1.77 - z0.mean())
    params = [P, Q, c, c0, A]
    opt = torch.optim.AdamW([{"params": [P, Q, c, A], "weight_decay": wd}, {"params": [c0], "weight_decay": 0.}], lr=lr)
    per = len(ev) // batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * per, eta_min=.1 * lr)
    t0 = time.time()
    for ep in range(epochs):
        order = torch.randperm(len(ev), generator=g).to(dev)
        tot = 0.
        for s in range(0, per * batch, batch):
            i = order[s:s + batch]
            u = pooled(P, h[i], alpha, scale)
            q = Q[cand[i]]
            z = ((u @ A)[:, None, :] * q).sum(2) + c[cand[i]] + c0
            nll = 10 * head.choice_nll(z, y[i])
            reg = u.pow(2).sum(1).mean() + q.pow(2).sum(2).mean()
            opt.zero_grad(set_to_none=True)
            (nll + l2 * reg).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.)
            opt.step()
            sched.step()
            tot += float(nll.detach())
        next_day = holdout(d, nxt, params, alpha, scale, dev) if len(nxt) else ()
        print(f"  ep {ep} nll {tot / per:.5f} next-day nll, raw auc {next_day} {time.time() - t0:.0f}s", flush=True)
    return (P.detach() @ A.detach()).cpu().numpy(), Q.detach().cpu().numpy(), c.detach().cpu().numpy(), np.ones(NI) / NI
