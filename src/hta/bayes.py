"""Posterior-predictive click probability under the sphere model, instead of a plug-in dot with one global slope.

Each node has a von Mises-Fisher belief (natural parameter eta). A row's marginal is the expected choice probability over
the beliefs of the user and of all ten candidates, so certain pairs score steeply and uncertain ones keep the convex uplift.
"""

import numpy as np
import torch


@torch.no_grad()
def predictive(
    eta_u, eta_i, cand, kappa=6.3, c0=-3.4, samples=64, seed=0, chunk=4096, boost_u=1.0, boost_i=1.0, device="cpu"
):
    """eta_u (n,p), eta_i (items,p), cand (n,10) -> (log marginal, log mean utility) (n,10). A projected normal stands in for vMF."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Eu, Ei = torch.as_tensor(eta_u * boost_u).to(device), torch.as_tensor(eta_i * boost_i).to(device)
    c = torch.as_tensor(cand.astype(np.int64)).to(device)
    out, sd, ev = torch.zeros(len(c), 10), torch.zeros(len(c), 10), torch.zeros(len(c))
    for i in range(0, len(c), chunk):
        cc, eu = c[i : i + chunk], Eu[i : i + chunk]
        n, p = len(cc), eu.shape[1]
        x = eu[None] + (eu.norm(dim=-1, keepdim=True) + 1).sqrt()[None] * torch.randn(samples, n, p, generator=g).to(
            device
        )
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ec = Ei[cc]
        y = ec[None] + (ec.norm(dim=-1, keepdim=True) + 1).sqrt()[None] * torch.randn(
            samples, n, 10, p, generator=g
        ).to(device)
        y = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        z = kappa * (x[:, :, None, :] * y).sum(-1) + c0
        den = torch.logaddexp(torch.zeros_like(z[..., 0]), torch.logsumexp(z, -1))
        lp = z - den[..., None]
        out[i : i + chunk] = (torch.logsumexp(lp, 0) - np.log(samples)).cpu()
        sd[i : i + chunk] = z.std(0).cpu()
        ev[i : i + chunk] = (-den).exp().mean(0).cpu()  # probability of no click
    return out.numpy(), sd.numpy(), ev.numpy()


def event_eta(d, ev, day_t, eu, ei, kappa):
    """A user's belief at an event: clicks known at the table's cutoff, plus same-day clicks in the event's own history."""
    from . import bp

    h, ht = d.hist_item[ev], d.hist_ts[ev]
    new = (h >= 0) & (ht >= day_t)
    y = bp.mean_of(ei)
    return eu[d.user[ev]] + kappa * (y[np.maximum(h, 0)] * new[..., None]).sum(1)


def extras(d, cfg, seed=113, kmsg=4.5, sweeps=2, kappa=6.3, c0=-3.4, boost=0.5, top=8):
    """model.run_seed extra_head: predictive log marginal, posterior mean dot and the two log concentrations, per scored batch."""
    from . import bp, pipe, states
    from .lib import NI, NU, day_ts

    cache = {}

    def table(day):
        if day not in cache:
            V = pipe.state(d, day_ts(day), seed, **cfg)[1]
            pu, pi, _, w = states.positives(d, day_ts(day))
            Vt = np.linalg.svd(V, full_matrices=False)[2][:top].T
            cache[day] = bp.run(pu, pi, w, V @ Vt, NU, NI, kappa=kmsg, sweeps=sweeps)[:2]
        return cache[day]

    def f(day, ev, _):
        eu, ei = table(day)
        e = event_eta(d, ev, day_ts(day), eu, ei, kmsg)
        lp, sd, none = predictive(e, ei, d.cand[ev], kappa=kappa, c0=c0, boost_u=boost, boost_i=boost)
        dot = (bp.mean_of(e)[:, None, :] * bp.mean_of(ei)[d.cand[ev]]).sum(2)
        ru = np.repeat(np.log1p(np.linalg.norm(e, axis=1))[:, None], 10, 1)
        ri = np.log1p(np.linalg.norm(ei, axis=1))[d.cand[ev]]
        x = np.stack([lp, dot, ru, ri, sd, np.repeat(np.log(none)[:, None], 10, 1)], -1).astype(np.float32)
        return ((x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-9)).astype(np.float32)

    return f
