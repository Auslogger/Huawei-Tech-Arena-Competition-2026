"""Item correction fit after the stack on shown rows, a convex ridge in the supervised-view role S was meant to fill.

Every training row was scored by tables fitted before its own day, so its label reached neither its user vector nor its
offset. Only item parameters are free: r = u @ dQ[item] + dc[item], with fixed standardised users and Gaussian priors.
Usage: python -m hta.correct NAME cut seed[,seed] tau:tau_c[,tau:tau_c] [run_seed key=value...].
Writes runs/NAME_t{tau}_{tau_c}_c{cut}_s{seed}.npy.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import model
from .lib import D, NI, RUNS, auc, check_query, day_ts, first_events, leak_gap


def fit(u, cand, y, off, tau=0.05, tau_c=0.05, iters=150, init=None, log=print):
    """(dQ, dc) minimising summed row BCE over offset log-odds, plus ||dQ||^2/2tau^2 + ||dc||^2/2tau_c^2.

    Float64 on CPU and an unnormalised objective: a per-row mean in float32 stops L-BFGS within a few steps.
    """
    ut, c, yy, o = [
        torch.as_tensor(a.astype(t)) for a, t in ((u, np.float64), (cand, np.int64), (y, np.float64), (off, np.float64))
    ]
    k = u.shape[1]
    dQ = torch.nn.Parameter(torch.zeros(NI, k, dtype=torch.float64) if init is None else init[0].clone())
    dc = torch.nn.Parameter(torch.zeros(NI, dtype=torch.float64) if init is None else init[1].clone())
    opt = torch.optim.LBFGS(
        [dQ, dc],
        lr=1,
        max_iter=iters,
        max_eval=iters * 2,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-3,
        tolerance_change=1e-4,
    )

    def closure():
        opt.zero_grad()
        r = (ut[:, None, :] * dQ[c]).sum(2) + dc[c]
        loss = F.binary_cross_entropy_with_logits(o + r, yy, reduction="sum")
        loss = loss + dQ.pow(2).sum() / (2 * tau**2) + dc.pow(2).sum() / (2 * tau_c**2)
        loss.backward()
        return loss

    opt.step(closure)
    st = opt.state_dict()["state"][0]
    log(f"    lbfgs iters {st['n_iter']} evals {st['func_evals']} max grad {float(dQ.grad.abs().max()):.2e}")
    return dQ.detach(), dc.detach()


def apply(st, u, cand):
    dQ, dc = st[0].cpu().numpy(), st[1].cpu().numpy()
    return (u[:, None, :] * dQ[cand]).sum(2) + dc[cand]


def main():
    name, cut = sys.argv[1], int(sys.argv[2])
    seeds = [int(x) for x in sys.argv[3].split(",")]
    taus = [tuple(float(v) for v in x.split(":")) for x in sys.argv[4].split(",")]
    opts = {k: json.loads(v) for k, v in (a.split("=") for a in sys.argv[5:])}
    d = D()
    q = first_events(d, day_ts(cut), day_ts(27))
    check_query(d, q, cut)
    y = d.y[q]
    for seed in seeds:
        o = {}
        stack, head_q, _ = model.run_seed(d, cut, q, seed, train_out=o, **opts)
        ev, cand, yt = o["ev"], d.cand[o["ev"]], d.y[o["ev"]]
        leak_gap("head logits", yt, o["head"], y, head_q)
        leak_gap("stacked logits", yt, o["logit"], y, stack)
        print(f"seed {seed}: head {auc(y, head_q):.6f} stack {auc(y, stack):.6f}", flush=True)
        for level, off, base in (("head", o["head"], head_q), ("stack", o["logit"], stack)):
            for tau, tau_c in taus:
                t0 = time.time()
                st = fit(o["su"], cand, yt, off, tau=tau, tau_c=tau_c)
                pred = base + apply(st, o["suq"], d.cand[q])
                print(
                    f"  {level} tau {tau} tau_c {tau_c}: {auc(y, pred):.6f} gain {auc(y, pred) - auc(y, base):+.6f} "
                    f"|dQ| {float(st[0].norm(dim=1).mean()):.4f} |dc| {float(st[1].std()):.4f} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
                if level == "stack":
                    np.save(RUNS / f"{name}_t{tau}_{tau_c}_c{cut}_s{seed}.npy", pred.astype(np.float32))


if __name__ == "__main__":
    main()
