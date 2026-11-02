"""The slate chains behind C8, at the settings they were run with.

Usage: python -m hta.c8.chains NAME cut[,cut...] seed[,seed...].
Each run writes runs/slate/NAME_c{cut}_s{seed}.npz, the file slate.Inputs reads, and is skipped when that file exists.
A run takes 7 to 15 minutes on an M1 Max GPU, and two can share it.
"""

import sys
import time

import numpy as np

from .. import slate
from ..lib import D, RUNS, auc, day_ts, first_events

# Chain priors and link, the same for both chains
CHAIN = dict(
    kappa=5.92, c0=-3.4, tau=0.7, kd=4.0, kp=4.0, link=1, item_sd=0.2, em=1, coin=1.0, rfix=0.0, ws=1.0, pois=0
)
# Inputs and sampler length
FIT = dict(
    mix="causal", beta=0.5, burn=400, samples=2500, thin=1, click=100, qh=1, cov=3, leaf=2, poly=1, wo=1, lint=1, clk=1
)
SETTINGS = {
    # saturating warm-up link, polynomial share channels, leaf click share, leaf and item intercepts, covariate cells
    "s6": dict(FIT, wck=0, bt0=(0.64, 0.06, 0.43, -0.04, 0.04, 0.98)),
    # same as s6, but each warm-up click's interest term comes from that user's earlier clicks
    "s7": dict(FIT, wck=1, bt0=(0.6, 0.02, 0.39, -0.06, 0.02, 1.1)),
}
# What C8 needs: s6 at every training day of the C7 trees and at the cut, s7 at the cut under four seeds
NEEDED = [("s6", c, s) for s in (113, 211) for c in range(21, 28)] + [("s7", 27, s) for s in (113, 211, 307, 401)]


def path(name, cut, seed):
    return RUNS / "slate" / f"{name}_c{cut}_s{seed}.npz"


def run(d, name, cut, seed):
    p = path(name, cut, seed)
    if p.exists():
        print(f"{p.name} already exists", flush=True)
        return
    o = dict(SETTINGS[name])
    kw = dict(CHAIN, bt0=np.array(o.pop("bt0"), np.float32))
    q = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
    t0 = time.time()

    def tick(n, sc):
        r = sc.out()
        msg = f"  {name} cut {cut} seed {seed}: {n} samples, {time.time() - t0:.0f}s"
        if cut < 27:  # the cut's own query rows are labelled, so report how the chain does alone
            msg += f", AUC {auc(d.y[q], np.log(r['p'][np.searchsorted(r['rows'], q)])):.6f}"
        print(msg, flush=True)

    r, ch = slate.fit(d, cut, seed, tick=(500, tick), **o, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p,
        **r,
        X=ch.X.cpu().numpy(),
        Y=ch.Y.cpu().numpy(),
        b_last=ch.b.cpu().numpy(),
        d_last=ch.d.cpu().numpy(),
        a=ch.a.cpu().numpy(),
        kappa=ch.k,
        bt=ch.bt.cpu().numpy(),
    )
    print(f"wrote {p.name} in {time.time() - t0:.0f}s", flush=True)


def main():
    name, cuts, seeds = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    d = D()
    for seed in seeds:
        for cut in cuts:
            run(d, name, int(cut), int(seed))


if __name__ == "__main__":
    main()
