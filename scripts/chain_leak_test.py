"""Leak test for the slate chain. Usage: python scripts/chain_leak_test.py cut [samples]

Fits chain s7 twice at the cut, once on the real data and once with every outcome after the cut randomised and every
later history scrambled, then scores both against the true labels of the cut's query rows. A chain that read anything
from after the cut would lose AUC on the scrambled data, so the two scores should match. Runs don't repeat bit for bit
on the GPU, so compare the pair from one run rather than against a saved score. At cut 26 with the default 1,500 samples
we got .8406 clean and .840798 scrambled. Each fit takes about ten minutes on an M1 Max.
"""

import sys

import numpy as np

from hta import slate
from hta.c8 import chains
from hta.lib import D, auc, day_ts, first_events

cut = int(sys.argv[1])
samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
o = dict(chains.SETTINGS["s7"], samples=samples)
kw = dict(chains.CHAIN, bt0=np.array(o.pop("bt0"), np.float32))
d = D()
T = day_ts(cut)
q = first_events(d, T, day_ts(27))
y = d.y[q].copy()


def score():
    r, _ = slate.fit(d, cut, 113, log=lambda s: None, **o, **kw)
    return auc(y, r["p"][np.searchsorted(r["rows"], q)])


print(f"cut {cut}, {samples} samples, real data: {score():.6f}", flush=True)
# every labelled event after the cut gets a random outcome, one click or none
rng = np.random.default_rng(0)
late = np.flatnonzero(d.ts[: d.n_train] >= T)
pick = rng.integers(0, 11, len(late))
d.y[late] = 0
hit = pick < 10
d.y[late[hit], pick[hit]] = 1
# and every later history is scrambled, apart from the query rows' own, which only hold clicks before the cut
other = np.setdiff1d(late, q)
mask = d.hist_item[other] >= 0
d.hist_item[other] = np.where(mask, rng.integers(0, 50000, mask.shape), -1)
print(f"cut {cut}, {samples} samples, everything after the cut randomised: {score():.6f}", flush=True)
