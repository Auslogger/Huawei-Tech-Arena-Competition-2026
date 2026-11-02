"""Leak test for the exposure inputs (groups.Cats) at cut 24. Usage: python scripts/exposure_leak_test.py

The inputs for a row must not move when later events change. This tampers with the outcomes and slates of every event
after the rows being scored, holds the item categories and leaves fixed, rebuilds the inputs and prints the largest
change. Query rows after the cut and training rows from the day before it should both show 0.000000.
"""

import copy

import numpy as np

from hta import groups
from hta.lib import D, day_ts, first_events

d = D()
cut = 24
q = first_events(d, day_ts(cut), day_ts(27))[:20000]
head = np.flatnonzero((d.ts[: d.n_train] >= day_ts(23)) & (d.ts[: d.n_train] < day_ts(24)))[:20000]
f = groups.Cats(d, cut, rich=3)
clean = {"query": f(cut, q), "head": f(23, head)}

# keep the taxonomy fixed, so only the tampered events can change the inputs
groups.categories = lambda *a, **k: f.lab
groups.leaves = lambda *a, **k: f.leaf
rng = np.random.default_rng(0)
for name, ev, day, after in (("query", q, cut, d.ts[q].max()), ("head", head, 23, day_ts(24))):
    d2 = copy.copy(d)
    d2.y, d2.cand = d.y.copy(), d.cand.copy()
    late = (d.ts >= after + 60) & (np.arange(d.n) < d.n_train)
    d2.y[late] = 0
    d2.cand[late] = rng.permutation(d.cand[late].ravel()).reshape(-1, 10)
    moved = np.abs(clean[name] - groups.Cats(d2, cut, rich=3)(day, ev)).max()
    print(f"{name} rows: largest change after tampering with every later event {moved:.6f}", flush=True)
