"""Fixed-candidate transfer checks. Usage: python -m hta.transfer NAME cut[,cut...] [seed,seed] [run_seed key=value...]."""
import json
import sys

import numpy as np

from . import model
from .lib import D, RUNS, auc, check_query, day_ts, first_events


def main():
    name, cuts = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [113, 211]
    opts = {k: json.loads(v) for k, v in (a.split("=") for a in sys.argv[4:])}
    d = D()
    for cut in cuts:
        T = day_ts(cut)
        q = first_events(d, T, day_ts(27))  # every user's first event after the cutoff, all genuine delays
        check_query(d, q, cut)
        delay = (d.ts[q] - T) / 3600.
        day1 = delay < 24
        s, h = 0., 0.
        for i, seed in enumerate(seeds):
            p, lq, _ = model.run_seed(d, cut, q, seed, **opts)
            np.save(RUNS / f"{name}_c{cut}_s{seed}.npy", p.astype(np.float32))
            s, h = s + p, h + lq
            r = {"cut": cut, "n_seeds": i + 1, "day1_events": int(day1.sum()),
                 "day1": round(auc(d.y[q][day1], s[day1]), 6), "all_delay": round(auc(d.y[q], s), 6),
                 "day1_head": round(auc(d.y[q][day1], h[day1]), 6), "all_events": len(q)}
            for lo, hi in [(24, 48), (48, 96), (96, 200)]:
                m = (delay >= lo) & (delay < hi)
                if m.sum() > 500:
                    r[f"d{lo}-{hi}"] = round(auc(d.y[q][m], s[m]), 5)
            print(name, r, flush=True)
            model.ledger(name, **r)


if __name__ == "__main__":
    main()
