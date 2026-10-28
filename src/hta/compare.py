"""Matched gain between saved predictions, the tool behind every gate decision.

Usage: python -m hta.compare NEW BASE cut[,cut...] [seed,seed]. Seeds are averaged before any gain is computed.
NAME is a run (runs/NAME_c{cut}_s{seed}.npy) or portfolio:RUN (its members at frozen weights), portfolio:RUN:eq (equal
weights) or portfolio:RUN:M (member M alone).
"""
import json
import sys

import numpy as np

from . import model, portfolio
from .lib import D, RUNS, auc, check_query, day_ts, first_events, paired_gain

GATE = .003


def load(name, cut, seeds):
    if not name.startswith("portfolio:"):
        return sum(np.load(RUNS / f"{name}_c{cut}_s{s}.npy") for s in seeds) / len(seeds)
    _, run, *opt = name.split(":")
    w = json.loads((RUNS / "members" / f"{run}.json").read_text())["weights"]
    if opt:
        w = {m: 1 / len(w) for m in w} if opt[0] == "eq" else {opt[0]: 1.}
    return sum(v * sum(np.load(portfolio.member_path(run, m, cut, s)) for s in seeds) / len(seeds) for m, v in w.items())


def compare(d, new, base, cut, seeds):
    ev = first_events(d, day_ts(cut), day_ts(27))
    check_query(d, ev, cut)
    a, b, y = load(new, cut, seeds), load(base, cut, seeds), d.y[ev]
    delay = (d.ts[ev] - day_ts(cut)) / 3600.
    day1 = delay < 24
    gain, lo, hi = paired_gain(y, a, b)
    g1, lo1, _ = paired_gain(y[day1], a[day1], b[day1])
    r = {"new": new, "base": base, "cut": cut, "seeds": seeds, "all": round(auc(y, a), 6), "all_base": round(auc(y, b), 6),
         "gain": round(gain, 6), "q05": round(lo, 6), "q95": round(hi, 6), "day1_gain": round(g1, 6),
         "day1_q05": round(lo1, 6), "clears_gate": bool(gain > GATE and lo > 0)}
    for a_h, b_h in [(24, 48), (48, 96), (96, 200)]:
        m = (delay >= a_h) & (delay < b_h)
        if m.sum() > 500:
            r[f"gain{a_h}-{b_h}"] = round(auc(y[m], a[m]) - auc(y[m], b[m]), 5)
    print(r, flush=True)
    model.ledger(f"compare {new} vs {base}", **r)
    return r


def main():
    new, base, cuts = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [113, 211]
    d = D()
    for cut in cuts:
        compare(d, new, base, cut, seeds)


if __name__ == "__main__":
    main()
