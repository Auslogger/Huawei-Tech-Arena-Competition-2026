"""One new prediction, alone and blended into a baseline. Usage: python -m hta.blend NEW BASE cut [seed,seed] [w,w].

NEW and BASE are names as in hta.compare. The blend is (1-w) BASE + w NEW in logit space, and each line goes to the ledger.
"""

import sys

import numpy as np

from . import compare, model
from .lib import D, auc, check_query, day_ts, first_events, paired_gain


def main():
    new, base, cut = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [113]
    ws = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.1, 0.2, 0.3, 0.4]
    d = D()
    q = first_events(d, day_ts(cut), day_ts(27))
    check_query(d, q, cut)
    y, day1 = d.y[q], (d.ts[q] - day_ts(cut)) < 86400
    a, b = compare.load(new, cut, seeds), compare.load(base, cut, seeds)
    g, lo, _ = paired_gain(y, a, b)
    print(
        f"{new} alone {auc(y, a):.6f} day-1 {auc(y[day1], a[day1]):.6f} corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.4f}"
        f" over {base} ({auc(y, b):.6f}) {g:+.6f} q05 {lo:+.6f}",
        flush=True,
    )
    for w in ws:
        m = (1 - w) * b + w * a
        g, lo, hi = paired_gain(y, m, b)
        print(
            f"  w {w}: {g:+.6f} q05 {lo:+.6f} q95 {hi:+.6f} day-1 {auc(y[day1], m[day1]) - auc(y[day1], b[day1]):+.6f}"
            f" gate {g > compare.GATE and lo > 0}",
            flush=True,
        )
        model.ledger(f"blend {new} into {base}", cut=cut, seeds=seeds, w=w, gain=round(g, 6), q05=round(lo, 6))


if __name__ == "__main__":
    main()
