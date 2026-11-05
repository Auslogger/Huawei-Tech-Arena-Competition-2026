"""C8 and C8_s78: the C7 trees blended with six slate chains. Usage: python -m hta.c8.blend [NAME] [cut[,cut...]].

The C7 trees take chain s6 in head and tree. They are averaged by logit over members and seeds, the chains by
probability, and the two meet at equal weight in logit space. The two files share the trees and differ only in their
chains. NAME is C8_s78 (the default) or C8. Cut 27 writes and validates submissions/NAME.csv. An earlier cut prints the
local AUC of the trees, the chains and the blend instead, if the runs exist at that cut.
"""

import sys

import numpy as np

from .. import final, portfolio, slate
from ..lib import D, auc, day_ts, first_events

TREES = ("P2", "G2", "B2", "E2", "K2")
SEEDS = (113, 211)
CHAINS = {
    "C8": (("s6", 113), ("s6", 211), ("s7", 113), ("s7", 211), ("s7", 307), ("s7", 401)),
    # the same s7 chains, with s8 in place of s6
    "C8_s78": (("s7", 113), ("s7", 211), ("s7", 307), ("s7", 401), ("s8", 113), ("s8", 211)),
}
WEIGHT = 0.5  # the chains' share


def predict(cut, ev, name="C8_s78"):
    """(trees, chains, blend) log-odds (n,10) of events ev."""
    T = sum(np.load(portfolio.member_path("C7", m, cut, s)) for m in TREES for s in SEEDS) / (len(TREES) * len(SEEDS))
    p = sum(slate.Inputs(n, s).p(cut, ev) for n, s in CHAINS[name]) / len(CHAINS[name])
    S = np.log(p / (1 - p))
    return T, S, (1 - WEIGHT) * T + WEIGHT * S


def build(d, cut=27, name="C8_s78"):
    ev = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
    T, S, bl = predict(cut, ev, name)
    if cut == 27:
        final.write_submission(name, bl)
        final.validate(name)
        return
    y = d.y[ev]
    print(f"{name} cut {cut}: trees {auc(y, T):.6f} chains {auc(y, S):.6f} blend {auc(y, bl):.6f}", flush=True)


def main():
    args = sys.argv[1:]
    name = args.pop(0) if args and args[0] in CHAINS else "C8_s78"
    d = D()
    for cut in args[0].split(",") if args else [27]:
        build(d, int(cut), name)


if __name__ == "__main__":
    main()
