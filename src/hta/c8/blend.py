"""C8: the C7 trees blended with six slate chains. Usage: python -m hta.c8.blend [cut[,cut...]].

The C7 trees take chain s6 in head and tree. They are averaged by logit over members and seeds, the chains by
probability, and the two meet at equal weight in logit space. Cut 27 writes and validates submissions/C8.csv. An
earlier cut prints the local AUC of the trees, the chains and the blend instead, if the runs exist at that cut.
"""

import sys

import numpy as np

from .. import final, portfolio, slate
from ..lib import D, auc, day_ts, first_events

TREES = ("P2", "G2", "B2", "E2", "K2")
SEEDS = (113, 211)
CHAINS = (("s6", 113), ("s6", 211), ("s7", 113), ("s7", 211), ("s7", 307), ("s7", 401))
WEIGHT = 0.5  # the chains' share


def predict(cut, ev):
    """(trees, chains, blend) log-odds (n,10) of events ev."""
    T = sum(np.load(portfolio.member_path("C7", m, cut, s)) for m in TREES for s in SEEDS) / (len(TREES) * len(SEEDS))
    p = sum(slate.Inputs(n, s).p(cut, ev) for n, s in CHAINS) / len(CHAINS)
    S = np.log(p / (1 - p))
    return T, S, (1 - WEIGHT) * T + WEIGHT * S


def build(d, cut=27, name="C8"):
    ev = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
    T, S, bl = predict(cut, ev)
    if cut == 27:
        final.write_submission(name, bl)
        final.validate(name)
        return
    y = d.y[ev]
    print(f"cut {cut}: trees {auc(y, T):.6f} chains {auc(y, S):.6f} blend {auc(y, bl):.6f}", flush=True)


def main():
    d = D()
    for cut in sys.argv[1].split(",") if len(sys.argv) > 1 else [27]:
        build(d, int(cut))


if __name__ == "__main__":
    main()
