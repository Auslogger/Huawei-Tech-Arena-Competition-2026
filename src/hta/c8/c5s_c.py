"""C5S_c, the fallback to C8 and C8_s78: C4_refined blended with chain s6. Usage: python -m hta.c8.c5s_c [cut[,cut...]].

C4_refined is the stack of five tables with no chain input at all. Chain s6 under two seeds is averaged by probability
and blended in logit space with a share of .75, fixed before the file was scored.

With no argument this rebuilds submissions/C5S_c.csv from the raw data, skipping work whose output already exists: the
two s6 chains at cut 27, which C8 and C8_s78 use too, then the C4_refined members at cut 27, then the blend. That takes
roughly two hours on an M1 Max. With a cut list it only blends runs that already exist, and an earlier cut prints local
AUCs.
"""

import sys

import numpy as np

from .. import compare, final, parse, portfolio
from ..lib import CACHE, D, auc, check_query, day_ts, first_events
from . import chains

SEEDS = (113, 211)
CHAIN = "s6"
WEIGHT = 0.75  # the chain's share


def predict(cut, ev):
    """(C4_refined, chain, blend) log-odds (n,10) of events ev."""
    st = compare.load("portfolio:C4_refined", cut, list(SEEDS))
    p = 0.0
    for s in SEEDS:
        z = np.load(chains.path(CHAIN, cut, s))
        p = p + z["p"][np.searchsorted(z["rows"], ev)]
    p = np.clip(p / len(SEEDS), 1e-7, 1 - 1e-7)
    S = np.log(p / (1 - p))
    return st, S, (1 - WEIGHT) * st + WEIGHT * S


def build(d, cut=27, name="C5S_c"):
    ev = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
    st, S, bl = predict(cut, ev)
    if cut == 27:
        final.write_submission(name, bl)
        final.validate(name)
        return
    y = d.y[ev]
    print(f"cut {cut}: C4_refined {auc(y, st):.6f} chain {auc(y, S):.6f} blend {auc(y, bl):.6f}", flush=True)


def rebuild():
    if not (CACHE / "events.npz").exists():
        parse.main()
    d = D()
    for s in SEEDS:
        chains.run(d, CHAIN, 27, s)
    ev = np.arange(d.n_train, d.n)
    check_query(d, ev, 27)
    portfolio.guard("C4_refined")
    for m in portfolio.spec("C4_refined")[0]:
        for s in SEEDS:
            portfolio.score(d, "C4_refined", m, 27, s, ev)
            print(f"C4_refined {m} seed {s} done", flush=True)
    build(d)


def main():
    if len(sys.argv) == 1:
        return rebuild()
    d = D()
    for cut in sys.argv[1].split(","):
        build(d, int(cut))


if __name__ == "__main__":
    main()
