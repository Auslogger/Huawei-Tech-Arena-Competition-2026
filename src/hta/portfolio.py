"""Frozen multi-view portfolio. Usage: python -m hta.portfolio NAME cut[,cut...] [seed,seed] (cut 27 writes the CSV)."""
import json
import sys

import numpy as np

from . import final, model, pipe
from .lib import D, RUNS, auc, check_query, day_ts, first_events

A = dict(pipe.CFG)
MEMBERS = {
    "A": A,
    "P": {**A, "power": 1., "beta": 3.},
    "B": {**A, "iota": .5, "own_noise": 1, "mix": .5, "slate_refs": 1},
}
WEIGHTS = {"P": .4, "A": .35, "B": .25}


def member_path(name, member, cut, seed):
    return RUNS / "members" / f"{name}_{member}_c{cut}_s{seed}.npy"


def guard(name):
    """Cached members are keyed by run name, so a run name stays bound to the config it was first written with."""
    p = RUNS / "members" / f"{name}.json"
    spec = {"members": {m: MEMBERS[m] for m in WEIGHTS}, "weights": WEIGHTS}
    if p.exists():
        assert json.loads(p.read_text()) == spec, f"{name} was written with another config"
    else:
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(json.dumps(spec))


def score(d, name, member, cut, seed, ev):
    p = member_path(name, member, cut, seed)
    if p.exists():
        return np.load(p)
    out, _, _ = model.run_seed(d, cut, ev, seed, cfg=MEMBERS[member])
    p.parent.mkdir(exist_ok=True, parents=True)
    np.save(p, out.astype(np.float32))
    return out


def main():
    name, cuts = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [113, 211]
    guard(name)
    d = D()
    for cut in cuts:
        ev = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
        check_query(d, ev, cut)
        per = {m: sum(score(d, name, m, cut, s, ev) for s in seeds) / len(seeds) for m in WEIGHTS}
        blend = sum(WEIGHTS[m] * per[m] for m in WEIGHTS)
        if cut == 27:
            final.write_submission(name, blend)
            final.validate(name)
            model.ledger(name, kind="final", seeds=seeds, weights=WEIGHTS)
            continue
        delay = (d.ts[ev] - day_ts(cut)) / 3600.
        day1 = delay < 24
        r = {"cut": cut, "seeds": seeds, "day1": round(auc(d.y[ev][day1], blend[day1]), 6),
             "all_delay": round(auc(d.y[ev], blend), 6)}
        r["members_day1"] = {m: round(auc(d.y[ev][day1], per[m][day1]), 5) for m in per}
        print(name, r, flush=True)
        model.ledger(name, **r)


if __name__ == "__main__":
    main()
