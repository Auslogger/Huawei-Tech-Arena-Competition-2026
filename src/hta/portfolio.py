"""Frozen multi-view portfolio. Usage: python -m hta.portfolio NAME cut[,cut...] [seed,seed] (cut 27 writes the CSV)."""
import json
import sys
from importlib import import_module

import numpy as np

from . import final, model, pipe
from .lib import D, RUNS, auc, check_query, day_ts, first_events

A = dict(pipe.CFG)
MEMBERS = {
    "A": A,
    "P": {**A, "power": 1., "beta": 3.},
    "B": {**A, "iota": .5, "own_noise": 1, "mix": .5, "slate_refs": 1},
    "E": {**A, "cat_noise": 1, "cat_power": 1.},  # noise from the exposure model; see groups.py
}
# Tables whose noise removes exposure hold nothing useful beyond the preference block. In whitened coordinates the head's ridge
# is about sixteen times harder, so it is rescaled.
MEMBERS["B8"] = {**MEMBERS["B"], "proj": 8, "lam": .00625}
MEMBERS["E8"] = {**MEMBERS["E"], "proj": 8, "lam": .00625}
MEMBERS["K16"] = {**MEMBERS["E"], "k": 16, "lam": .1 * (16 / 64) ** 2}  # the same model fitted directly at rank 16
MEMBERS["G"] = {**A, "batch": 32768, "epochs": 160, "graph": 1, "layers": 2, "wd": .6}  # smoothed over the positive graph
MEMBERS["E8s"] = {**MEMBERS["E8"], "swa": 20}  # mean of the last 20 epochs
PORTFOLIOS = {  # name: (weights, model.run_seed options)
    "C2": ({"P": .4, "A": .35, "B": .25}, {}),
    "C3": ({"P": 1 / 3, "A": 1 / 3, "B": 1 / 3}, dict(day_std=2, oof=1, support=1)),
    "C4": ({"P": 1 / 6, "A": 1 / 6, "B8": 1 / 3, "E8": 1 / 3}, dict(day_std=2, oof=1, support=1, cats=3)),  # three noise families
}


def member_path(name, member, cut, seed):
    return RUNS / "members" / f"{name}_{member}_c{cut}_s{seed}.npy"


def spec(name):
    return PORTFOLIOS[name] if name in PORTFOLIOS else PORTFOLIOS[name.split("_")[0]]


def guard(name):
    """Cached members are keyed by run name, so a run name stays bound to the config it was first written with."""
    weights, opts = spec(name)
    p = RUNS / "members" / f"{name}.json"
    want = {"members": {m: MEMBERS[m] for m in weights}, "weights": weights, **({"opts": opts} if opts else {})}
    if p.exists():
        assert json.loads(p.read_text()) == want, f"{name} was written with another config"
    else:
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(json.dumps(want))


def run_member(d, name, member, cut, seed, ev):
    cfg = dict(MEMBERS[member])
    kind = cfg.pop("kind", None)
    if kind is None:
        own = {k: cfg.pop(k) for k in ("lam", "prx") if k in cfg}
        return model.run_seed(d, cut, ev, seed, cfg=cfg, **spec(name)[1], **own)[0]
    cfg.pop("seedless", None)
    return import_module(f"hta.{kind}").member(d, cut, ev, seed, **cfg)


def score(d, name, member, cut, seed, ev):
    p = member_path(name, member, cut, seed)
    if p.exists():
        return np.load(p)
    out = run_member(d, name, member, cut, seed, ev)
    p.parent.mkdir(exist_ok=True, parents=True)
    np.save(p, out.astype(np.float32))
    return out


def main():
    name, cuts = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [113, 211]
    only = sys.argv[4] if len(sys.argv) > 4 else None
    weights = spec(name)[0]
    guard(name)
    d = D()
    for cut in cuts:
        ev = np.arange(d.n_train, d.n) if cut == 27 else first_events(d, day_ts(cut), day_ts(27))
        check_query(d, ev, cut)
        if only:
            for s in seeds:
                score(d, name, only, cut, s, ev)
            continue
        per = {m: sum(score(d, name, m, cut, s, ev) for s in seeds) / len(seeds) for m in weights}
        blend = sum(weights[m] * per[m] for m in weights)
        if cut == 27:
            final.write_submission(name, blend)
            final.validate(name)
            model.ledger(name, kind="final", seeds=seeds, weights=weights)
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
