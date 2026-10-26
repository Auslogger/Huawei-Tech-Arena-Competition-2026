"""Full-week refit and submission export. Usage: python -m hta.final NAME [seed,seed,...]."""
import csv
import math
import sys

import numpy as np

from . import model
from .lib import CACHE, D, ROOT, RUNS, auc


def write_submission(name, logit):
    """Write the CSV and save the exact predictions used, so a later replay can compare them."""
    keys = np.load(CACHE / "keys.npz")["eval_rowid"]
    p = 1 / (1 + np.exp(-logit.ravel()))
    out = ROOT / "submissions" / f"{name}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rowid", "predicted_probability"])
        w.writerows(zip(keys.tolist(), [f"{v:.10f}" for v in p]))
    np.save(RUNS / f"{name}_final_prob.npy", p)
    return out


def validate(name):
    """Independent of the model: re-read the raw evaluation keys and the written CSV from disk."""
    path = ROOT / "submissions" / f"{name}.csv"
    with open(ROOT / "data/evaluation.csv", newline="", encoding="utf-8") as f:
        want = [r["rowid"] for r in csv.DictReader(f)]
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    assert header == ["rowid", "predicted_probability"], header
    assert all(len(x) == 2 for x in rows), "column count"
    got = [x[0] for x in rows]
    assert len(want) == 529390 == len(got), len(got)
    assert len(set(got)) == len(got) and set(got) == set(want), "rowid mismatch"
    vals = [float(x[1]) for x in rows]
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in vals), "range"
    saved = np.load(RUNS / f"{name}_final_prob.npy")
    order = {k: i for i, k in enumerate(want)}
    replay = np.zeros(len(vals))
    for k, v in zip(got, vals):
        replay[order[k]] = v
    gap = np.abs(replay - saved).max()
    assert gap < 1e-9, gap
    print(f"VALID {path.name}: {len(vals)} rows, min {min(vals):.6f} max {max(vals):.6f} mean {np.mean(vals):.5f} "
          f"distinct {len(set(vals))}, replay gap {gap:.2e}")


def main():
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [113, 211]
    d = D()
    ev = np.arange(d.n_train, d.n)
    tot = 0.
    for seed in seeds:
        p, _, _ = model.run_seed(d, 27, ev, seed)
        np.save(RUNS / f"{name}_eval_s{seed}.npy", p.astype(np.float32))
        tot = tot + p
    write_submission(name, tot / len(seeds))
    validate(name)
    model.ledger(name, kind="final", seeds=seeds)


if __name__ == "__main__":
    main()
