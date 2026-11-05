"""Rebuild C8_s78 or C8 from the raw data in one command: python -m hta.c8 [NAME]. Needs data/train.csv and
data/evaluation.csv.

NAME is C8_s78 (the default, our best file) or C8. Every step skips work whose output already exists, so the command
can be stopped and run again. In order:

1. parse the raw CSVs into cache/ (hta.parse)
2. chain s6 at cuts 21 to 27 for the trees, then the file's own chains at cut 27 (hta.c8.chains), each about ten
   minutes on the GPU
3. the five C7 tree members at cut 27, which read chain s6 at each of their training days (hta.portfolio)
4. the blend, which writes and validates submissions/NAME.csv (hta.c8.blend)

In a single process this takes roughly six hours on an M1 Max. It goes faster split up: two chain jobs at a time for
step 2, then one job per tree member at HTA_THREADS=2 for step 3 (python -m hta.portfolio C7 27 113,211 MEMBER), and
python -m hta.c8.blend NAME to finish. Once one file is built, the other needs only its missing chains and the blend.

The chains sample on the GPU and don't repeat bit for bit, even with the same seed. A rerun differs from the original
about as much as a different seed does, so a rebuilt file comes out close to the uploaded one rather than identical.
"""

import sys

import numpy as np

from .. import parse, portfolio
from ..lib import CACHE, D, check_query
from . import blend, chains


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "C8_s78"
    if not (CACHE / "events.npz").exists():
        parse.main()
    d = D()
    for chain, cut, seed in dict.fromkeys(chains.TREES + [(n, 27, s) for n, s in blend.CHAINS[name]]):
        chains.run(d, chain, cut, seed)
    ev = np.arange(d.n_train, d.n)
    check_query(d, ev, 27)
    portfolio.guard("C7")
    for m in blend.TREES:
        for s in blend.SEEDS:
            portfolio.score(d, "C7", m, 27, s, ev)
            print(f"C7 {m} seed {s} done", flush=True)
    blend.build(d, name=name)


if __name__ == "__main__":
    main()
