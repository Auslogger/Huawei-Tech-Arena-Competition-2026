"""EASE item-item scorer (Steck 2019) on binary positives known before T: closed form, no user vectors.

B = -P / diag(P) with P = (X'X + lam I)^-1 and a zero diagonal, and score(event, c) is the sum over its known clicks of
B[j, c]. Dense over all 50,000 items, about 30 GB at peak, so run it alone. Scores are gathered directly: truncating B would
zero most rows, because few history and candidate pairs co-occur. Head rows must pass positives known before their own day.
"""

import numpy as np
import scipy.sparse as sp
import torch

from . import states
from .lib import CACHE, NI, NU, THREADS, day_ts

torch.set_num_threads(THREADS)
PATH = CACHE / "ease"


def solve(pu, pi, lam):
    X = sp.csr_matrix((np.ones(len(pu), np.float32), (pu, pi)), shape=(NU, NI))
    X.sum_duplicates()
    X.data[:] = 1
    G = torch.from_numpy((X.T @ X).toarray().astype(np.float32))
    G.diagonal().add_(lam)
    P = torch.linalg.inv(G)  # use LU here, cholesky_inverse fails in float32 on this matrix
    del G
    d = P.diagonal().clone()
    B = -P / d[None, :]
    B.fill_diagonal_(0.0)
    return B


def scores(B, hist, cand, chunk=20000):
    """(n,10) sum of B[h, c] over the events' known clicks. hist is -1 padded."""
    out = []
    Bn = torch.as_tensor(B)
    for i in range(0, len(hist), chunk):
        h = torch.as_tensor(hist[i : i + chunk].astype(np.int64))
        c = torch.as_tensor(cand[i : i + chunk].astype(np.int64))
        m = (h >= 0).float()
        s = Bn[h.clamp(min=0)[:, :, None], c[:, None, :]]  # (n,L,10)
        out.append((s * m[..., None]).sum(1))
    return torch.cat(out).numpy()
