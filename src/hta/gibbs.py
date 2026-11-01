"""Gibbs sampler over unit preference vectors under p(click u,i) proportional to exp(kappa x_u.y_i).

Offers are blind to preference, so given the items a user's vector is von Mises-Fisher with natural parameter
kappa * sum of clicked y, and the same for items. The partition function acts to first order by removing the click-weighted
mean, which is what stops the sweep collapsing onto the degree direction. A sweep is two sparse products. Scoring averages the
choice probability over samples, so it is the posterior predictive, including the correlation of a user and item sharing a click.
"""

import numpy as np
import scipy.sparse as sp
import torch

from .lib import NI, NU


def unit(a):
    return a / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class Chain:
    def __init__(self, pu, pi, w, y0, kappa=4.0, seed=0, device="cpu"):
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.dev = torch.device(device)
        C = sp.csr_matrix((w.astype(np.float32), (pu, pi)), shape=(NU, NI))
        C.sum_duplicates()
        C = C.tocoo()
        self.C = (
            torch.sparse_coo_tensor(np.stack([C.row, C.col]), C.data, (NU, NI)).coalesce().to_sparse_csr().to(self.dev)
        )
        self.Ct = (
            torch.sparse_coo_tensor(np.stack([C.col, C.row]), C.data, (NI, NU)).coalesce().to_sparse_csr().to(self.dev)
        )
        self.y = unit(torch.as_tensor(y0, dtype=torch.float32)).to(self.dev)
        self.p = self.y.shape[1]
        self.kappa = kappa
        self.has = torch.as_tensor(np.bincount(pi, minlength=NI) > 0).to(self.dev)
        wi, wu = np.bincount(pi, weights=w, minlength=NI), np.bincount(pu, weights=w, minlength=NU)
        self.wi, self.wu = [torch.as_tensor(a / a.sum(), dtype=torch.float32).to(self.dev)[:, None] for a in (wi, wu)]
        self.x = None
        self.sweep()

    def noise(self, n):
        return torch.randn(n, self.p, generator=self.g).to(self.dev)

    def draw(self, eta):
        """A projected normal with the tangent variance 1/r of a von Mises-Fisher."""
        r = eta.norm(dim=-1, keepdim=True)
        return unit(eta + (r + 1).sqrt() * self.noise(len(eta)))

    def whiten(self):
        d = self.y[self.has]
        d = d - d.mean(0)
        S = d.T @ d * self.p / len(d)
        lam, Q = torch.linalg.eigh(S.cpu().double())
        self.y = unit(self.y @ ((Q * lam**-0.5) @ Q.T).float().to(self.dev))

    def sweep(self):
        self.eu = self.kappa * (self.C @ (self.y - (self.wi * self.y).sum(0)))
        self.x = self.draw(self.eu)
        self.ei = self.kappa * (self.Ct @ (self.x - (self.wu * self.x).sum(0)))
        self.y = self.draw(self.ei)
        self.whiten()

    def event_x(self, users, extra=None):
        """A fresh draw of the event's user given current items. extra adds evidence the table lacks, e.g. same-day clicks."""
        e = self.kappa * (self.C @ (self.y - (self.wi * self.y).sum(0)))[users]
        if extra is not None:
            e = e + extra
        return self.draw(e)


@torch.no_grad()
def predictive(chain, users, cand, burn=100, samples=100, thin=2, kappa=6.3, c0=-3.4, log=None, hist=None, offset=None):
    """(n,10) log posterior-predictive marginal and (n,10) posterior mean dot.

    offset=(mean, sd) (n,) is a Gaussian belief over each event's utility intercept, sampled along with the vectors.
    """
    u = torch.as_tensor(users.astype(np.int64)).to(chain.dev)
    c = torch.as_tensor(cand.astype(np.int64)).to(chain.dev)
    for s in range(burn):
        chain.sweep()
    pt, dt = torch.zeros(c.shape, device=chain.dev), torch.zeros(c.shape, device=chain.dev)
    for s in range(samples):
        for _ in range(thin):
            chain.sweep()
        extra = None
        if hist is not None:
            h, m = hist
            extra = chain.kappa * (chain.y[h.clamp(min=0)] * m[..., None]).sum(1)
        x = chain.event_x(u, extra)
        dot = (x[:, None, :] * chain.y[c]).sum(2)
        z = kappa * dot + c0
        if offset is not None:
            z = z + (offset[0] + offset[1] * torch.randn(len(u), generator=chain.g).to(chain.dev))[:, None]
        lp = z - torch.logaddexp(torch.zeros_like(z[:, 0]), torch.logsumexp(z, 1))[:, None]
        pt += lp.exp()
        dt += dot
        if log and (s + 1) % 25 == 0:
            log(s + 1, (pt / (s + 1)).log().cpu().numpy(), (dt / (s + 1)).cpu().numpy())
    return (pt / samples).log().cpu().numpy(), (dt / samples).cpu().numpy()
