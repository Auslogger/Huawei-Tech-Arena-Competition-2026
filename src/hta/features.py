"""Label-derived causal counters. Each row only sees events strictly earlier than its own timestamp."""

import numpy as np

from . import states
from .lib import CACHE, NI, NU, day_ts


def _prior(keys, t, val):
    """For rows grouped by `keys`, count and sum `val` over strictly earlier timestamps in the same group."""
    order = np.lexsort((t, keys))
    k, ts, v = keys[order], t[order], val[order]
    n = len(k)
    new_group = np.r_[True, k[1:] != k[:-1]]
    new_tie = new_group | np.r_[True, ts[1:] != ts[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    tie_start = np.maximum.accumulate(np.where(new_tie, np.arange(n), 0))
    cv = np.r_[0, np.cumsum(v)]
    cnt, s = np.zeros(n, np.int32), np.zeros(n, np.int32)
    cnt[order] = tie_start - group_start
    s[order] = cv[tie_start] - cv[group_start]
    return cnt, s


def pair_counts(d):
    """(n_offer, n_click) per candidate row and (u_events, u_clicked_events) per event. Cached."""
    p = CACHE / "counts.npz"
    if p.exists():
        z = np.load(p)
        return z["n_offer"], z["n_click"], z["u_events"], z["u_clicks"]
    t = np.repeat(d.ts, 10).astype(np.int64)
    key = np.repeat(d.user, 10).astype(np.int64) * NI + d.cand.ravel()
    lab = np.maximum(d.y, 0).ravel().astype(np.int64)
    known = (d.y >= 0).ravel().astype(np.int64)
    # evaluation rows have no labels, so they get zero value and no count
    n_off, n_clk = _prior(key, t, lab)
    n_seen = _prior(key, t, known)[1]
    n_offer, n_click = n_seen.reshape(-1, 10).astype(np.int16), n_clk.reshape(-1, 10).astype(np.int16)
    u_ev, u_ck = _prior(d.user.astype(np.int64), d.ts.astype(np.int64), np.maximum(d.y, 0).max(1).astype(np.int64))
    known_ev = (d.y[:, 0] >= 0).astype(np.int64)
    u_events = _prior(d.user.astype(np.int64), d.ts.astype(np.int64), known_ev)[1]
    np.savez(p, n_offer=n_offer, n_click=n_click, u_events=u_events.astype(np.int32), u_clicks=u_ck.astype(np.int32))
    return n_offer, n_click, u_events, u_ck


def item_day_counts(d):
    """rows[day,item], clicks[day,item] of labelled rows on Jan 20+day."""
    from .lib import DAY, T0

    day = (d.ts[: d.n_train] - T0) // DAY
    n = np.zeros((7, NI), np.int32)
    c = np.zeros((7, NI), np.int32)
    for k in range(7):
        ev = np.where(day == k)[0]
        n[k] = np.bincount(d.cand[ev].ravel(), minlength=NI)
        c[k] = np.bincount(d.cand[ev].ravel(), weights=d.y[ev].ravel(), minlength=NI)
    return n, c


def scalars(d, ev, cut_day=None):
    """Per-candidate causal scalars for events `ev`. cut_day: Jan day whose start bounds global item statistics
    (defaults to each event's own day). User and pair counters always use that user's strictly earlier events."""
    from .lib import DAY, T0

    no, nc, ue, uc = pair_counts(d)
    h, c = d.hist_item[ev], d.cand[ev]
    eq = c[:, :, None] == h[:, None, :]
    rep = eq.sum(2)
    N, C = no[ev].astype(np.float32), nc[ev].astype(np.float32)
    reject = np.maximum(N - C, 0)
    a, b = 7.0, 4.0  # Beta prior on the event click rate: mean .63, strength set by how much users differ
    rate = (uc[ev] + a) / (ue[ev] + a + b)
    u_logit = np.log(rate / (1 - rate))
    n_hist = (h >= 0).sum(1)
    last = np.where(n_hist > 0, d.hist_ts[ev].max(1), d.ts[ev] - 30 * DAY)
    last_age = np.log1p(np.minimum((d.ts[ev] - last) / 86400.0, 30))
    day = np.minimum((d.ts[ev] - T0) // DAY, 7) if cut_day is None else np.full(len(ev), cut_day - 20)
    n, k = item_day_counts(d)
    cn, ck = np.cumsum(n, 0), np.cumsum(k, 0)
    prior = 0.0629
    ni = np.where(day[:, None] > 0, cn[np.maximum(day - 1, 0)[:, None], c], 0).astype(np.float32)
    ci = np.where(day[:, None] > 0, ck[np.maximum(day - 1, 0)[:, None], c], 0).astype(np.float32)
    p = (ci + 50 * prior) / (ni + 50)
    i_logit = np.log(p / (1 - p))
    rp = lambda v: np.repeat(np.asarray(v, np.float32)[:, None], 10, 1)
    rival_rep = (rep.sum(1, keepdims=True) - rep).astype(np.float32)
    x = np.stack(
        [
            np.log1p(N),
            np.log1p(C),
            np.log1p(reject),
            (reject > 0) & (rep > 0),
            rival_rep,
            rp(u_logit),
            rp(np.log1p(ue[ev])),
            rp(n_hist / 20.0),
            rp(n_hist == 0),
            rp(last_age),
            i_logit,
            np.log1p(ni),
        ],
        -1,
    )
    return x.astype(np.float32)


def prior_sum(keys, t, vals):
    """For rows grouped by `keys`, the sum of each column of float `vals` over strictly earlier timestamps in the group."""
    order = np.lexsort((t, keys))
    k, ts = keys[order], t[order]
    n = len(k)
    new_group = np.r_[True, k[1:] != k[:-1]]
    new_tie = new_group | np.r_[True, ts[1:] != ts[:-1]]
    gs = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    ti = np.maximum.accumulate(np.where(new_tie, np.arange(n), 0))
    cv = np.r_[np.zeros((1, vals.shape[1])), np.cumsum(vals[order], 0)]
    out = np.zeros_like(vals)
    out[order] = cv[ti] - cv[gs]
    return out


def intercept(d, pany, tau2=0.5):
    """User intercept from past events only: one Newton step of a random effect, (O - E) / (I + 1/tau2), on the logit of a click.

    pany (n_train,) is the model's chance of any click for each labelled event, with NaN where it has no estimate. Those events
    use the mean rate, so they still count as evidence. Returns (estimate, posterior sd) per event of the whole data, labelled
    then evaluation. Evaluation rows contribute no evidence.
    """
    n = d.n_train
    p = np.where(np.isnan(pany), 0.63, pany).astype(np.float64)
    c = (d.y[:n] == 1).any(1).astype(np.float64)
    vals = np.zeros((d.n, 3))
    vals[:n, 0], vals[:n, 1], vals[:n, 2] = c, p, p * (1 - p)
    s = prior_sum(d.user.astype(np.int64), d.ts.astype(np.int64), vals)
    prec = s[:, 2] + 1 / tau2
    return (s[:, 0] - s[:, 1]) / prec, prec**-0.5


def support(d, day, ev, D):
    """Known positive mass of the candidate and of the user at the day's cutoff, and the density rank in the slate."""
    pu, pi, _, w = states.positives(d, day_ts(day))
    ni, nu = np.bincount(pi, weights=w, minlength=NI), np.bincount(pu, weights=w, minlength=NU)
    rank = D[..., 0].argsort(1).argsort(1) / 9.0
    return np.stack([np.log1p(ni[d.cand[ev]]), np.repeat(np.log1p(nu[d.user[ev]])[:, None], 10, 1), rank], -1).astype(
        np.float32
    )


SCALAR_NAMES = [
    "log_offers",
    "log_clicks",
    "log_rejects",
    "rejected_repeat",
    "rival_repeat",
    "u_logit",
    "log_u_events",
    "hist_frac",
    "hist_empty",
    "last_age",
    "i_logit",
    "log_i_rows",
]
