"""One-pass CSV parse into compact event arrays under cache/. Raw data is never modified.

Events keep export order (file order), so `first event per user` ties follow export order.
Row order inside an event is preserved in `cand`/`y` columns 0..9.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
COLS = [
    "rowid",
    "event_id",
    "user_id",
    "item_id",
    "timestamp",
    "time_bucket",
    "day_type",
    "history_item_ids",
    "history_timestamps",
]


def read(path, label):
    cols = COLS + (["clicked"] if label else [])
    df = pl.read_csv(
        path,
        columns=cols,
        schema_overrides={"timestamp": pl.Int64, "time_bucket": pl.Int8, "day_type": pl.Int8},
        low_memory=False,
    )
    n = df.height
    assert n % 10 == 0
    ev = df["event_id"].to_numpy()[::10]
    # each event is ten contiguous rows with the same user, time and history
    for c in ["event_id", "user_id", "timestamp", "history_item_ids", "history_timestamps"]:
        a = df[c].to_numpy()
        assert (a.reshape(-1, 10) == a[::10, None]).all(), c
    assert len(set(ev)) == len(ev)
    return df


def main():
    t0 = time.time()
    tr, va = read(ROOT / "data/train.csv", True), read(ROOT / "data/evaluation.csv", False)
    print("read", tr.shape, va.shape, round(time.time() - t0), flush=True)
    n_tr, n_va = tr.height // 10, va.height // 10
    hist = pl.concat(
        [
            tr[::10].select(["history_item_ids", "history_timestamps"]),
            va[::10].select(["history_item_ids", "history_timestamps"]),
        ]
    )
    hi = hist["history_item_ids"].str.extract_all(r"itm_[0-9a-f]+")
    ht = hist["history_timestamps"].str.extract_all(r"\d+")
    hlen = hi.list.len().to_numpy().astype(np.int8)
    assert (hlen == ht.list.len().to_numpy()).all() and hlen.max() <= 20
    keep = hlen > 0
    hi_flat = hi.filter(keep).explode()
    ht_flat = ht.filter(keep).explode().cast(pl.Int64).to_numpy()
    assert len(hi_flat) == hlen.astype(np.int64).sum() == len(ht_flat)

    users = np.unique(np.concatenate([tr["user_id"].to_numpy()[::10], va["user_id"].to_numpy()[::10]]))
    items = np.unique(np.concatenate([tr["item_id"].to_numpy(), va["item_id"].to_numpy(), hi_flat.to_numpy()]))
    uid = pl.DataFrame({"k": users, "v": np.arange(len(users), dtype=np.int32)})
    iid = pl.DataFrame({"k": items, "v": np.arange(len(items), dtype=np.int32)})

    def code(s, tab):
        return pl.DataFrame({"k": s}).join(tab, on="k", how="left", maintain_order="left")["v"].to_numpy()

    hi_code = code(hi_flat, iid)
    n = n_tr + n_va
    hist_item = np.full((n, 20), -1, np.int32)
    hist_ts = np.zeros((n, 20), np.int32)
    ev_idx = np.repeat(np.arange(n), hlen)
    pos = np.arange(len(ev_idx)) - np.repeat(np.cumsum(np.r_[0, hlen[:-1].astype(np.int64)]), hlen)
    hist_item[ev_idx, pos] = hi_code
    hist_ts[ev_idx, pos] = ht_flat

    cand = np.concatenate([code(tr["item_id"], iid), code(va["item_id"], iid)]).reshape(n, 10)
    y = np.full((n, 10), -1, np.int8)
    y[:n_tr] = tr["clicked"].to_numpy().astype(np.int8).reshape(-1, 10)
    assert y[:n_tr].sum(1).max() == 1
    user = np.concatenate([code(tr["user_id"].to_numpy()[::10], uid), code(va["user_id"].to_numpy()[::10], uid)])
    ts = np.concatenate([tr["timestamp"].to_numpy()[::10], va["timestamp"].to_numpy()[::10]])
    bucket = np.concatenate([tr["time_bucket"].to_numpy()[::10], va["time_bucket"].to_numpy()[::10]])
    day_type = np.concatenate([tr["day_type"].to_numpy()[::10], va["day_type"].to_numpy()[::10]])
    assert (np.diff(hist_ts[:n_tr], axis=1)[hist_item[:n_tr, 1:] >= 0] >= 0).all()

    CACHE.mkdir(exist_ok=True)
    np.savez(
        CACHE / "events.npz",
        n_train=n_tr,
        user=user.astype(np.int32),
        ts=ts.astype(np.int32),
        bucket=bucket,
        day_type=day_type,
        cand=cand,
        y=y,
        hist_item=hist_item,
        hist_ts=hist_ts,
        hist_len=hlen,
    )
    np.savez(
        CACHE / "keys.npz",
        users=users.astype("U"),
        items=items.astype("U"),
        eval_rowid=va["rowid"].to_numpy().astype("U"),
    )
    print("done", n, len(users), len(items), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
