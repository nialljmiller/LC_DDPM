"""Build the LC_DDPM v3 training file from NN_Periodogram shards and bank v2.2 (contract: docs/DESIGN_v3.md).

    python training/build_v3.py --shards SHARD_DIR --bank BANK_V22_DIR --out V3_DIR [--workers 16]

Stars: exactly the rows of the bank's profiles.npz (training-split stars with a label period; the bank
applies the split rule). Survey, class, period, amplitude, magnitude, tier, snr and repeats are taken from
the bank row. As a guard, a row is also dropped (and counted) if its shard entry is not "train" or its base
ID is in the bank's excluded.json. Light curves are read with build_bank.load_star (no Kepler/TESS
thinning), sorted by time, and cut with the window rule (lc_ddpm.data_v3.window, seeded per star).

Writes to --out:
  train_v3.npz   ragged points (time, obs_mag, obs_err, offsets) + per-example conditions
  cadences.npz   up to --cadence-cap examples per survey with their times (cadence="real" baseline)
  vocab.json     copy of the bank's vocab.json
  summary.json   counts per survey and rule, dropped stars by reason
"""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bank as bb  # noqa: E402
from lc_ddpm.data import load_bank  # noqa: E402
from lc_ddpm.data_v3 import MIN_POINTS, N_MAX, window  # noqa: E402

ROW_KEYS = ("survey", "cls", "period", "amplitude", "mag", "tier", "snr", "repeats", "shard", "index", "sourceid")


def work(job):
    """Windowed examples for a list of (bank row, index, period) of one shard."""
    shard_dir, shard, items, n_max = job
    for other in [k for k in bb._ARR if k != shard]:  # keep one shard's arrays per worker (npz shards are read whole)
        del bb._ARR[other]
    out = []
    for row, i, period in items:
        t, m, e = bb.load_star(shard_dir, shard, i, subsample=False)
        o = np.argsort(t, kind="stable")
        t, m, e = t[o], m[o], e[o]
        w = window(t, period, np.random.default_rng(bb.seed_of(shard, i, salt=3)), n_max)
        if w is None:
            out.append({"row": row, "fail": "short_span" if len(t) >= 2 else "too_few_points"})
            continue
        idx, rule = w
        if len(idx) < MIN_POINTS:
            out.append({"row": row, "fail": "too_few_points"})
            continue
        tw = t[idx]
        out.append({"row": row, "rule": rule, "t": tw - tw[0], "m": m[idx], "e": e[idx], "n_total": len(t)})
    return out


def build(a):
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bank, vocab = load_bank(a.bank)
    excluded = json.loads((Path(a.bank) / "excluded.json").read_text()) if (Path(a.bank) / "excluded.json").exists() else {}
    excluded = {fam: set(ids) for fam, ids in excluded.items()}
    meta, _ = bb.load_meta(a.shards)
    rows = np.arange(len(bank["period"]))
    if a.sample:
        rows = np.sort(np.random.default_rng(a.seed).choice(rows, min(a.sample, len(rows)), replace=False))
    dropped, by_shard = Counter(), {}
    for r in rows:
        s, i = str(bank["shard"][r]), int(bank["index"][r])
        if s not in meta or i >= len(meta[s]):
            dropped["not_in_shards"] += 1
            continue
        fam, b = bb.base_id(s, meta[s][i])
        if meta[s][i]["split"] != "train" or b in excluded.get(fam, ()):
            dropped["not_train"] += 1
            continue
        by_shard.setdefault(s, []).append((int(r), i, float(bank["period"][r])))
    jobs = [(a.shards, s, items[k:k + a.chunk], a.n_max) for s, items in sorted(by_shard.items())
            for k in range(0, len(items), a.chunk)]
    print(f"[{time.time()-t0:.0f}s] {sum(map(len, by_shard.values()))} bank rows in {len(by_shard)} shards, "
          f"{len(jobs)} jobs, {a.workers} workers; dropped {dict(dropped)}", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res += r
            if k % max(1, len(jobs) // 20) == 0:
                print(f"[{time.time()-t0:.0f}s] {k+1}/{len(jobs)} jobs", flush=True)
    res.sort(key=lambda r: r["row"])
    dropped.update(r["fail"] for r in res if "fail" in r)
    ok = [r for r in res if "fail" not in r]
    sel = np.array([r["row"] for r in ok], np.int64)

    ex = {k: bank[k][sel] for k in ROW_KEYS if k in bank}
    if "repeats" not in ex:
        ex["repeats"] = np.ones(len(sel), np.int8)
    ex.update(bank_row=sel.astype(np.int32), rule=np.array([r["rule"] for r in ok], np.int8),
              n_points=np.array([len(r["t"]) for r in ok], np.int32),
              n_total=np.array([r["n_total"] for r in ok], np.int32),
              span=np.array([r["t"][-1] for r in ok], np.float64))
    offsets = np.concatenate([[0], np.cumsum(ex["n_points"])]).astype(np.int64)
    cat = lambda k, dt: np.concatenate([r[k] for r in ok]).astype(dt) if ok else np.zeros(0, dt)
    np.savez(out / "train_v3.npz", time=cat("t", np.float64), obs_mag=cat("m", np.float32),
             obs_err=cat("e", np.float32), offsets=offsets, n_max=np.int32(a.n_max), **ex)

    # cadence baseline: random examples per survey with their times
    rng = np.random.default_rng(a.seed)
    pick = np.sort(np.concatenate([rng.choice(np.flatnonzero(ex["survey"] == s), min(a.cadence_cap, (ex["survey"] == s).sum()),
                                              replace=False) for s in np.unique(ex["survey"])])) if ok else np.zeros(0, int)
    np.savez(out / "cadences.npz", time=np.concatenate([ok[j]["t"] for j in pick]) if len(pick) else np.zeros(0),
             offsets=np.concatenate([[0], np.cumsum(ex["n_points"][pick])]).astype(np.int64),
             **{k: ex[k][pick] for k in ("survey", "period", "mag", "span", "n_points", "sourceid")})
    shutil.copy(Path(a.bank) / "vocab.json", out / "vocab.json")

    surveys = vocab["surveys"]
    summary = {"runtime_s": round(time.time() - t0, 1), "n_examples": len(ok), "n_points": int(offsets[-1]),
               "n_max": a.n_max, "dropped": dict(dropped), "per_survey": {}}
    for s in np.unique(ex["survey"]):
        k = ex["survey"] == s
        summary["per_survey"][surveys[s]] = {
            "n": int(k.sum()), "rules": {str(r): int((ex["rule"][k] == r).sum()) for r in (1, 2, 3)},
            "n_points_q": np.percentile(ex["n_points"][k], [5, 50, 95]).round(1).tolist(),
            "span_q": np.percentile(ex["span"][k], [5, 50, 95]).round(3).tolist(),
            "span_over_period_q": np.percentile(ex["span"][k] / ex["period"][k], [5, 50, 95]).round(2).tolist()}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"[{time.time()-t0:.0f}s] wrote {out / 'train_v3.npz'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="NN_Periodogram shard directory")
    ap.add_argument("--bank", required=True, help="bank v2.2 directory (profiles.npz, vocab.json, excluded.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200, help="stars per job")
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--cadence-cap", type=int, default=2000, help="examples per survey in cadences.npz")
    ap.add_argument("--sample", type=int, default=0, help="random bank rows only (test runs)")
    ap.add_argument("--seed", type=int, default=20261119)
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
