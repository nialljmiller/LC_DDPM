"""Evaluate an LC_DDPM v3 run against real stars of one NN_Periodogram shard (docs/DESIGN_v3.md, Evaluation).

    python training/evaluate_v3.py --model RUN_DIR --shards SHARD_DIR --shard kepler_eb [--split val --n 1000]
                                   [--v2-model V2_RUN_DIR] [--out result.json]

Real stars: up to --n labelled stars of --split (never in the bank), read and cut with the same window rule as
the v3 training file (build_v3.work), so real and synthetic curves have comparable lengths. Their conditions:
label period, amplitude and median magnitude from build_bank.extract on the star (the bank's own definition),
class family from build_bank.raw_class/family, survey token from build_bank.survey_token. Results per survey
token found in the shard.

Synthetic curves, same conditions per star:
  v3_full     generate(...) with each real star's own window n and span (times from the cadence model)
  v3_measure  generate(..., times=real window times): the measurement model alone
  v3_real_cadence  generate(..., cadence="real") (times of a training-split example; needs cadences.npz)
  v2_full     the v2 generator (--v2-model), its curves cut with the same window rule
Scores per mode (and per class where a class has >= --min-class stars):
  auc         real-vs-synthetic MLP AUC (0.5 = indistinguishable) for evaluate.py's feature sets profile,
              stats, both, cadence (n, span) and here also gaps (quantiles of log10 time gaps)
  period      Lomb-Scargle recovery of the label period (evaluate.period_recovery), next to the real stars'
  amplitude   folded peak-to-peak / requested amplitude (median, IQR); for real stars / extracted amplitude
  cadence     quantiles of n, span and pooled log10 gaps (days), KS distance of the pooled gaps to real

Requires the package: run `pip install -e .` from the repo root first.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bank as bb  # noqa: E402
from evaluate import curve_features, feature_set_aucs, fold_binned, mlp_auc, period_recovery  # noqa: E402
from lc_ddpm import Generator, GeneratorV3  # noqa: E402
from lc_ddpm.data_v3 import MIN_POINTS, window  # noqa: E402

Q = [5, 25, 50, 75, 95]


def load_real(shards, shard, split, n, n_max, seed):
    """Up to n windowed real stars of ``split``: dicts with t, m, e, period, amplitude, mag, survey, cls."""
    meta, labels = bb.load_meta(shards)
    meta, labels = meta[shard], labels[shard]
    idx = [i for i in labels if meta[i]["split"] == split]
    out = []
    for i in np.random.default_rng(seed).permutation(idx):
        i = int(i)
        tok = bb.survey_token(shard, meta[i])
        t, m, e = bb.load_star(shards, shard, i, subsample=False)
        o = np.argsort(t, kind="stable")
        t, m, e = t[o], m[o], e[o]
        w = window(t, 1 / labels[i], np.random.default_rng(bb.seed_of(shard, i, salt=3)), n_max)
        if w is None or len(w[0]) < MIN_POINTS:
            continue
        fit = bb.extract(*bb.load_star(shards, shard, i), labels[i], seed=bb.seed_of(shard, i))
        if "fail" in fit:
            continue
        tw = t[w[0]]
        out.append({"t": tw - tw[0], "m": m[w[0]], "e": e[w[0]], "period": 1 / labels[i], "amplitude": fit["amplitude"],
                    "mag": np.nan if tok in bb.NO_ABS_MAG else fit["mag"], "survey": tok,
                    "cls": bb.family(bb.raw_class(shard, meta[i]))})
        if len(out) == n:
            break
    return out


def synthetic(gen, stars, mode, guidance, steps, seed, n_max):
    """Synthetic (t, m, e, period) per star (same order), generated per class."""
    out = [None] * len(stars)
    for k, c in enumerate(sorted({s["cls"] for s in stars})):
        j = [i for i, s in enumerate(stars) if s["cls"] == c]
        P, A, M = (np.array([stars[i][x] for i in j]) for x in ("period", "amplitude", "mag"))
        kw = dict(cls=c if c in gen.classes else None, n=len(j), seed=seed + k, guidance=guidance, steps=steps,
                  mag=M if np.isfinite(M).all() else None)
        if mode == "v3_full":
            kw.update(n_points=[len(stars[i]["t"]) for i in j], span=[stars[i]["t"][-1] for i in j])
        elif mode == "v3_measure":
            kw.update(times=[stars[i]["t"] for i in j])
        elif mode == "v3_real_cadence":
            kw.update(cadence="real")
        curves = gen.generate(stars[j[0]]["survey"], P, A, **kw)
        rng = np.random.default_rng(seed + k)
        for i, cv in zip(j, curves):
            t, m, e = cv["time"], cv["mag"], cv["err"]
            if mode == "v2_full":  # cut like the real stars
                w = window(t, cv["period"], rng, n_max)
                if w is not None:
                    t, m, e = t[w[0]] - t[w[0][0]], m[w[0]], e[w[0]]
            out[i] = (t, m, e, cv["period"])
    return out


def gap_features(curves):
    """Quantiles of log10 time gaps (days) and the fraction of gaps above 0.5 d, per curve."""
    f = []
    for t, *_ in curves:
        g = np.log10(np.maximum(np.diff(t), 1e-6))
        f.append(np.r_[np.percentile(g, Q), np.mean(g > np.log10(0.5))])
    return np.array(f)


def ks(a, b):
    x = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), x, "right") / len(a) - np.searchsorted(np.sort(b), x, "right") / len(b))))


def summary(curves, amps, bins, period_n, seed, real_gaps=None):
    """Period recovery, amplitude ratio and cadence statistics of (t, m, e, period) curves."""
    ratio = np.array([np.ptp(fold_binned(t, m, p, bins)[0]) / a for (t, m, e, p), a in zip(curves, amps)])
    gaps = np.concatenate([np.log10(np.maximum(np.diff(c[0]), 1e-6)) for c in curves])
    r = {"n": len(curves), "period": period_recovery(curves[:period_n], 3000, 10.0, 0.01, seed),
         "amplitude_ratio_median": float(np.median(ratio)),
         "amplitude_ratio_iqr": np.percentile(ratio, [25, 75]).round(4).tolist(),
         "n_points_q": np.percentile([len(c[0]) for c in curves], Q).round(1).tolist(),
         "span_q": np.percentile([c[0][-1] - c[0][0] for c in curves], Q).round(4).tolist(),
         "log10_gap_q": np.percentile(gaps, Q).round(4).tolist()}
    if real_gaps is not None:
        r["gap_ks_vs_real"] = ks(gaps, real_gaps)
    return r


def compare(real, fake, amps, test, a):
    """AUCs (feature sets of evaluate.py + gaps) and summary of one synthetic set against the real one."""
    fr, ff = curve_features(real, a.bins), curve_features(fake, a.bins)
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    aucs = feature_set_aucs(fr, ff, test, a.epochs, a.seed)
    aucs["gaps"] = mlp_auc(np.concatenate([gap_features(real), gap_features(fake)]), y, np.r_[test, test], a.epochs, a.seed)
    real_gaps = np.concatenate([np.log10(np.maximum(np.diff(c[0]), 1e-6)) for c in real])
    return {"auc": aucs, **summary(fake, amps, a.bins, a.period_n, a.seed, real_gaps)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="v3 run directory (lc_ddpm_measure.pt, lc_ddpm_cadence.pt)")
    p.add_argument("--v2-model", help="v2 run directory for the comparison")
    p.add_argument("--shards", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--split", default="val", help="shard split of the real stars (never in the bank)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--modes", default="v3_full,v3_measure", help="also: v3_real_cadence (v2_full is added with --v2-model)")
    p.add_argument("--period-n", type=int, default=300, help="curves per mode for Lomb-Scargle recovery")
    p.add_argument("--min-class", type=int, default=40, help="per-class scores for classes with at least this many stars")
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--asassn-classes", help="JSON: VIRAC sourceid -> ASAS-SN class (vvv_asassn shard)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.asassn_classes:
        bb.ASASSN_CLASSES.update(json.loads(Path(a.asassn_classes).read_text()))

    g3 = Generator.from_pretrained(a.model, device=a.device)
    if not isinstance(g3, GeneratorV3):
        raise SystemExit(f"{a.model} is not a v3 run")
    gens = {m: g3 for m in a.modes.split(",")}
    if a.v2_model:
        gens["v2_full"] = Generator.from_pretrained(a.v2_model, device=a.device)
    stars = load_real(a.shards, a.shard, a.split, a.n, g3.n_max, a.seed)
    res = {"shard": a.shard, "split": a.split, "model": str(a.model), "v2_model": a.v2_model, "surveys": {}}
    for survey in sorted({s["survey"] for s in stars}):
        ss = [s for s in stars if s["survey"] == survey]
        real = [(s["t"], s["m"], s["e"], s["period"]) for s in ss]
        amps = np.array([s["amplitude"] for s in ss])
        cls = np.array([s["cls"] for s in ss])
        test = np.isin(np.arange(len(ss)), np.random.default_rng(a.seed).permutation(len(ss))[:len(ss) // 2])
        out = {"n_stars": len(ss), "classes": {c: int((cls == c).sum()) for c in np.unique(cls)},
               "real": summary(real, amps, a.bins, a.period_n, a.seed)}
        for mode, gen in gens.items():
            fake = synthetic(gen, ss, mode, a.guidance, a.steps, a.seed, g3.n_max)
            out[mode] = compare(real, fake, amps, test, a)
            out[mode]["per_class"] = {}
            for c in np.unique(cls):
                k = np.flatnonzero(cls == c)
                if len(k) >= a.min_class:
                    pick = lambda xs: [xs[i] for i in k]
                    out[mode]["per_class"][c] = {"real": summary(pick(real), amps[k], a.bins, a.period_n, a.seed),
                                                 **compare(pick(real), pick(fake), amps[k], test[k], a)}
            print(survey, mode, json.dumps(out[mode]["auc"]), json.dumps(out[mode]["period"]), flush=True)
        print(survey, "real", json.dumps(out["real"]["period"]), flush=True)
        res["surveys"][survey] = out
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
