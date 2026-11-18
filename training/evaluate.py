"""Evaluate a trained LC_DDPM model.

    python training/evaluate.py grid       --model RUN --bank BANK --out grid.png
    python training/evaluate.py period     --model RUN --bank BANK [--n 200 --method ls|nn]
    python training/evaluate.py classifier --model RUN --shards SHARD_DIR --shard NAME --survey OGLE-I [--mode shape|full]
    python training/evaluate.py sweep      --model RUN --bank BANK --shards SHARD_DIR --shard NAME --survey OGLE-I

grid:       generated profiles vs the real bank profiles nearest in condition space.
period:     render light curves with bank-like conditions, recover the period with a numpy
            generalised Lomb-Scargle (or nn_periodogram, if installed) and the amplitude from the
            folded curve; report recovery fractions per survey, once with residual noise and once
            with Gaussian noise (same profiles, templates and phases; only the noise differs).
classifier: real light curves from an NN_Periodogram shard (``NAME.npyd/`` memory-mapped arrays or
            ``NAME.npz``, plus ``NAME_meta.json``, ``NAME_labels.json``) vs synthetic curves; a small
            MLP; held-out AUC (0.5 = indistinguishable), train/test split by star.
            mode shape: synthetic curves on the real star's own times and errors with white noise at
            that star's scatter (tests the shape model alone).
            mode full:  synthetic curves from Generator.generate for the same survey/period/amplitude/mag
            (template cadence and template noise from the bank, i.e. training-split stars): the
            realistic pipeline.
            Real stars above --cap-points points are subsampled as the bank builder does for templates.
            Feature sets, each reported: profile (folded binned normalised profile at the known period
            + residual rms / amplitude), stats (noise and light-curve statistics, STAT_NAMES[NOISE]),
            both (profile + stats), cadence (point count and time span only, STAT_NAMES[CADENCE]).
            With --period-n, Lomb-Scargle period recovery on real and synthetic curves of the same stars.
sweep:      for each guidance and DDIM step count: shape-mode classifier AUC on one shard and the
            diversity of samples for fixed conditions (mean pairwise rms distance among samples)
            vs that among the nearest real bank profiles; JSON table.

Requires the package: run `pip install -e .` from the repo root first.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from lc_ddpm import Generator
from lc_ddpm.data import load_bank
from lc_ddpm.render import interp_profile


# ---------------------------------------------------------------- helpers
def lomb_scargle(t, y, dy, freqs, chunk=2000):
    """Generalised (floating-mean, weighted) Lomb-Scargle power, Zechmeister & Kuerster 2009."""
    w = 1 / dy ** 2
    w = w / w.sum()
    y = y - (w * y).sum()
    YY = (w * y * y).sum()
    out = np.empty(len(freqs))
    for s in range(0, len(freqs), chunk):
        arg = 2 * np.pi * freqs[s:s + chunk, None] * t[None]
        c, sn = np.cos(arg), np.sin(arg)
        C, S = c @ w, sn @ w
        YC, YS = (c * y) @ w, (sn * y) @ w
        CC = (c * c) @ w - C * C
        SS = (sn * sn) @ w - S * S
        CS = (c * sn) @ w - C * S
        D = CC * SS - CS ** 2
        out[s:s + chunk] = (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (YY * D)
    return out


def best_period(t, m, e, fmax, oversample=5, method="ls"):
    if method == "nn":
        from nn_periodogram import periodogram
        return float(periodogram(t, m, e).best_period)
    T = t.max() - t.min()
    df = 1 / (oversample * T)
    if method == "ls":
        try:
            import nifty_ls  # fast (NUFFT) Lomb-Scargle, same answer as the numpy fallback below
            r = nifty_ls.lombscargle(t, m, e, fmin=1 / T, fmax=fmax, Nf=int((fmax - 1 / T) / df) + 1)
            return 1 / r.freq()[np.argmax(r.power)]
        except ImportError:
            pass
    freqs = np.arange(1 / T, fmax, df)
    return 1 / freqs[np.argmax(lomb_scargle(t, m, e, freqs))]


def fold(t, m, period, nbins):
    """Median of each phase bin (empty bins interpolated periodically) and the residuals around it."""
    ph = np.mod(t / period, 1.0)
    k = np.minimum((ph * nbins).astype(int), nbins - 1)
    prof = np.full(nbins, np.nan)
    for b in np.unique(k):
        prof[b] = np.median(m[k == b])
    good = np.flatnonzero(np.isfinite(prof))
    x = (np.arange(nbins) + 0.5) / nbins
    prof = np.interp(x, x[good], prof[good], period=1.0)
    return prof, m - prof[k]


def fold_binned(t, m, period, nbins):
    """Folded binned profile and the residual rms."""
    prof, r = fold(t, m, period, nbins)
    return prof, float(np.std(r))


def profile_features(t, m, period, nbins):
    """Folded binned profile, min-max normalised, faintest bin rolled to index 0, plus the
    residual rms in units of the amplitude."""
    prof, rms = fold_binned(t, m, period, nbins)
    amp = prof.max() - prof.min()
    prof = (prof - prof.min()) / max(amp, 1e-12)
    return np.append(np.roll(prof, -int(np.argmax(prof))), rms / max(amp, 1e-12))


STAT_NAMES = ("log10_n_points", "log10_span", "skewness", "excess_kurtosis", "eta_resid", "outlier_frac_3sigma",
              "resid_rms_over_median_err")
CADENCE = [0, 1]          # columns of STAT_NAMES that describe the cadence only ("cadence" feature set)
NOISE = [2, 3, 4, 5, 6]   # columns that describe the light curve and its noise ("stats" feature set)


def stat_features(t, m, e, period, nbins):
    """Statistics a skeptic would check, none of which uses absolute time or mean magnitude: point count,
    time span, skewness and excess kurtosis of the magnitudes, von Neumann ratio of the time-ordered
    residuals around the folded profile, fraction of residuals beyond 3 robust sigma (1.4826 MAD),
    and residual rms / median error."""
    o = np.argsort(t)
    t, m, e = t[o], m[o], e[o]
    _, r = fold(t, m, period, nbins)
    z = (m - m.mean()) / max(m.std(), 1e-12)
    sig = 1.4826 * np.median(np.abs(r - np.median(r)))
    return np.array([np.log10(len(t)), np.log10(max(t[-1] - t[0], 1e-6)), np.mean(z ** 3), np.mean(z ** 4) - 3,
                     np.mean(np.diff(r) ** 2) / max(np.var(r), 1e-24),
                     np.mean(np.abs(r - np.median(r)) > 3 * max(sig, 1e-12)), np.std(r) / max(np.median(e), 1e-12)])


def curve_features(curves, nbins):
    """(profile features, stat features) for a list of (t, m, e, period)."""
    prof = np.array([profile_features(t, m, p, nbins) for t, m, e, p in curves])
    stats = np.array([stat_features(t, m, e, p, nbins) for t, m, e, p in curves])
    return prof, stats


def auc(score, label):
    """Rank-based ROC AUC (ties averaged)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mlp_auc(X, y, test, epochs, seed):
    """Train a small MLP on the non-test rows (features standardised on them) and return the test AUC."""
    X = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[~test].mean(0), X[~test].std(0)
    X = torch.tensor((X - mu) / np.where(sd > 0, sd, 1.0), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = torch.as_tensor(~test)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[tr])[:, 0], yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        score = net(X[torch.as_tensor(test)])[:, 0].numpy()
    return auc(score, y[test])


def feature_set_aucs(real, fake, test_stars, epochs, seed, sets=("profile", "stats", "both", "cadence")):
    """AUC per feature set. real/fake: (profile, stats) feature arrays, row i of both belongs to star i;
    the same star split (``test_stars``, boolean per star) is used for every set. "stats" uses the
    noise statistics only and "both" is profile + noise statistics; the cadence columns (point count,
    time span) are scored separately as "cadence" so a cadence difference cannot pass for a noise one."""
    y = np.r_[np.zeros(len(real[0])), np.ones(len(fake[0]))]
    test = np.r_[test_stars, test_stars]
    cols = {"profile": lambda f: f[0], "stats": lambda f: f[1][:, NOISE], "cadence": lambda f: f[1][:, CADENCE],
            "both": lambda f: np.concatenate([f[0], f[1][:, NOISE]], 1)}
    return {k: mlp_auc(np.concatenate([cols[k](real), cols[k](fake)]), y, test, epochs, seed) for k in sets}


def period_recovery(curves, max_points, fmax, tol, seed, method="ls"):
    """Fractions of curves (t, m, e, period) whose best periodogram period is within ``tol`` (relative) of
    the label, and within ``tol`` of 0.5, 1 or 2 times it. Curves are subsampled to ``max_points``."""
    rng = np.random.default_rng(seed)
    exact = alias = 0
    for t, m, e, p in curves:
        if len(t) > max_points:
            keep = np.sort(rng.choice(len(t), max_points, replace=False))
            t, m, e = t[keep], m[keep], e[keep]
        ratio = best_period(t, m, e, max(fmax, 1.5 / p), method=method) / p
        exact += abs(ratio - 1) < tol
        alias += min(abs(ratio - k) / k for k in (0.5, 1, 2)) < tol
    return {"period_exact": exact / len(curves), "period_with_aliases": alias / len(curves)}


def bank_conditions(arrays, rows, vocab):
    return [(vocab["surveys"][arrays["survey"][r]], vocab["classes"][arrays["cls"][r]], float(arrays["period"][r]),
             float(arrays["amplitude"][r]), float(arrays["mag"][r])) for r in rows]


def normalised_conditions(arrays, norm):
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.stack([np.log10(arrays["period"]), np.log10(arrays["amplitude"]), arrays["mag"]], 1)
    return (num - np.asarray(norm["mean"])) / np.asarray(norm["std"])


def nearest_real(arrays, num, r, k):
    """Indices of the ``k`` bank rows of the same survey and class nearest to row ``r`` in condition space."""
    same = np.flatnonzero((arrays["survey"] == arrays["survey"][r]) & (arrays["cls"] == arrays["cls"][r]))
    return same[np.argsort(np.nansum((num[same] - num[r]) ** 2, 1))[:k]]


def mean_pairwise_rms(p):
    """Mean rms distance over all pairs of rows of ``p``."""
    d = np.sqrt(((p[:, None] - p[None]) ** 2).mean(-1))
    iu = np.triu_indices(len(p), 1)
    return float(d[iu].mean()) if len(iu[0]) else float("nan")


# ---------------------------------------------------------------- (a) sample grids
def cmd_grid(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = Generator.from_pretrained(a.model, device=a.device)
    arrays, vocab = load_bank(a.bank)
    rng = np.random.default_rng(a.seed)
    rows = rng.choice(len(arrays["profile"]), a.n_cond, replace=False)
    num = normalised_conditions(arrays, g.norm)
    cols = 4
    fig, axes = plt.subplots((a.n_cond + cols - 1) // cols, cols, figsize=(4 * cols, 2.8 * ((a.n_cond + cols - 1) // cols)),
                             squeeze=False)
    ph = (np.arange(arrays["profile"].shape[1]) + 0.5) / arrays["profile"].shape[1]
    for ax, r in zip(axes.flat, rows):
        real = arrays["profile"][nearest_real(arrays, num, r, a.k_real)]
        n = a.n_samples
        fake = g.sample_profiles(np.full(n, arrays["survey"][r]), np.full(n, arrays["cls"][r]),
                                 np.full(n, arrays["period"][r]), np.full(n, arrays["amplitude"][r]),
                                 np.full(n, arrays["mag"][r]), a.guidance, a.steps, seed=a.seed + int(r))
        ax.plot(ph, real.T, color="0.6", lw=0.8)
        ax.plot(ph, fake.T, color="C0", lw=0.8, alpha=0.8)
        ax.invert_yaxis()
        s, c, P, A, M = bank_conditions(arrays, [r], vocab)[0]
        ax.set_title(f"{s} {c} P={P:.3g} A={A:.2g} m={M:.3g}", fontsize=8)
    for ax in list(axes.flat)[a.n_cond:]:
        ax.axis("off")
    fig.suptitle("grey: nearest real profiles, blue: generated")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


# ---------------------------------------------------------------- (b) period / amplitude consistency
def cmd_period(a):
    g = Generator.from_pretrained(a.model, device=a.device)
    arrays, vocab = load_bank(a.bank)
    rng = np.random.default_rng(a.seed)
    modes = [m.strip() for m in a.noise.split(",")]
    report = {}
    for s_idx, survey in enumerate(vocab["surveys"]):
        rows = np.flatnonzero(arrays["survey"] == s_idx)
        if len(rows) == 0:
            continue
        rows = rng.choice(rows, min(a.n, len(rows)), replace=False)
        groups = [(c_idx, rows[arrays["cls"][rows] == c_idx], int(rng.integers(2 ** 31)))
                  for c_idx in np.unique(arrays["cls"][rows])]
        report[survey] = {}
        for mode in modes:  # same seed per class group: same profiles, templates and t0; only the noise differs
            exact = alias = 0
            amp_ratio, applied = [], Counter()
            for c_idx, r, seed in groups:
                curves = g.generate(survey, arrays["period"][r], arrays["amplitude"][r], mag=arrays["mag"][r],
                                    cls=vocab["classes"][c_idx], n=len(r), seed=seed, guidance=a.guidance,
                                    steps=a.steps, noise=mode)
                sub = np.random.default_rng(seed)
                for c in curves:
                    applied[c["noise"]] += 1
                    t, m, e = c["time"], c["mag"], c["err"]
                    if len(t) > a.max_points:
                        keep = np.sort(sub.choice(len(t), a.max_points, replace=False))
                        t, m, e = t[keep], m[keep], e[keep]
                    ratio = best_period(t, m, e, max(a.fmax, 1.5 / c["period"]), method=a.method) / c["period"]
                    exact += abs(ratio - 1) < a.tol
                    alias += min(abs(ratio - k) / k for k in (0.5, 1, 2)) < a.tol
                    prof, _ = fold_binned(c["time"], c["mag"] - c["mean_mag"], c["period"], a.bins)
                    amp_ratio.append((prof.max() - prof.min()) / c["amplitude"])
            n = len(rows)
            report[survey][mode] = {"n": int(n), "noise_applied": dict(applied), "period_exact": exact / n,
                                    "period_with_aliases": alias / n,
                                    "amplitude_ratio_median": float(np.median(amp_ratio)),
                                    "amplitude_ratio_iqr": [float(q) for q in np.percentile(amp_ratio, [25, 75])]}
            print(survey, mode, json.dumps(report[survey][mode]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------- (c) real vs synthetic classifier
def load_shard(shards, name, split, n, rng, min_points=10, max_points=None):
    """Up to ``n`` random stars of ``split`` as (t, m, e, period). Arrays come from the memory-mapped
    ``NAME.npyd/`` directory when present; otherwise each ``NAME.npz`` array is read once. Stars with
    more than ``max_points`` points are randomly subsampled to that many (the bank builder does the same
    to Kepler and TESS templates, so real and synthetic curves then have comparable cadences)."""
    npyd = Path(shards) / f"{name}.npyd"
    if npyd.is_dir():
        d = {k: np.load(npyd / f"{k}.npy", mmap_mode="r") for k in ("time", "mag", "err", "offsets")}
    else:  # NpzFile re-reads an array on every access, so load each one once
        with np.load(Path(shards) / f"{name}.npz") as z:
            d = {k: z[k] for k in ("time", "mag", "err", "offsets")}
    meta = json.loads((Path(shards) / f"{name}_meta.json").read_text())
    labels = [r for r in json.loads((Path(shards) / f"{name}_labels.json").read_text())
              if meta[r["index"]]["split"] == split]
    off, out = d["offsets"], []
    for i in rng.permutation(len(labels)):
        a, b = int(off[labels[i]["index"]]), int(off[labels[i]["index"] + 1])
        t, m, e = (np.asarray(d[k][a:b], np.float64) for k in ("time", "mag", "err"))
        ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
        if max_points and ok.sum() > max_points:
            ok[np.flatnonzero(ok)[np.sort(rng.permutation(ok.sum())[max_points:])]] = False
        if ok.sum() >= min_points:
            out.append((t[ok], m[ok], e[ok], 1 / labels[i]["freq"]))
        if len(out) == n:
            break
    return out


def star_conditions(stars, nbins):
    """Per-star period, folded amplitude, median magnitude and scatter ratio (rms residual / median err)."""
    P = np.array([s[3] for s in stars])
    fits = [fold_binned(t, m, p, nbins) for t, m, e, p in stars]
    amp = np.array([max(f[0].max() - f[0].min(), 1e-4) for f in fits])
    med = np.array([np.median(s[1]) for s in stars])
    scat = np.array([f[1] / np.median(s[2]) for f, s in zip(fits, stars)])
    return P, amp, med, scat


def shape_fakes(g, stars, s_idx, cond, guidance, steps, seed):
    """Synthetic curves on each real star's own times and errors, white noise at that star's scatter."""
    P, amp, med, scat = cond
    has_mag = g.templates.mags_known(s_idx)
    prof = g.sample_profiles(np.full(len(stars), s_idx), np.full(len(stars), -1), P, amp,
                             med if has_mag else np.full(len(stars), np.nan), guidance, steps, seed=seed)
    rng = np.random.default_rng(seed)
    return [(t, M + A * (interp_profile(pr, (t - rng.uniform(0, p)) / p) - pr.mean()) + rng.normal(0, 1, len(t)) * e * s,
             e, p) for (t, m, e, p), pr, A, M, s in zip(stars, prof, amp, med, scat)]


def full_fakes(g, survey, stars, s_idx, cond, guidance, steps, seed, noise):
    """Synthetic curves from Generator.generate: template cadence and template noise."""
    P, amp, med, _ = cond
    curves = g.generate(survey, P, amp, mag=med if g.templates.mags_known(s_idx) else None, n=len(stars), seed=seed,
                        guidance=guidance, steps=steps, noise=noise)
    return [(c["time"], c["mag"], c["err"], c["period"]) for c in curves], Counter(c["noise"] for c in curves)


def cmd_classifier(a):
    g = Generator.from_pretrained(a.model, device=a.device)
    rng = np.random.default_rng(a.seed)
    stars = load_shard(a.shards, a.shard, a.split, a.n, rng, max_points=a.cap_points)
    s_idx = g.surveys.index(a.survey)
    cond = star_conditions(stars, a.bins)
    real = curve_features(stars, a.bins)
    test_stars = np.isin(np.arange(len(stars)), rng.permutation(len(stars))[:len(stars) // 2])
    res = {"shard": a.shard, "survey": a.survey, "n_stars": len(stars), "stat_names": list(STAT_NAMES),
           "cap_points": a.cap_points}
    if a.period_n:
        res["period_real"] = period_recovery(stars[:a.period_n], 3000, 10.0, 0.01, a.seed)
        print("period real", json.dumps(res["period_real"]), flush=True)
    for mode in (("shape", "full") if a.mode == "both" else (a.mode,)):
        if mode == "shape":
            fake_curves, info = shape_fakes(g, stars, s_idx, cond, a.guidance, a.steps, a.seed), {}
        else:
            fake_curves, applied = full_fakes(g, a.survey, stars, s_idx, cond, a.guidance, a.steps, a.seed, a.noise)
            info = {"noise_applied": dict(applied)}
        fake = curve_features(fake_curves, a.bins)
        res[mode] = {"auc": feature_set_aucs(real, fake, test_stars, a.epochs, a.seed), **info,
                     "stat_median_real": dict(zip(STAT_NAMES, np.median(real[1], 0).round(4).tolist())),
                     "stat_median_fake": dict(zip(STAT_NAMES, np.median(fake[1], 0).round(4).tolist()))}
        if a.period_n:
            res[mode]["period"] = period_recovery(fake_curves[:a.period_n], 3000, 10.0, 0.01, a.seed)
        print(mode, json.dumps(res[mode]["auc"]), json.dumps(res[mode].get("period", {})), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- (d) guidance / steps sweep
def cmd_sweep(a):
    g = Generator.from_pretrained(a.model, device=a.device)
    arrays, _ = load_bank(a.bank)
    rng = np.random.default_rng(a.seed)
    stars = load_shard(a.shards, a.shard, a.split, a.n, rng, max_points=a.cap_points)
    s_idx = g.surveys.index(a.survey)
    cond = star_conditions(stars, a.bins)
    real = curve_features(stars, a.bins)
    test_stars = np.isin(np.arange(len(stars)), rng.permutation(len(stars))[:len(stars) // 2])
    num = normalised_conditions(arrays, g.norm)
    rows = np.flatnonzero(arrays["survey"] == s_idx)
    rows = rng.choice(rows, min(a.n_cond, len(rows)), replace=False)
    div_real = float(np.mean([mean_pairwise_rms(arrays["profile"][nearest_real(arrays, num, r, a.k_real)])
                              for r in rows]))
    table = {"shard": a.shard, "survey": a.survey, "n_stars": len(stars), "n_cond": len(rows),
             "n_samples": a.n_samples, "k_real": a.k_real, "diversity_real": div_real, "rows": []}
    for steps in [int(v) for v in a.step_list.split(",")]:
        for guidance in [float(v) for v in a.guidances.split(",")]:
            fake = curve_features(shape_fakes(g, stars, s_idx, cond, guidance, steps, a.seed), a.bins)
            aucs = feature_set_aucs(real, fake, test_stars, a.epochs, a.seed, sets=("profile", "both"))
            div = []
            for r in rows:
                k = a.n_samples
                prof = g.sample_profiles(np.full(k, arrays["survey"][r]), np.full(k, arrays["cls"][r]),
                                         np.full(k, arrays["period"][r]), np.full(k, arrays["amplitude"][r]),
                                         np.full(k, arrays["mag"][r]), guidance, steps, seed=a.seed + int(r))
                div.append(mean_pairwise_rms(prof))
            row = {"guidance": guidance, "steps": steps, "shape_auc_profile": aucs["profile"],
                   "shape_auc_both": aucs["both"], "diversity_fake": float(np.mean(div)),
                   "diversity_ratio": float(np.mean(div)) / div_real}
            table["rows"].append(row)
            print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(table, indent=1))
    print(f"wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="checkpoint or run directory")
    common.add_argument("--device")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--guidance", type=float, default=1.0)
    common.add_argument("--steps", type=int, default=50)
    common.add_argument("--bins", type=int, default=32)
    common.add_argument("--out")
    shard = argparse.ArgumentParser(add_help=False)
    shard.add_argument("--shards", required=True)
    shard.add_argument("--shard", required=True)
    shard.add_argument("--survey", required=True, help="survey token of this shard, e.g. OGLE-I")
    shard.add_argument("--split", default="test", help="shard split to draw real stars from (never in the bank)")
    shard.add_argument("--epochs", type=int, default=300)
    shard.add_argument("--cap-points", type=int, default=20000,
                       help="subsample real stars to at most this many points (the bank builder's cap for "
                            "Kepler and TESS templates); 0 = no cap")
    s = sub.add_parser("grid", parents=[common])
    s.add_argument("--bank", required=True)
    s.add_argument("--n-cond", type=int, default=16)
    s.add_argument("--n-samples", type=int, default=6)
    s.add_argument("--k-real", type=int, default=6)
    s.set_defaults(func=cmd_grid, out="grid.png")
    s = sub.add_parser("period", parents=[common])
    s.add_argument("--bank", required=True)
    s.add_argument("--n", type=int, default=200, help="curves per survey")
    s.add_argument("--method", choices=["ls", "nn"], default="ls")
    s.add_argument("--tol", type=float, default=0.01, help="relative period tolerance")
    s.add_argument("--fmax", type=float, default=10.0, help="max search frequency (1/day); raised to 1.5/P if needed")
    s.add_argument("--max-points", type=int, default=3000, help="random subsample for the periodogram")
    s.add_argument("--noise", default="residual,gaussian", help="comma-separated noise modes to compare")
    s.set_defaults(func=cmd_period)
    s = sub.add_parser("classifier", parents=[common, shard])
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--mode", choices=["shape", "full", "both"], default="both")
    s.add_argument("--noise", default="residual", help="noise mode of the full-mode synthetic curves")
    s.add_argument("--period-n", type=int, default=300,
                   help="also run Lomb-Scargle period recovery on this many real and synthetic curves (same "
                        "conditions), 0 = skip")
    s.set_defaults(func=cmd_classifier)
    s = sub.add_parser("sweep", parents=[common, shard])
    s.add_argument("--bank", required=True, help="bank for the real nearest-neighbour profiles")
    s.add_argument("--n", type=int, default=500, help="real stars for the classifier")
    s.add_argument("--guidances", default="1,1.5,2,3,4")
    s.add_argument("--step-list", default="25,50,100", help="DDIM step counts")
    s.add_argument("--n-cond", type=int, default=16, help="conditions (bank rows of the survey) for diversity")
    s.add_argument("--n-samples", type=int, default=8, help="samples per condition")
    s.add_argument("--k-real", type=int, default=8, help="nearest real profiles per condition")
    s.set_defaults(func=cmd_sweep, out="sweep.json")
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
