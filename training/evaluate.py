"""Evaluate a trained LC_DDPM model.

    python training/evaluate.py grid       --model RUN --bank BANK --out grid.png
    python training/evaluate.py period     --model RUN --bank BANK [--n 200 --method ls|nn]
    python training/evaluate.py classifier --model RUN --shards SHARD_DIR --shard NAME --survey OGLE-I

grid:       generated profiles vs the real bank profiles nearest in condition space.
period:     render light curves with bank-like conditions, recover the period with a numpy
            generalised Lomb-Scargle (or nn_periodogram, if installed) and the amplitude from the
            folded curve; report recovery fractions per survey.
classifier: real light curves from an NN_Periodogram shard (``NAME.npz``, ``NAME_meta.json``,
            ``NAME_labels.json``) vs synthetic curves on the same times and errors; a small MLP on
            folded, binned, normalised profiles; report the held-out AUC (0.5 = indistinguishable).

Requires the package: run `pip install -e .` from the repo root first.
"""
import argparse
import json
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


def fold_binned(t, m, period, nbins):
    """Median of each phase bin (empty bins interpolated periodically) and the residual rms."""
    ph = np.mod(t / period, 1.0)
    k = np.minimum((ph * nbins).astype(int), nbins - 1)
    prof = np.full(nbins, np.nan)
    for b in np.unique(k):
        prof[b] = np.median(m[k == b])
    good = np.flatnonzero(np.isfinite(prof))
    x = (np.arange(nbins) + 0.5) / nbins
    prof = np.interp(x, x[good], prof[good], period=1.0)
    return prof, float(np.std(m - prof[k]))


def profile_features(t, m, period, nbins):
    """Folded binned profile, min-max normalised, faintest bin rolled to index 0, plus the
    residual rms in units of the amplitude."""
    prof, rms = fold_binned(t, m, period, nbins)
    amp = prof.max() - prof.min()
    prof = (prof - prof.min()) / max(amp, 1e-12)
    return np.append(np.roll(prof, -int(np.argmax(prof))), rms / max(amp, 1e-12))


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


def bank_conditions(arrays, rows, vocab):
    return [(vocab["surveys"][arrays["survey"][r]], vocab["classes"][arrays["cls"][r]], float(arrays["period"][r]),
             float(arrays["amplitude"][r]), float(arrays["mag"][r])) for r in rows]


# ---------------------------------------------------------------- (a) sample grids
def cmd_grid(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = Generator.from_pretrained(a.model, device=a.device)
    arrays, vocab = load_bank(a.bank)
    rng = np.random.default_rng(a.seed)
    rows = rng.choice(len(arrays["profile"]), a.n_cond, replace=False)
    num = np.stack([np.log10(arrays["period"]), np.log10(arrays["amplitude"]), arrays["mag"]], 1)
    num = (num - np.asarray(g.norm["mean"])) / np.asarray(g.norm["std"])
    cols = 4
    fig, axes = plt.subplots((a.n_cond + cols - 1) // cols, cols, figsize=(4 * cols, 2.8 * ((a.n_cond + cols - 1) // cols)),
                             squeeze=False)
    ph = (np.arange(arrays["profile"].shape[1]) + 0.5) / arrays["profile"].shape[1]
    for ax, r in zip(axes.flat, rows):
        same = np.flatnonzero((arrays["survey"] == arrays["survey"][r]) & (arrays["cls"] == arrays["cls"][r]))
        d = np.nansum((num[same] - num[r]) ** 2, 1)
        real = arrays["profile"][same[np.argsort(d)[:a.k_real]]]
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
    report = {}
    for s_idx, survey in enumerate(vocab["surveys"]):
        rows = np.flatnonzero(arrays["survey"] == s_idx)
        if len(rows) == 0:
            continue
        rows = rng.choice(rows, min(a.n, len(rows)), replace=False)
        exact = alias = 0
        amp_ratio = []
        for c_idx in np.unique(arrays["cls"][rows]):
            r = rows[arrays["cls"][rows] == c_idx]
            curves = g.generate(survey, arrays["period"][r], arrays["amplitude"][r], mag=arrays["mag"][r],
                                cls=vocab["classes"][c_idx], n=len(r), seed=int(rng.integers(2 ** 31)),
                                guidance=a.guidance, steps=a.steps)
            for c in curves:
                t, m, e = c["time"], c["mag"], c["err"]
                if len(t) > a.max_points:
                    keep = np.sort(rng.choice(len(t), a.max_points, replace=False))
                    t, m, e = t[keep], m[keep], e[keep]
                ratio = best_period(t, m, e, max(a.fmax, 1.5 / c["period"]), method=a.method) / c["period"]
                exact += abs(ratio - 1) < a.tol
                alias += min(abs(ratio - k) / k for k in (0.5, 1, 2)) < a.tol
                prof, _ = fold_binned(c["time"], c["mag"] - c["mean_mag"], c["period"], a.bins)
                amp_ratio.append((prof.max() - prof.min()) / c["amplitude"])
        n = len(rows)
        report[survey] = {"n": int(n), "period_exact": exact / n, "period_with_aliases": alias / n,
                          "amplitude_ratio_median": float(np.median(amp_ratio)),
                          "amplitude_ratio_iqr": [float(q) for q in np.percentile(amp_ratio, [25, 75])]}
        print(survey, json.dumps(report[survey]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------- (c) real vs synthetic classifier
def load_shard(shards, name, split, n, rng):
    d = np.load(Path(shards) / f"{name}.npz")
    meta = json.loads((Path(shards) / f"{name}_meta.json").read_text())
    labels = [r for r in json.loads((Path(shards) / f"{name}_labels.json").read_text())
              if meta[r["index"]]["split"] == split]
    labels = [labels[i] for i in rng.permutation(len(labels))[:n]]
    off = d["offsets"]
    return [(d["time"][off[r["index"]]:off[r["index"] + 1]].astype(float),
             d["mag"][off[r["index"]]:off[r["index"] + 1]].astype(float),
             d["err"][off[r["index"]]:off[r["index"] + 1]].astype(float), 1 / r["freq"]) for r in labels]


def cmd_classifier(a):
    g = Generator.from_pretrained(a.model, device=a.device)
    rng = np.random.default_rng(a.seed)
    stars = load_shard(a.shards, a.shard, a.split, a.n, rng)
    s_idx = g.surveys.index(a.survey)
    has_mag = g.templates.mags_known(s_idx)
    P = np.array([s[3] for s in stars])
    fits = [fold_binned(t, m, p, a.bins) for t, m, e, p in stars]
    amp = np.array([f[0].max() - f[0].min() for f in fits])
    med = np.array([np.median(s[1]) for s in stars])
    scat = np.array([f[1] / np.median(s[2]) for f, s in zip(fits, stars)])
    prof = g.sample_profiles(np.full(len(stars), s_idx), np.full(len(stars), -1), P, amp,
                             med if has_mag else np.full(len(stars), np.nan), a.guidance, a.steps, seed=a.seed)
    real, fake = [], []
    for (t, m, e, p), pr, A, M, s in zip(stars, prof, amp, med, scat):
        real.append(profile_features(t, m, p, a.bins))
        mm = M + A * (interp_profile(pr, (t - rng.uniform(0, p)) / p) - pr.mean()) + rng.normal(0, 1, len(t)) * e * s
        fake.append(profile_features(t, mm, p, a.bins))
    X = torch.tensor(np.concatenate([real, fake]), dtype=torch.float32)
    y = torch.tensor(np.r_[np.zeros(len(real)), np.ones(len(fake))], dtype=torch.float32)
    star = np.r_[np.arange(len(real)), np.arange(len(fake))]
    test = np.isin(star, rng.permutation(len(stars))[:len(stars) // 2])  # split by star: pairs stay together
    torch.manual_seed(a.seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = torch.as_tensor(~test)
    for _ in range(a.epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[tr])[:, 0], y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        score = net(X[torch.as_tensor(test)])[:, 0].numpy()
    res = {"shard": a.shard, "survey": a.survey, "n_stars": len(stars), "auc": auc(score, y.numpy()[test])}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="checkpoint or run directory")
    common.add_argument("--device")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--guidance", type=float, default=2.0)
    common.add_argument("--steps", type=int, default=50)
    common.add_argument("--bins", type=int, default=32)
    common.add_argument("--out")
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
    s.set_defaults(func=cmd_period)
    s = sub.add_parser("classifier", parents=[common])
    s.add_argument("--shards", required=True)
    s.add_argument("--shard", required=True)
    s.add_argument("--survey", required=True, help="survey token of this shard, e.g. OGLE-I")
    s.add_argument("--split", default="test", help="shard split to draw real stars from (never in the bank)")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--epochs", type=int, default=300)
    s.set_defaults(func=cmd_classifier)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
