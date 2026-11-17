"""Build the LC_DDPM training bank from NN_Periodogram shards (contract: docs/DESIGN.md).

Writes to --out:
  profiles.npz   one 128-point phase profile per labelled train star
  templates.npz  real cadence + error + residual templates (ragged) with scatter ratios
  vocab.json     survey tokens, class families and the raw -> family map
  excluded.json  base star IDs left out because some entry is in val or test
  summary.json   counts and distributions

Split rule: entries are grouped by base star ID within an ID family (ASAS-SN band
suffix removed; all OGLE shards share one family; vvv_real, vvv_asassn and
vvv_quiet share the VIRAC family). A star is used only if every entry is "train".

Profiles: fold at the label period (1/freq from {shard}_labels.json). With at least
32 points per bin on average the profile is a binned median (see binned_profile). With
fewer than 8 it is a weighted truncated Fourier series with order chosen by BIC (error
scale fitted: N ln(chi2/N) + k ln N, because quoted errors are often underestimated) up
to min(15, (N-1)//4); orders whose model peak-to-peak exceeds the data range (ringing in
phase gaps) are not eligible. In the thin binned regime (8-32 points per bin) both are
computed and the one with the lower 5-fold cross-validated error-weighted residual is kept
(see choose_thin). Profile convention: magnitudes scaled to [0, 1], faintest point at index 0.

Templates carry the residual series and a robust scatter ratio 1.4826 MAD(resid) / median(err)
(the rms-based ratio is kept as scatter_rms), a quality flag (template_quality) and the
periodic-leak powers leak_power / leak_control (leak_powers). Residuals of labelled stars are
cross-fitted (cross_resid): the points are split at random into two halves and each half's
residual comes from a model of the star at its label period built only on the other half, so
the residual keeps neither the star's periodic signal nor the fitted model's own noise. vvv_quiet
stars use mag minus median. vvv_asassn classes come from --asassn-classes.

Usage (Aurora):
  python build_bank.py --shards /net/scratch/nillmill/nnp/shards --out BANK --workers 48
  python build_bank.py ... --sample 50 --only ogle_lmc_rrlyr,vvv_real --check CHECK.npz
  python build_bank.py --plot CHECK.npz --plot-dir DIR      # needs matplotlib
"""
import argparse
import json
import os
import re
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NBINS = 128
PHASES = (np.arange(NBINS) + 0.5) / NBINS
MAX_ORDER = 15
DENSE_PER_BIN = 8
SMOOTH_BELOW = 32
FINE_PER_BIN = 8                       # template residual model: points per bin on average
FINE_MAX_BINS = 4096
FINE_MIN_POINTS = 1024                 # fine-fold residual model from this many points
LEAK_HARMONICS = 5
LEAK_POWER, LEAK_RATIO = 0.1, 4.0      # periodic-leak rule
CV_FOLDS = 5
SUBSAMPLE = 20000                      # max points folded / stored for Kepler and TESS
TEMPLATE_CAP = {"Kepler-Kp": 300, "TESS-T": 300}
TEMPLATE_CAP_DEFAULT = 30000
SURVEYS = ["OGLE-I", "VVV-Ks", "ASAS-SN-V", "ASAS-SN-g", "Kepler-Kp", "TESS-T"]
NO_ABS_MAG = {"ASAS-SN-V", "ASAS-SN-g", "Kepler-Kp", "TESS-T"}
CLASSES = ["RR", "CEP", "T2CEP", "ACEP", "DSCT", "ECL", "ELL", "ROT", "LPV", "YSO",
           "DPV", "SHORT", "MISC", "UNKNOWN"]
OGLE_CLASS = {"rrlyr": "RR", "cep": "CEP", "t2cep": "T2CEP", "acep": "ACEP",
              "dsct": "DSCT", "ecl": "ECL", "dpv": "DPV"}
# first token of a VSX / PRIMVS (Gaia DR3 style) class -> family
ASASSN_CLASSES = {}                    # vvv_asassn: VIRAC sourceid -> ASAS-SN class (set in build)
TOKEN_FAMILY = {
    **dict.fromkeys(["RR", "RRAB", "RRC", "RRD"], "RR"),
    **dict.fromkeys(["CEP", "DCEP", "DCEPS", "DCEP-FU", "DCEP-FO"], "CEP"),
    **dict.fromkeys(["CW", "CWA", "CWB", "CW-FU", "CW-FO", "RV", "RVA", "RVB", "AHB1",
                     "BLHER", "T2CEP"], "T2CEP"),
    "ACEP": "ACEP",
    **dict.fromkeys(["DSCT", "DSCTC", "DSCTR", "HADS", "SXPHE", "GDOR",
                     "DSCT|GDOR|SXPHE"], "DSCT"),
    **dict.fromkeys(["E", "EA", "EB", "EW", "EC", "ED", "ESD", "EP", "ECL", "KEPLER_EB",
                     "TESS_EB"], "ECL"),
    "ELL": "ELL",
    **dict.fromkeys(["ROT", "RS", "BY", "SOLAR_LIKE", "ACV", "SXARI", "FKCOM"], "ROT"),
    **dict.fromkeys(["M", "SR", "SRA", "SRB", "SRC", "SRD", "SRS", "L", "LB", "LC",
                     "LPV", "MIRA"], "LPV"),
    **dict.fromkeys(["YSO", "TTS", "CTTS", "WTTS", "IN", "INA", "INB", "INT", "INS",
                     "INSA", "INSB", "INST", "UXOR", "EXOR"], "YSO"),
    "DPV": "DPV",
    "S": "SHORT",                      # PRIMVS/Gaia DR3 "S" = short timescale (unverified)
    "": "UNKNOWN",
}


# ----------------------------------------------------------------------------- metadata

def survey_token(shard, entry):
    if shard.startswith("ogle_"):
        return "OGLE-I"
    if shard.startswith("vvv_"):
        return "VVV-Ks"
    if shard == "kepler_eb":
        return "Kepler-Kp"
    if shard == "tess_eb":
        return "TESS-T"
    if shard == "asassn_vsx":
        return "ASAS-SN-" + entry["band"]
    raise ValueError(f"unknown shard {shard}")


def base_id(shard, entry):
    """(ID family, base star ID) for the split rule."""
    sid = str(entry["sourceid"])
    if shard == "asassn_vsx":
        return "asassn", sid.split(":")[0]
    if shard.startswith("ogle_"):
        return "ogle", sid
    if shard.startswith("vvv_"):
        return "virac", sid
    return shard.split("_")[0], sid      # kepler, tess


def raw_class(shard, entry):
    if shard.startswith("ogle_"):
        return "ogle/" + shard.split("_")[-1]
    if shard == "vvv_real":
        return "primvs/" + (entry.get("type") or "")
    if shard == "vvv_asassn":           # class from PRIMVS_P_ASASSN (--asassn-classes); "" if absent
        return "vvv_asassn/" + ASASSN_CLASSES.get(str(entry["sourceid"]), "")
    if shard == "vvv_quiet":
        return "vvv_quiet/"
    return entry.get("cls") or ""      # asassn VSX class, kepler_eb, tess_eb


def family(raw):
    if raw.startswith("ogle/"):
        return OGLE_CLASS[raw[5:]]
    if raw.startswith("primvs/"):
        t = raw[7:]
        return TOKEN_FAMILY.get(t.upper(), "MISC") if t else "UNKNOWN"
    if raw in ("vvv_asassn/", "vvv_quiet/"):
        return "UNKNOWN"
    if raw.startswith("vvv_asassn/"):  # ASAS-SN class string: same path as VSX classes
        raw = raw[len("vvv_asassn/"):]
    if raw.upper() in TOKEN_FAMILY:
        return TOKEN_FAMILY[raw.upper()]
    tok = re.split(r"[!+/ ]", raw)[0]
    tok = re.sub(r"\(.*?\)|:", "", tok).upper()
    return TOKEN_FAMILY.get(tok, "MISC")


def tier(shard):
    return 1 if shard == "vvv_real" else 0


def seed_of(shard, index, salt=0):
    return (zlib.crc32(shard.encode()) * 1000003 + int(index) * 7919 + salt) % (2**63)


# ----------------------------------------------------------------------------- fitting

_ARR = {}


def arrays(shard_dir, shard):
    if shard not in _ARR:
        d = Path(shard_dir) / f"{shard}.npyd"
        _ARR[shard] = tuple(np.load(d / f"{k}.npy", mmap_mode="r") for k in ("time", "mag", "err", "offsets"))
    return _ARR[shard]


def load_star(shard_dir, shard, i):
    t, m, e, off = arrays(shard_dir, shard)
    a, b = int(off[i]), int(off[i + 1])
    t, m, e = np.array(t[a:b], float), np.array(m[a:b], float), np.array(e[a:b], float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e)
    t, m, e = t[ok], m[ok], e[ok]
    good = e > 0
    if good.any() and not good.all():
        e[~good] = np.median(e[good])
    if shard in ("kepler_eb", "tess_eb") and len(t) > SUBSAMPLE:
        keep = np.sort(np.random.default_rng(seed_of(shard, i)).choice(len(t), SUBSAMPLE, replace=False))
        t, m, e = t[keep], m[keep], e[keep]
    return t, m, e


def periodic_interp(x, xp, fp):
    return np.interp(x, np.concatenate([xp - 1, xp, xp + 1]), np.tile(fp, 3))


def binned_profile(ph, m):
    """Median profile on 128 * mult sub-bins (mult = largest of 1, 2, 4, 8 keeping >= 8 points
    per sub-bin on average), empty sub-bins filled periodically, then averaged down to 128 bins
    (area-preserving, so eclipses narrower than one bin keep their integrated depth)."""
    mult = max((x for x in (1, 2, 4, 8) if len(m) / (NBINS * x) >= DENSE_PER_BIN), default=1)
    sub, nfilled = bin_medians(ph, m, NBINS * mult)
    prof = sub.reshape(NBINS, mult).mean(1)
    if mult == 1 and len(m) / NBINS < SMOOTH_BELOW:
        prof = 0.25 * np.roll(prof, 1) + 0.5 * prof + 0.25 * np.roll(prof, -1)
    return prof, nfilled


def bin_medians(ph, m, nb):
    """Median of m in nb equal phase bins; empty bins filled by periodic linear interpolation.
    Returns (medians at bin centres, number of filled bins)."""
    k = np.minimum((ph * nb).astype(int), nb - 1)
    order = np.argsort(k, kind="stable")
    ks, ms = k[order], m[order]
    starts = np.searchsorted(ks, np.arange(nb + 1))
    sub = np.full(nb, np.nan)
    for j in range(nb):
        if starts[j + 1] > starts[j]:
            sub[j] = np.median(ms[starts[j]:starts[j + 1]])
    have = np.isfinite(sub)
    if not have.all():
        centres = (np.arange(nb) + 0.5) / nb
        sub = periodic_interp(centres, centres[have], sub[have])
    return sub, int(have.sum())


def cross_resid(ph, m, e, K, seed):
    """Cross-fitted template residual. Points are assigned at random to halves A and B; the
    residual of each half is m minus a model built only on the other half and interpolated
    periodically to its phases. Model: with n >= 1024 points the fine phase-binned median
    (n // 16 bins per half, at most 4096, ~8 points per bin); otherwise the star's chosen profile
    model refitted on the half (Fourier at order K, capped at (n_half - 1) // 4 as in
    fourier_profile; binned_profile if K == 0)."""
    n = len(m)
    A = np.random.default_rng(seed).random(n) < 0.5
    res = np.empty(n)
    for test in (A, ~A):
        tr = ~test
        if n >= FINE_MIN_POINTS:
            nb = min(n // (2 * FINE_PER_BIN), FINE_MAX_BINS)
            sub, _ = bin_medians(ph[tr], m[tr], nb)
            xp = (np.arange(nb) + 0.5) / nb
        else:
            Kh = min(K, (int(tr.sum()) - 1) // 4)
            sub = fourier_fixed(ph[tr], m[tr], e[tr], Kh) if Kh >= 1 else binned_profile(ph[tr], m[tr])[0]
            xp = PHASES
        res[test] = m[test] - periodic_interp(ph[test], xp, sub)
    return res


def ls_power(t, y, freqs):
    """Standard-normalised Lomb-Scargle power with a floating mean (unweighted), at freqs."""
    y = y - y.mean()
    yy = np.mean(y * y)
    out = np.empty(len(freqs))
    for k in range(0, len(freqs), 64):
        w = 2 * np.pi * np.outer(t, freqs[k:k + 64])
        c, s = np.cos(w), np.sin(w)
        C, S = c.mean(0), s.mean(0)
        YC, YS = y @ c / len(y), y @ s / len(y)
        CC, SS, CS = (c * c).mean(0) - C * C, (s * s).mean(0) - S * S, (c * s).mean(0) - C * S
        D = CC * SS - CS * CS
        out[k:k + 64] = (SS * YC**2 + CC * YS**2 - 2 * CS * YC * YS) / (yy * D)
    return out


def leak_powers(t, resid, f0):
    """Periodic leak of a residual series: max LS power within +-3/T of k f0 (k = 1..5) and, as the
    template's own control, within +-3/T of (k + 0.5) f0 (61 frequencies per window)."""
    T = np.ptp(t)
    if not T > 0 or len(t) < 10:
        return np.nan, np.nan
    def best(centres):
        f = np.concatenate([np.linspace(max(c - 3 / T, 1e-6), c + 3 / T, 61) for c in centres])
        return float(np.max(ls_power(t, resid, f)))
    k = np.arange(1, LEAK_HARMONICS + 1)
    return best(k * f0), best((k + 0.5) * f0)


def leak_flag(power, control):
    return bool(np.isfinite(power) and power > LEAK_POWER and power > LEAK_RATIO * control)


def outlier_ok(resid):
    """< 2 % of points beyond 8 robust sigma and max |resid| below max(1 mag, 15 robust sigma),
    robust sigma = 1.4826 MAD (mag)."""
    med = np.median(resid)
    sig = 1.4826 * np.median(np.abs(resid - med))
    if not sig > 0:
        return False
    return bool(np.mean(np.abs(resid - med) > 8 * sig) < 0.02 and np.max(np.abs(resid)) < max(1.0, 15 * sig))


def template_quality(resid, power, control):
    """Good template: passes the outlier rule (outlier_ok) and is not flagged by the periodic-leak
    rule (leak power > 0.1 and > 4 x its own off-harmonic control). Returns (good, outlier_bad, leak_bad)."""
    ob, lb = not outlier_ok(resid), leak_flag(power, control)
    return not (ob or lb), ob, lb


def design(ph, K):
    w = 2 * np.pi * np.outer(ph, np.arange(1, K + 1))
    return np.hstack([np.ones((len(ph), 1)), np.cos(w), np.sin(w)])


def fourier_profile(ph, m, e):
    """Weighted Fourier fit, order by BIC among non-ringing orders. Returns (profile, K, bic_K)."""
    N = len(m)
    Kmax = min(MAX_ORDER, (N - 1) // 4)
    if Kmax < 1:
        return None, 0, 0
    w = 1 / e
    y = m - np.average(m, weights=w**2)
    full = design(ph, Kmax)
    cols = lambda K: np.r_[0, 1:K + 1, Kmax + 1:Kmax + K + 1]
    span = y.max() - y.min()
    best, best_any = None, None
    for K in range(1, Kmax + 1):
        A = full[:, cols(K)]
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        chi2 = np.sum(((y - A @ c) * w) ** 2)
        bic = N * np.log(max(chi2, 1e-300) / N) + (2 * K + 1) * np.log(N)   # error scale fitted, not trusted
        if best_any is None or bic < best_any[0]:
            best_any = (bic, K)
        prof = design(PHASES, K) @ c
        if np.ptp(prof) <= span and (best is None or bic < best[0]):
            best = (bic, K, prof)
    if best is None:
        return None, 0, best_any[1]
    return best[2] + np.average(m, weights=w**2), best[1], best_any[1]


def submultiple(prof, frac=0.8):
    """Largest k in 2..8 such that >= frac of the profile's harmonic power sits in harmonics
    divisible by k (the profile then repeats k times per label period); 1 if none."""
    pw = np.abs(np.fft.rfft(prof - prof.mean()))[1:] ** 2
    h = np.arange(1, len(pw) + 1)
    tot = pw.sum()
    ks = [k for k in range(2, 9) if tot > 0 and pw[h % k == 0].sum() >= frac * tot]
    return max(ks) if ks else 1


def fourier_fixed(ph, m, e, K):
    """Weighted Fourier fit of fixed order K, returned on the 128 profile phases."""
    w = 1 / e
    c, *_ = np.linalg.lstsq(design(ph, K) * w[:, None], m * w, rcond=None)
    return design(PHASES, K) @ c


def choose_thin(ph, m, e, seed):
    """Thin binned regime (8-32 points per bin): binned median vs Fourier (order chosen on all
    points by fourier_profile), chosen by 5-fold cross-validation: the points are split at random
    into 5 folds; each candidate is rebuilt on 4/5 (binned_profile as stored; Fourier at the fixed
    order) and scored by sum(((m - model) / e)^2) on the held-out 1/5, the model being the
    128-point profile interpolated periodically. Lower total kept. If the Fourier fit has no
    eligible (non-ringing) order, binned is kept.
    Returns (profile, K, Kbic, dof, alternative profile, switched)."""
    N = len(m)
    pb, nfilled = binned_profile(ph, m)
    pf, K, Kbic = fourier_profile(ph, m, e)
    if pf is None or not np.all(np.isfinite(pf)):
        return pb, 0, Kbic, max(N - nfilled, 1), None, False
    folds = np.array_split(np.random.default_rng(seed).permutation(N), CV_FOLDS)
    cv = np.zeros(2)
    for f in folds:
        tr = np.ones(N, bool); tr[f] = False
        for j, model in enumerate((binned_profile(ph[tr], m[tr])[0], fourier_fixed(ph[tr], m[tr], e[tr], K))):
            cv[j] += np.sum(((m[f] - periodic_interp(ph[f], PHASES, model)) / e[f]) ** 2)
    if np.isfinite(cv[1]) and cv[1] < cv[0]:
        return pf, K, Kbic, max(N - (2 * K + 1), 1), pb, True
    return pb, 0, Kbic, max(N - nfilled, 1), pf, False


def extract(t, m, e, freq, seed=0):
    """Profile record for one star; dict with 'fail' reason on failure."""
    N = len(t)
    if N < 10:
        return {"fail": "too_few_points", "n_points": N}
    ph = ((t - t[0]) * freq) % 1.0
    alt, thin, switched = None, False, False
    if N / NBINS >= SMOOTH_BELOW:
        prof, nfilled = binned_profile(ph, m)
        K, Kbic, dof = 0, 0, max(N - nfilled, 1)
    elif N / NBINS >= DENSE_PER_BIN:
        thin = True
        prof, K, Kbic, dof, alt, switched = choose_thin(ph, m, e, seed)
    else:
        prof, K, Kbic = fourier_profile(ph, m, e)
        if prof is None:
            return {"fail": "no_order" if Kbic == 0 else "ringing", "n_points": N}
        dof = max(N - (2 * K + 1), 1)
    if not np.all(np.isfinite(prof)):
        return {"fail": "nonfinite", "n_points": N}
    amp = float(prof.max() - prof.min())
    if not amp > 0:
        return {"fail": "flat", "n_points": N}
    resid = m - periodic_interp(ph, PHASES, prof)
    mederr = float(np.median(e))
    roll = int(np.argmax(prof))
    out = {
        "profile": np.roll((prof - prof.min()) / amp, -roll).astype(np.float32),
        "amplitude": amp, "order": K, "order_bic": Kbic, "n_points": N,
        "harmonic": submultiple(prof),
        "chi2r": float(np.sum((resid / e) ** 2) / dof),
        "snr": amp / (mederr / np.sqrt(N)),
        "scatter": robust_scatter(resid, mederr),
        "scatter_rms": float(np.sqrt(np.mean(resid**2)) / mederr),
        "resid": resid.astype(np.float32),
        "mag": float(np.median(m)),
        "phase0": (roll + 0.5) / NBINS,          # phase of the faintest point (for plots)
        "prof_min": float(prof.min()),
        "thin": thin, "switched": switched,
    }
    if alt is not None:                          # other thin-regime candidate, same scaling (plots)
        out["alt"] = (np.roll(alt, -roll) - prof.min()) / amp
    return out


def robust_scatter(resid, mederr):
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))) / mederr)


def work(job):
    shard_dir, shard, items = job
    out = []
    for i, freq, want_template in items:
        try:
            t, m, e = load_star(shard_dir, shard, i)
            if freq is None:            # unlabelled template (vvv_quiet): residual about median
                if len(t) < 10:
                    r = {"fail": "too_few_points", "n_points": len(t)}
                else:
                    res = m - np.median(m)
                    r = {"n_points": len(t), "mag": float(np.median(m)), "resid": res.astype(np.float32),
                         "scatter": robust_scatter(res, float(np.median(e))),
                         "scatter_rms": float(np.sqrt(np.mean(res**2)) / np.median(e))}
            else:
                r = extract(t, m, e, freq, seed_of(shard, i, 1))
                if want_template and "fail" not in r:
                    res = cross_resid(((t - t[0]) * freq) % 1.0, m, e, r["order"], seed_of(shard, i, 2))
                    r["resid"] = res.astype(np.float32)
                    r["scatter"] = robust_scatter(res, float(np.median(e)))
                    r["scatter_rms"] = float(np.sqrt(np.mean(res**2)) / np.median(e))
                    r["leak_power"], r["leak_control"] = leak_powers(t - t[0], res, freq)
        except Exception as ex:          # recorded and counted, never silent
            r = {"fail": f"error:{type(ex).__name__}:{ex}"[:200], "n_points": -1}
        r["shard"], r["index"] = shard, i
        if want_template and "fail" not in r:
            r["t"] = t - t[0]
            r["e"] = e.astype(np.float32)
            r.setdefault("leak_power", np.nan); r.setdefault("leak_control", np.nan)
            r["quality"], r["q_outlier"], r["q_leak"] = template_quality(
                r["resid"].astype(float), r["leak_power"], r["leak_control"])
        else:
            r.pop("resid", None)         # only templates keep their residual series
        out.append(r)
    return out


# ----------------------------------------------------------------------------- main build

def load_meta(shard_dir):
    shard_dir = Path(shard_dir)
    shards = sorted(p.name[:-len("_meta.json")] for p in shard_dir.glob("*_meta.json"))
    meta, labels = {}, {}
    for s in shards:
        meta[s] = json.load(open(shard_dir / f"{s}_meta.json"))
        lp = shard_dir / f"{s}_labels.json"
        labels[s] = {int(r["index"]): float(r["freq"]) for r in json.load(open(lp))} if lp.exists() else {}
    return meta, labels


def split_groups(meta):
    bad = {}
    for s, entries in meta.items():
        for e in entries:
            fam, b = base_id(s, e)
            bad.setdefault(fam, {})
            bad[fam][b] = bad[fam].get(b, False) or e["split"] != "train"
    return bad


def build(a):
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, labels = load_meta(a.shards)
    ASASSN_CLASSES.update({str(k): str(v) for k, v in json.load(open(a.asassn_classes)).items()})
    bad = split_groups(meta)
    excluded = {fam: sorted(b for b, x in ids.items() if x) for fam, ids in bad.items()}
    print(f"[{time.time()-t0:.0f}s] meta loaded; excluded base IDs: "
          f"{ {f: len(v) for f, v in excluded.items()} }", flush=True)

    only = set(a.only.split(",")) if a.only else None
    rng = np.random.default_rng(a.seed)
    class_map, prof_jobs, tmpl_pool = {}, [], {}
    for s, entries in meta.items():
        if only and s not in only:
            continue
        usable = [i for i, e in enumerate(entries) if not bad[base_id(s, e)[0]][base_id(s, e)[1]]]
        for i in usable:
            raw = raw_class(s, entries[i]); class_map[raw] = family(raw)
        lab = [i for i in usable if i in labels[s]]
        if s == "vvv_quiet":
            lab = []
            cand = usable
        else:
            cand = lab
        if a.sample:
            lab = sorted(rng.choice(lab, min(a.sample, len(lab)), replace=False).tolist()) if lab else []
            cand = lab if s != "vvv_quiet" else sorted(rng.choice(cand, min(a.sample, len(cand)), replace=False).tolist())
        prof_jobs += [(s, i) for i in lab]
        for i in cand:
            tmpl_pool.setdefault(survey_token(s, entries[i]), []).append((s, i))

    # template choice: labelled train stars (scatter from the profile fit) plus vvv_quiet
    # train stars; 5 % extra drawn so failed fits can be dropped before trimming to the cap
    chosen = set()
    for tok, pool in sorted(tmpl_pool.items()):
        cap = TEMPLATE_CAP.get(tok, TEMPLATE_CAP_DEFAULT)
        k = min(len(pool), int(cap * 1.05) + 5)
        pick = rng.choice(len(pool), k, replace=False)
        chosen.update(pool[j] for j in pick)
    jobs_by_shard = {}
    for s, i in prof_jobs:
        jobs_by_shard.setdefault(s, []).append((i, labels[s][i], (s, i) in chosen))
    for s, i in chosen:
        if s == "vvv_quiet":
            jobs_by_shard.setdefault(s, []).append((i, None, True))
    jobs = []
    for s, items in jobs_by_shard.items():
        items.sort()
        size = 20 if s in ("kepler_eb", "tess_eb") else 200
        jobs += [(a.shards, s, items[k:k + size]) for k in range(0, len(items), size)]
    jobs.sort(key=lambda j: -len(j[2]) * (50 if j[1] in ("kepler_eb", "tess_eb") else 1))
    print(f"[{time.time()-t0:.0f}s] {len(prof_jobs)} labelled train stars, {len(chosen)} template "
          f"candidates, {len(jobs)} jobs, {a.workers} workers", flush=True)

    results = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            results += r
            if k % max(1, len(jobs) // 20) == 0:
                print(f"[{time.time()-t0:.0f}s] {k+1}/{len(jobs)} jobs", flush=True)
    results.sort(key=lambda r: (r["shard"], r["index"]))
    print(f"[{time.time()-t0:.0f}s] fitting done", flush=True)

    # ---- profiles
    classes = CLASSES
    P = [r for r in results if "profile" in r]
    def sid(r): return str(meta[r["shard"]][r["index"]]["sourceid"])
    def tok(r): return survey_token(r["shard"], meta[r["shard"]][r["index"]])
    prof = {
        "profile": np.array([r["profile"] for r in P], np.float32).reshape(-1, NBINS),
        "period": np.array([1 / labels[r["shard"]][r["index"]] for r in P], np.float64),
        "amplitude": np.array([r["amplitude"] for r in P], np.float32),
        "mag": np.array([np.nan if tok(r) in NO_ABS_MAG else r["mag"] for r in P], np.float32),
        "survey": np.array([SURVEYS.index(tok(r)) for r in P], np.int16),
        "cls": np.array([classes.index(family(raw_class(r["shard"], meta[r["shard"]][r["index"]]))) for r in P], np.int16),
        "tier": np.array([tier(r["shard"]) for r in P], np.int8),
        "snr": np.array([r["snr"] for r in P], np.float32),
        "chi2r": np.array([r["chi2r"] for r in P], np.float32),
        "order": np.array([r["order"] for r in P], np.int16),
        "repeats": np.array([r["harmonic"] for r in P], np.int8),
        "n_points": np.array([r["n_points"] for r in P], np.int32),
        "shard": np.array([r["shard"] for r in P], dtype="U"),
        "index": np.array([r["index"] for r in P], np.int32),
        "sourceid": np.array([sid(r) for r in P], dtype="U"),
    }
    np.savez(out / "profiles.npz", **prof)

    # ---- templates
    T = [r for r in results if "t" in r]
    by_tok = {}
    for r in T:
        by_tok.setdefault(tok(r), []).append(r)
    T = []
    for tk in sorted(by_tok):
        rs = by_tok[tk]
        cap = TEMPLATE_CAP.get(tk, TEMPLATE_CAP_DEFAULT)
        if len(rs) > cap:
            keep = np.sort(rng.choice(len(rs), cap, replace=False))
            rs = [rs[j] for j in keep]
        T += rs
    offsets = np.concatenate([[0], np.cumsum([len(r["t"]) for r in T])]).astype(np.int64)
    np.savez(out / "templates.npz",
             time=np.concatenate([r["t"] for r in T]).astype(np.float64) if T else np.zeros(0),
             err=np.concatenate([r["e"] for r in T]).astype(np.float32) if T else np.zeros(0, np.float32),
             offsets=offsets,
             survey=np.array([SURVEYS.index(tok(r)) for r in T], np.int16),
             mag=np.array([np.nan if tok(r) in NO_ABS_MAG else r["mag"] for r in T], np.float32),
             resid=np.concatenate([r["resid"] for r in T]).astype(np.float32) if T else np.zeros(0, np.float32),
             scatter=np.array([r["scatter"] for r in T], np.float32),
             scatter_rms=np.array([r["scatter_rms"] for r in T], np.float32),
             quality=np.array([r["quality"] for r in T], bool),
             leak_power=np.array([r["leak_power"] for r in T], np.float32),
             leak_control=np.array([r["leak_control"] for r in T], np.float32),
             sourceid=np.array([sid(r) for r in T], dtype="U"),
             shard=np.array([r["shard"] for r in T], dtype="U"),
             index=np.array([r["index"] for r in T], np.int32))

    json.dump({"surveys": SURVEYS, "classes": classes, "class_map": dict(sorted(class_map.items())),
               "nbins": NBINS}, open(out / "vocab.json", "w"), indent=1)
    json.dump(excluded, open(out / "excluded.json", "w"))
    if a.check:
        save_check(a, P, T, meta, labels)

    summary = summarise(results, P, prof, T, excluded, meta, classes, tok, time.time() - t0)
    summary.update(summarise_v2(P, prof, T, meta, classes, tok))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    verify(out)
    print(json.dumps(summary, indent=1))
    print(f"[{time.time()-t0:.0f}s] done", flush=True)


def summarise(results, P, prof, T, excluded, meta, classes, tok, runtime):
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    labelled = [r for r in results if r["shard"] != "vvv_quiet" and ("profile" in r or "fail" in r)]
    s = {"runtime_s": round(runtime, 1), "n_profiles": len(P), "n_templates": len(T),
         "excluded_base_ids": {f: len(v) for f, v in excluded.items()},
         "excluded_total": sum(len(v) for v in excluded.values()),
         "profiles_per_survey": {}, "profiles_per_class": {}, "profiles_per_survey_class": {},
         "templates_per_survey": {}, "fit_failures": {}, "quantiles": {"q": q},
         "binned_fraction": {}, "repeats_ge3_fraction": {},
         "ringing_guard_changed_order": 0}
    for k, name in enumerate(SURVEYS):
        sel = prof["survey"] == k
        if not sel.any():
            continue
        s["profiles_per_survey"][name] = int(sel.sum())
        s["profiles_per_survey_class"][name] = {classes[c]: int(n) for c, n in
                                                zip(*np.unique(prof["cls"][sel], return_counts=True))}
        s["binned_fraction"][name] = round(float(np.mean(prof["order"][sel] == 0)), 4)
        h = np.array([r["harmonic"] for r in P])[sel]
        s["repeats_ge3_fraction"][name] = round(float(np.mean(h >= 3)), 4)
        s["quantiles"][name] = {key: [float(x) for x in np.nanquantile(prof[key][sel], q)]
                                if np.isfinite(prof[key][sel]).any() else None
                                for key in ("period", "amplitude", "mag", "snr", "chi2r", "n_points")}
    for c, n in zip(*np.unique(prof["cls"], return_counts=True)):
        s["profiles_per_class"][classes[c]] = int(n)
    for r in T:
        s["templates_per_survey"][tok(r)] = s["templates_per_survey"].get(tok(r), 0) + 1
    tot, fails = {}, {}
    for r in labelled:
        k = tok(r)
        tot[k] = tot.get(k, 0) + 1
        if "fail" in r:
            reason = r["fail"].split(":")[0]
            fails.setdefault(k, {}).setdefault(reason, 0)
            fails[k][reason] += 1
    for k in tot:
        nf = sum(fails.get(k, {}).values())
        s["fit_failures"][k] = {"attempted": tot[k], "failed": nf, "fraction": round(nf / tot[k], 5),
                                "reasons": fails.get(k, {})}
    nall = sum(tot.values())
    s["fit_failures"]["all"] = {"attempted": nall, "failed": nall - len(P),
                                "fraction": round((nall - len(P)) / max(nall, 1), 5)}
    s["ringing_guard_changed_order"] = int(sum(1 for r in P if r["order"] > 0 and r["order"] != r["order_bic"]))
    errs = [r["fail"] for r in results if r.get("fail", "").startswith("error")]
    s["errors_example"] = errs[:5]
    return s


def summarise_v2(P, prof, T, meta, classes, tok):
    """Bank v2 additions: robust vs rms scatter, thin-regime choices, repeats, vvv_asassn classes."""
    s = {"scatter_per_survey": {}, "thin_regime": {}, "thin_regime_per_class": {},
         "repeats_distribution": {}, "vvv_asassn_classes": {}, "template_quality": {}}
    def stats(x):
        x = np.asarray(x, float)
        return {"n": int(len(x)), "median": round(float(np.median(x)), 4),
                "frac_gt3": round(float(np.mean(x > 3)), 4), "frac_gt10": round(float(np.mean(x > 10)), 4),
                "n_zero": int(np.sum(x == 0))}
    groups = {}
    for r in T:
        groups.setdefault(tok(r), []).append(r)
        if r["shard"] == "vvv_quiet":
            groups.setdefault("VVV-Ks (vvv_quiet)", []).append(r)
    for g, rs in sorted(groups.items()):
        s["scatter_per_survey"][g] = {"robust": stats([r["scatter"] for r in rs]),
                                      "rms_old": stats([r["scatter_rms"] for r in rs])}
        ng = sum(r["quality"] for r in rs)
        s["template_quality"][g] = {"templates": len(rs), "good": ng,
                                    "flagged_fraction": round(1 - ng / len(rs), 4),
                                    "outlier_only": sum(r["q_outlier"] and not r["q_leak"] for r in rs),
                                    "leak_only": sum(r["q_leak"] and not r["q_outlier"] for r in rs),
                                    "both": sum(r["q_leak"] and r["q_outlier"] for r in rs)}
    for r in P:
        if r["thin"]:
            c = family(raw_class(r["shard"], meta[r["shard"]][r["index"]]))
            d = s["thin_regime_per_class"].setdefault(c, {"thin": 0, "fourier_kept": 0, "binned_kept": 0})
            d["thin"] += 1
            d["fourier_kept" if r["switched"] else "binned_kept"] += 1
    for k, name in enumerate(SURVEYS):
        rs = [r for r in P if SURVEYS.index(tok(r)) == k and r["thin"]]
        if rs:
            ns = sum(r["switched"] for r in rs)
            s["thin_regime"][name] = {"thin": len(rs), "fourier_kept": ns, "binned_kept": len(rs) - ns}
        sel = prof["survey"] == k
        if sel.any():
            v, n = np.unique(prof["repeats"][sel], return_counts=True)
            s["repeats_distribution"][name] = {int(a): int(b) for a, b in zip(v, n)}
    rs = [r for r in P if r["thin"]]
    s["thin_regime"]["all"] = {"thin": len(rs), "fourier_kept": sum(r["switched"] for r in rs),
                               "binned_kept": sum(not r["switched"] for r in rs)}
    v, n = np.unique(prof["repeats"], return_counts=True)
    s["repeats_distribution"]["all"] = {int(a): int(b) for a, b in zip(v, n)}
    va = [r for r in P if r["shard"] == "vvv_asassn"]
    raws = [raw_class("vvv_asassn", meta["vvv_asassn"][r["index"]]) for r in va]
    fam, rawc = {}, {}
    for x in raws:
        fam[family(x)] = fam.get(family(x), 0) + 1
        rawc[x[len("vvv_asassn/"):] or "(no class)"] = rawc.get(x[len("vvv_asassn/"):] or "(no class)", 0) + 1
    s["vvv_asassn_classes"] = {"profiles": len(va), "family": dict(sorted(fam.items())),
                               "raw": dict(sorted(rawc.items(), key=lambda kv: -kv[1])),
                               "unknown": fam.get("UNKNOWN", 0)}
    return s


def verify(out):
    """Reload outputs and assert the DESIGN.md contract."""
    p = np.load(out / "profiles.npz")
    want = {"profile": "float32", "period": "float64", "amplitude": "float32", "mag": "float32",
            "survey": "int16", "cls": "int16", "tier": "int8", "snr": "float32", "chi2r": "float32",
            "order": "int16", "repeats": "int8", "n_points": "int32", "index": "int32"}
    for k, dt in want.items():
        assert p[k].dtype == np.dtype(dt), (k, p[k].dtype)
    assert p["shard"].dtype.kind == "U" and p["sourceid"].dtype.kind == "U"
    N = len(p["period"])
    assert p["profile"].shape == (N, NBINS)
    pr = p["profile"]
    if N:
        assert np.all(pr.min(1) == 0) and np.all(pr.max(1) == 1) and np.all(np.argmax(pr, 1) == 0)
        assert np.all(p["amplitude"] > 0) and np.all(p["period"] > 0)
        assert np.all((p["repeats"] >= 1) & (p["repeats"] <= 8))
    for k in p.files:
        assert len(p[k]) == N, k
    t = np.load(out / "templates.npz")
    assert t["time"].dtype == np.float64 and t["err"].dtype == np.float32 and t["offsets"].dtype == np.int64
    assert t["offsets"][-1] == len(t["time"]) == len(t["err"])
    M = len(t["offsets"]) - 1
    for k, dt in (("survey", "int16"), ("mag", "float32"), ("scatter", "float32"), ("scatter_rms", "float32")):
        assert t[k].dtype == np.dtype(dt) and len(t[k]) == M, k
    assert t["resid"].dtype == np.float32 and len(t["resid"]) == len(t["time"])
    assert np.all(np.isfinite(t["resid"])) and np.all(np.isfinite(t["scatter_rms"]))
    assert t["quality"].dtype == np.bool_ and len(t["quality"]) == M
    lab = t["shard"] != "vvv_quiet"
    for k in ("leak_power", "leak_control"):
        assert t[k].dtype == np.float32 and len(t[k]) == M, k
        assert np.all(np.isfinite(t[k][lab])) and np.all(np.isnan(t[k][~lab])), k
    assert t["sourceid"].dtype.kind == "U" and len(t["sourceid"]) == M
    starts = t["time"][t["offsets"][:-1]]
    assert np.all(starts == 0)
    assert np.all(np.isfinite(t["scatter"])) and np.all(t["err"] > 0)
    # no excluded star anywhere
    exc = json.load(open(out / "excluded.json"))
    allx = {f"{f}:{b}" for f, v in exc.items() for b in v}
    meta_fam = lambda shard, s: (base_id(shard, {"sourceid": s}))
    for sh, si in zip(p["shard"], p["sourceid"]):
        f, b = meta_fam(sh, si)
        assert f"{f}:{b}" not in allx
    print(f"verify: OK ({N} profiles, {M} templates)", flush=True)


# ----------------------------------------------------------------------------- checks

def save_check(a, P, T, meta, labels):
    """Check data for plotting: --check-n stars per shard (spread over n_points) plus 6 vvv_asassn
    stars (different classes first), 6 thin-regime stars switched to Fourier, 6 stars with
    repeats >= 3, 10 random thin-regime stars of --check-class and the --check-ids stars; residual
    templates: one per survey token, one vvv_quiet, 2 random --check-class templates per survey
    token and the --check-ids templates, each with the template residual and the profile-fit
    residual (for comparison)."""
    rng = np.random.default_rng(0)
    picks, seen = [], set()
    def add(r, why):
        if (r["shard"], r["index"]) not in seen:
            seen.add((r["shard"], r["index"])); picks.append((r, why))
    def spread(rs, n, why):
        if rs:
            rs = sorted(rs, key=lambda r: r["n_points"])
            for j in np.unique(np.linspace(0, len(rs) - 1, n).round().astype(int)):
                add(rs[j], why)
    per = {}
    for r in P:
        per.setdefault(r["shard"], []).append(r)
    for sh, rs in sorted(per.items()):
        spread(rs, a.check_n, "shard")
    va = {}
    for r in per.get("vvv_asassn", []):
        va.setdefault(family(raw_class("vvv_asassn", meta["vvv_asassn"][r["index"]])), []).append(r)
    order = sorted(va, key=lambda f: -len(va[f]))
    for k in range(6):
        if order:
            f = order[k % len(order)]
            add(va[f][int(rng.integers(len(va[f])))], "vvv_asassn")
    spread([r for r in P if r["switched"]], 6, "switched")
    spread([r for r in P if r["harmonic"] >= 3], 6, "repeats")
    ids = set(x for x in a.check_ids.split(",") if x)
    def cls_of(r): return family(raw_class(r["shard"], meta[r["shard"]][r["index"]]))
    def sid_of(r): return str(meta[r["shard"]][r["index"]]["sourceid"])
    for r in P:
        if sid_of(r) in ids:
            add(r, "id")
    thin = [r for r in P if r["thin"] and cls_of(r) == a.check_class]
    for j in (rng.choice(len(thin), min(10, len(thin)), replace=False) if thin else []):
        add(thin[j], "thin " + a.check_class)
    rows = []
    for r, why in picks:
        s = r["shard"]
        t, m, e = load_star(a.shards, s, r["index"])
        f = labels[s][r["index"]]
        ph = (((t - t[0]) * f) - (r["phase0"] - 0.5 / NBINS)) % 1.0
        y = (m - r["prof_min"]) / r["amplitude"]
        if len(ph) > 5000:
            k = np.random.default_rng(0).choice(len(ph), 5000, replace=False)
            ph, y = ph[k], y[k]
        raw = raw_class(s, meta[s][r["index"]])
        info = {"shard": s, "sourceid": str(meta[s][r["index"]]["sourceid"]), "period": 1 / f,
                "amp": r["amplitude"], "order": r["order"], "n": r["n_points"], "snr": r["snr"],
                "chi2r": r["chi2r"], "scatter": r["scatter"], "scatter_rms": r["scatter_rms"],
                "harmonic": r["harmonic"], "why": why, "raw": raw, "family": family(raw),
                "thin": r["thin"], "switched": r["switched"]}
        rows.append((info, r["profile"], r.get("alt", np.full(NBINS, np.nan)), ph, y))
    tpick, tseen = [], set()
    def tadd(r, why):
        if (r["shard"], r["index"]) not in tseen:
            tseen.add((r["shard"], r["index"])); tpick.append((r, why))
    by, bycls = {}, {}
    for r in T:
        g = "vvv_quiet" if r["shard"] == "vvv_quiet" else survey_token(r["shard"], meta[r["shard"]][r["index"]])
        by.setdefault(g, []).append(r)
        if r["shard"] != "vvv_quiet" and cls_of(r) == a.check_class:
            bycls.setdefault(g, []).append(r)
    for r in T:
        if sid_of(r) in ids:
            tadd(r, "id")
    for g in sorted(by):
        tadd(by[g][int(rng.integers(len(by[g])))], g)
    leak = [r for r in T if r["q_leak"] and not r["q_outlier"]]
    for j in (rng.choice(len(leak), min(6, len(leak)), replace=False) if leak else []):
        tadd(leak[j], "leak only")
    for g in sorted(bycls):
        for j in rng.choice(len(bycls[g]), min(2, len(bycls[g])), replace=False):
            tadd(bycls[g][j], g + " " + a.check_class)
    trows = []
    for r, why in tpick:
        sh = r["shard"]
        if sh == "vvv_quiet":
            ph, old = np.zeros(len(r["t"])), r["resid"]
        else:
            t, m, e = load_star(a.shards, sh, r["index"])
            f = labels[sh][r["index"]]
            ph = ((t - t[0]) * f) % 1.0
            old = extract(t, m, e, f, seed_of(sh, r["index"], 1))["resid"]
        trows.append(({"group": why, "shard": sh, "sourceid": sid_of(r), "quality": bool(r["quality"]),
                       "q_outlier": bool(r["q_outlier"]), "q_leak": bool(r["q_leak"]),
                       "leak": [float(r["leak_power"]), float(r["leak_control"])],
                       "scatter": r["scatter"], "scatter_rms": r["scatter_rms"],
                       "mederr": float(np.median(r["e"]))}, r["t"], r["resid"], ph, old))
    np.savez(a.check, n=len(rows), nt=len(trows),
             **{f"meta{k}": json.dumps(x[0]) for k, x in enumerate(rows)},
             **{f"prof{k}": x[1] for k, x in enumerate(rows)},
             **{f"alt{k}": x[2] for k, x in enumerate(rows)},
             **{f"ph{k}": x[3] for k, x in enumerate(rows)},
             **{f"y{k}": x[4] for k, x in enumerate(rows)},
             **{f"tmeta{k}": json.dumps(x[0]) for k, x in enumerate(trows)},
             **{f"tt{k}": x[1] for k, x in enumerate(trows)},
             **{f"tr{k}": x[2] for k, x in enumerate(trows)},
             **{f"tph{k}": x[3] for k, x in enumerate(trows)},
             **{f"tro{k}": x[4] for k, x in enumerate(trows)})
    print(f"check data: {a.check} ({len(rows)} stars, {len(trows)} residual templates)", flush=True)


def plot(a):
    """Montages: folded data + kept profile (red) + other thin-regime candidate (blue dashed),
    12 stars per page; one page of residual series against time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(a.plot)
    out = Path(a.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, per = int(d["n"]), 12
    for page in range(0, n, per):
        fig, axs = plt.subplots(4, 3, figsize=(15, 13))
        for ax, k in zip(axs.flat, range(page, min(page + per, n))):
            m = json.loads(str(d[f"meta{k}"]))
            ax.plot(d[f"ph{k}"], d[f"y{k}"], ".", ms=1.5, color="0.5", alpha=0.5)
            if np.isfinite(d[f"alt{k}"]).all():
                ax.plot(PHASES, d[f"alt{k}"], "--", color="C0", lw=1.2,
                        label="binned (rejected)" if m["switched"] else "Fourier (rejected)")
                ax.legend(fontsize=6, loc="lower right")
            ax.plot(PHASES, d[f"prof{k}"], "-", color="C3", lw=1.5)
            lo, hi = np.nanpercentile(d[f"y{k}"], [0.5, 99.5])
            ax.set_ylim(max(hi, 1.1) + 0.05, min(lo, -0.1) - 0.05)
            ax.set_title(f"[{m['why']}] {m['shard']} {m['sourceid']} {m['raw']}->{m['family']}\n"
                         f"P={m['period']:.5g} amp={m['amp']:.3f} K={m['order']} n={m['n']} "
                         f"chi2r={m['chi2r']:.2f} s={m['scatter']:.2f} s_rms={m['scatter_rms']:.2f} "
                         f"rep={m['harmonic']}", fontsize=7)
        for ax in axs.flat[min(per, n - page):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"montage_profiles_{page // per}.png", dpi=80)
        plt.close(fig)
    nt, per = int(d["nt"]), 6
    for page in range(0, nt, per):
        ks = list(range(page, min(page + per, nt)))
        fig, axs = plt.subplots(len(ks), 2, figsize=(16, 2.4 * len(ks)), squeeze=False,
                                gridspec_kw={"width_ratios": [3, 1]})
        for row, k in zip(axs, ks):
            m = json.loads(str(d[f"tmeta{k}"]))
            t, r, ph, ro = d[f"tt{k}"], d[f"tr{k}"], d[f"tph{k}"], d[f"tro{k}"]
            for ax, x in zip(row, (t, ph)):
                ax.plot(x, ro, ".", ms=1, color="C1", alpha=0.4, label="profile-fit resid")
                ax.plot(x, r, ".", ms=1, color="k", alpha=0.5, label="template resid")
                v = m["scatter"] * m["mederr"]
                ax.axhline(v, color="C3", lw=0.8); ax.axhline(-v, color="C3", lw=0.8)
                lo, hi = np.percentile(np.r_[r, ro], [0.2, 99.8])
                ax.set_ylim(hi + 0.2 * (hi - lo), lo - 0.2 * (hi - lo))
            row[0].set_ylabel("resid (mag)", fontsize=7)
            row[0].set_title(f"[{m['group']}] {m['shard']} {m['sourceid']} n={len(t)} median err={m['mederr']:.4f} "
                             f"robust s={m['scatter']:.2f} rms s={m['scatter_rms']:.2f} "
                             f"quality={'good' if m['quality'] else 'FLAGGED'}"
                             f"{' outlier' if m['q_outlier'] else ''}{' leak' if m['q_leak'] else ''} "
                             f"leak={m['leak'][0]:.3f} ctrl={m['leak'][1]:.3f}", fontsize=8)
            row[1].set_title("vs phase at label period", fontsize=7)
        axs[0, 0].legend(fontsize=6, loc="upper right", markerscale=6)
        axs[-1, 0].set_xlabel("time - first time (d)"); axs[-1, 1].set_xlabel("phase")
        fig.tight_layout()
        fig.savefig(out / f"montage_residuals_{page // per}.png", dpi=80)
        plt.close(fig)
    print(f"plots: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="/net/scratch/nillmill/nnp/shards")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--sample", type=int, default=0, help="labelled stars per shard (test runs)")
    ap.add_argument("--only", default="", help="comma-separated shard names (test runs)")
    ap.add_argument("--check", default="", help="write folded data for check plots to this .npz")
    ap.add_argument("--check-n", type=int, default=6, help="check stars per shard")
    ap.add_argument("--check-ids", default="", help="comma-separated sourceids to add to the check plots")
    ap.add_argument("--check-class", default="ECL", help="class family for the random thin / template check picks")
    ap.add_argument("--asassn-classes", default="/net/scratch/nillmill/lcddpm/vvv_asassn_classes.json",
                    help="JSON: VIRAC sourceid -> ASAS-SN class (vvv_asassn)")
    ap.add_argument("--plot", default="", help="render a --check .npz to PNGs (needs matplotlib)")
    ap.add_argument("--plot-dir", default="check")
    a = ap.parse_args()
    if a.plot:
        return plot(a)
    if not a.out:
        sys.exit("--out is required")
    build(a)


if __name__ == "__main__":
    main()
