"""Build the LC_DDPM training bank from NN_Periodogram shards (contract: docs/DESIGN.md).

Writes to --out:
  profiles.npz   one 128-point phase profile per labelled train star
  templates.npz  real cadence + error templates (ragged) with scatter ratios
  vocab.json     survey tokens, class families and the raw -> family map
  excluded.json  base star IDs left out because some entry is in val or test
  summary.json   counts and distributions

Split rule: entries are grouped by base star ID within an ID family (ASAS-SN band
suffix removed; all OGLE shards share one family; vvv_real, vvv_asassn and
vvv_quiet share the VIRAC family). A star is used only if every entry is "train".

Profiles: fold at the label period (1/freq from {shard}_labels.json). With at least
8 points per bin on average the profile is a binned median (see binned_profile; a
[1,2,1]/4 smoothing when there are fewer than 32 points per bin). Otherwise a weighted
truncated Fourier series with order chosen by BIC (error scale fitted: N ln(chi2/N) +
k ln N, because quoted errors are often underestimated) up to min(15, (N-1)//4); orders whose model peak-to-peak exceeds the data range (ringing
in phase gaps) are not eligible. Profile convention: magnitudes scaled to [0, 1],
faintest point at index 0.

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
    if shard == "vvv_asassn":
        return "vvv_asassn/"            # no class in the shard metadata
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
    mult = max(x for x in (1, 2, 4, 8) if len(m) / (NBINS * x) >= DENSE_PER_BIN)
    nb = NBINS * mult
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
    prof = sub.reshape(NBINS, mult).mean(1)
    if mult == 1 and len(m) / NBINS < SMOOTH_BELOW:
        prof = 0.25 * np.roll(prof, 1) + 0.5 * prof + 0.25 * np.roll(prof, -1)
    return prof, int(have.sum())


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


def extract(t, m, e, freq):
    """Profile record for one star; dict with 'fail' reason on failure."""
    N = len(t)
    if N < 10:
        return {"fail": "too_few_points", "n_points": N}
    ph = ((t - t[0]) * freq) % 1.0
    if N / NBINS >= DENSE_PER_BIN:
        prof, nfilled = binned_profile(ph, m)
        K, Kbic, dof = 0, 0, max(N - nfilled, 1)
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
        "scatter": float(np.sqrt(np.mean(resid**2)) / mederr),
        "mag": float(np.median(m)),
        "phase0": (roll + 0.5) / NBINS,          # phase of the faintest point (for plots)
        "prof_min": float(prof.min()),
    }
    return out


def work(job):
    shard_dir, shard, items = job
    out = []
    for i, freq, want_template in items:
        try:
            t, m, e = load_star(shard_dir, shard, i)
            if freq is None:            # unlabelled template (vvv_quiet): rms about median
                if len(t) < 10:
                    r = {"fail": "too_few_points", "n_points": len(t)}
                else:
                    r = {"n_points": len(t), "mag": float(np.median(m)),
                         "scatter": float(np.sqrt(np.mean((m - np.median(m)) ** 2)) / np.median(e))}
            else:
                r = extract(t, m, e, freq)
        except Exception as ex:          # recorded and counted, never silent
            r = {"fail": f"error:{type(ex).__name__}:{ex}"[:200], "n_points": -1}
        r["shard"], r["index"] = shard, i
        if want_template and "fail" not in r:
            r["t"] = t - t[0]
            r["e"] = e.astype(np.float32)
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
    if a.check:
        save_check(a, P, meta, labels)
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
             scatter=np.array([r["scatter"] for r in T], np.float32),
             sourceid=np.array([sid(r) for r in T], dtype="U"),
             shard=np.array([r["shard"] for r in T], dtype="U"),
             index=np.array([r["index"] for r in T], np.int32))

    json.dump({"surveys": SURVEYS, "classes": classes, "class_map": dict(sorted(class_map.items())),
               "nbins": NBINS}, open(out / "vocab.json", "w"), indent=1)
    json.dump(excluded, open(out / "excluded.json", "w"))

    summary = summarise(results, P, prof, T, excluded, meta, classes, tok, time.time() - t0)
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


def verify(out):
    """Reload outputs and assert the DESIGN.md contract."""
    p = np.load(out / "profiles.npz")
    want = {"profile": "float32", "period": "float64", "amplitude": "float32", "mag": "float32",
            "survey": "int16", "cls": "int16", "tier": "int8", "snr": "float32", "chi2r": "float32",
            "order": "int16", "n_points": "int32", "index": "int32"}
    for k, dt in want.items():
        assert p[k].dtype == np.dtype(dt), (k, p[k].dtype)
    assert p["shard"].dtype.kind == "U" and p["sourceid"].dtype.kind == "U"
    N = len(p["period"])
    assert p["profile"].shape == (N, NBINS)
    pr = p["profile"]
    if N:
        assert np.all(pr.min(1) == 0) and np.all(pr.max(1) == 1) and np.all(np.argmax(pr, 1) == 0)
        assert np.all(p["amplitude"] > 0) and np.all(p["period"] > 0)
    for k in p.files:
        assert len(p[k]) == N, k
    t = np.load(out / "templates.npz")
    assert t["time"].dtype == np.float64 and t["err"].dtype == np.float32 and t["offsets"].dtype == np.int64
    assert t["offsets"][-1] == len(t["time"]) == len(t["err"])
    M = len(t["offsets"]) - 1
    for k, dt in (("survey", "int16"), ("mag", "float32"), ("scatter", "float32")):
        assert t[k].dtype == np.dtype(dt) and len(t[k]) == M, k
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

def save_check(a, P, meta, labels):
    """Folded data + profile for the first --check-n stars per shard (for plotting elsewhere)."""
    per = {}
    for r in P:
        per.setdefault(r["shard"], []).append(r)
    rows = []
    for s, rs in per.items():
        rs = sorted(rs, key=lambda r: r["n_points"])      # spread over sparse .. dense
        pick = np.unique(np.linspace(0, len(rs) - 1, a.check_n).round().astype(int))
        for r in (rs[j] for j in pick):
            t, m, e = load_star(a.shards, s, r["index"])
            f = labels[s][r["index"]]
            ph = (((t - t[0]) * f) - (r["phase0"] - 0.5 / NBINS)) % 1.0
            y = (m - r["prof_min"]) / r["amplitude"]
            if len(ph) > 5000:
                k = np.random.default_rng(0).choice(len(ph), 5000, replace=False)
                ph, y = ph[k], y[k]
            rows.append((s, str(meta[s][r["index"]]["sourceid"]), 1 / f, r, ph, y))
    np.savez(a.check, n=len(rows),
             **{f"meta{k}": json.dumps({"shard": s, "sourceid": sid, "period": P_,
                                        "amp": r["amplitude"], "order": r["order"], "n": r["n_points"],
                                        "snr": r["snr"], "chi2r": r["chi2r"], "scatter": r["scatter"],
                                        "harmonic": r["harmonic"]})
                for k, (s, sid, P_, r, ph, y) in enumerate(rows)},
             **{f"prof{k}": r["profile"] for k, (s, sid, P_, r, ph, y) in enumerate(rows)},
             **{f"ph{k}": ph for k, (s, sid, P_, r, ph, y) in enumerate(rows)},
             **{f"y{k}": y for k, (s, sid, P_, r, ph, y) in enumerate(rows)})
    print(f"check data: {a.check} ({len(rows)} stars)", flush=True)


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(a.plot)
    out = Path(a.plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(int(d["n"])):
        m = json.loads(str(d[f"meta{k}"]))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(d[f"ph{k}"], d[f"y{k}"], ".", ms=2, color="0.5", alpha=0.6)
        ax.plot(PHASES, d[f"prof{k}"], "-", color="C3", lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel("phase (faintest at 0)")
        ax.set_ylabel("scaled mag (1 = faintest)")
        ax.set_title(f"{m['shard']} {m['sourceid']} P={m['period']:.5g} d\n"
                     f"amp={m['amp']:.3f} K={m['order']} n={m['n']} snr={m['snr']:.0f} "
                     f"chi2r={m['chi2r']:.2f} s={m['scatter']:.2f} rep={m['harmonic']}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{k:03d}_{m['shard']}_{m['sourceid'].replace(':', '_')}.png", dpi=90)
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
    ap.add_argument("--check-n", type=int, default=6)
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
