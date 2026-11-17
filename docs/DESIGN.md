# LC_DDPM design (v2 rewrite, 2026-11-17)

Goal: `generate(survey, period, amplitude, mag=None, cls=None, n=1, seed=None)` returns
realistic light curves (time, mag, err) whose period is exactly the requested one.
Two uses: (1) labelled synthetic training data for NN_Periodogram, (2) a standalone,
publishable light-curve generator.

## Pipeline

1. **Shape model (diffusion).** A 1-D conditional DDPM generates a normalised
   phase-folded profile of `NBINS = 128` phase points. Conditions, each independently
   droppable (classifier-free guidance), so any subset can be given:
   survey token, class token, log10 period, log10 amplitude, mean magnitude.
2. **Real cadence and noise (template star).** The renderer picks a real
   *training-split* star of the requested survey (nearest in mean magnitude when the
   magnitude is given and known for that survey, otherwise random). Its observation
   times and per-point errors are reused. Noise per point is N(0, (err_i * s)^2),
   where s is that star's measured scatter ratio (residual rms / median err).
3. **Assembly.** `mag(t) = mean_mag + amplitude * (P((t - t0)/period mod 1) - mean(P))`
   plus noise, with P interpolated periodically from the 128 points, random phase t0.

Period is exact by construction; amplitude is exact by scaling; cadence is real.

## Profile convention (shared by data builder and model)

- 128 points at phases (k + 0.5)/128, k = 0..127.
- In magnitudes (larger = fainter). Scaled so max - min = 1 and min = 0 (so values in [0, 1]).
- Rolled so that the faintest point (max magnitude, = 1) is at index 0
  (primary eclipse for eclipsing binaries, minimum light for pulsators).
- Amplitude = peak-to-peak (max - min) of the fitted profile, in magnitudes.
- Model input is `2 * profile - 1` (range [-1, 1]).

## Data products (built by `training/build_bank.py` from NN_Periodogram shards)

All written to one directory (Aurora: `/net/scratch/nillmill/lcddpm/bank/`), never to git.

### `profiles.npz`
| key | dtype/shape | meaning |
|---|---|---|
| `profile` | float32 (N, 128) | profile, convention above |
| `period` | float64 (N,) | days, from `{shard}_labels.json` (1/freq) |
| `amplitude` | float32 (N,) | mag, peak-to-peak of fit |
| `mag` | float32 (N,) | median magnitude; NaN where the shard has no absolute magnitude (ASAS-SN, Kepler, TESS) |
| `survey` | int16 (N,) | index into `vocab.json` `surveys` |
| `cls` | int16 (N,) | index into `vocab.json` `classes` (class family) |
| `tier` | int8 (N,) | label quality: 0 = catalogue period (OGLE, ASAS-SN, Kepler, TESS, vvv_asassn); 1 = self-labelled (vvv_real) |
| `snr` | float32 (N,) | amplitude / (median err / sqrt(n_points)) style fit S/N |
| `chi2r` | float32 (N,) | reduced chi-square of the fit |
| `order` | int16 (N,) | Fourier order used (0 = binned median profile) |
| `n_points` | int32 (N,) | points in the light curve |
| `shard`, `index` | str / int32 | provenance |
| `sourceid` | str | provenance |

### `templates.npz` (cadence + noise bank, ragged)
| key | meaning |
|---|---|
| `time` | float64, concatenated observation times minus first time (days) |
| `err` | float32, concatenated per-point errors (mag) |
| `offsets` | int64 (M+1,) |
| `survey` | int16 (M,) |
| `mag` | float32 (M,) median magnitude, NaN if unknown |
| `scatter` | float32 (M,) residual rms of profile fit / median err |
| `sourceid` | str (M,) |

### `vocab.json`
`{"surveys": [...], "classes": [...], "class_map": {raw: family}, "nbins": 128}`

Survey tokens (survey + band): `OGLE-I`, `VVV-Ks`, `ASAS-SN-V`, `ASAS-SN-g`, `Kepler-Kp`, `TESS-T`.

### `excluded.json`
Base star IDs left out because some entry (any band) is in the val or test split.

## Split rule (no leakage into NN_Periodogram benchmarks)

Group entries by base star ID (sourceid without the `:band` suffix). A star is used
only if every entry is `split == "train"` in the shard metadata. Profiles and
templates both come only from such stars. The generator must never see a val or
test star, otherwise synthetic data would carry their shapes into NN_Periodogram
training.

## Contact binaries / period conventions

Each catalogue's own period convention is kept (OGLE/Kepler/TESS eclipsing binaries:
orbital period). It is recorded per star through `shard`; no convention is imposed.
