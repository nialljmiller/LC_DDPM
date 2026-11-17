# LC_DDPM

Conditional denoising diffusion models that generate realistic variable-star light curves.

Ask for a star by survey, period, amplitude and (optionally) mean magnitude and class, and get back
a light curve on a real survey cadence with realistic errors:

```python
from lc_ddpm import generate

curves = generate(survey="VVV-Ks", period=0.55, amplitude=0.4, mag=15.0, cls="RR", n=10, seed=1)
c = curves[0]
c["time"], c["mag"], c["err"]        # observation times (days), magnitudes, errors
c["period"], c["amplitude"]          # exactly what was requested
```

```bash
lc-ddpm generate --survey VVV-Ks --period 0.55 --amplitude 0.4 --mag 15 --cls RR -n 10 --out rr.csv --plot rr.png
lc-ddpm info          # surveys, classes and the model card
```

> **Status (November 2026):** v2.0 is a complete rewrite. Trained weights are not yet published;
> until they are, train a model with `training/train.py` (below) and point `--model` /
> `Generator.from_pretrained()` at it, or set `LC_DDPM_HOME`.

## How it works

1. **Shape (the diffusion model).** A 1-D U-Net with circular padding generates the phase-folded
   light-curve shape: 128 phase points in units of the amplitude, faintest point at phase 0.
   It is a v-prediction DDPM (cosine schedule) sampled with DDIM and classifier-free guidance.
   Every condition (survey, class, log period, log amplitude, magnitude) is dropped at random
   during training, so any subset of them can be given at generation time.
2. **Cadence and noise (real template stars).** The shape is placed on the observation times and
   per-point errors of a real star from the requested survey, chosen among the stars closest in
   magnitude. By default (`noise="residual"`) the noise is that star's own residual series (its
   light curve minus its fitted profile), circularly shifted by a random number of points and
   randomly sign-flipped, so real outliers, correlated noise and systematics are kept.
   `noise="gaussian"` draws N(0, (err * scatter)^2) from the star's errors and robust scatter
   ratio, and `noise=None` (CLI `--noise none`) gives noiseless curves. Template banks without
   residuals (bank v1) use Gaussian noise.
3. **Assembly.** `mag(t) = mean_mag + amplitude * (P(((t - t0) / period) mod 1) - mean(P)) + noise`.
   Period and amplitude are therefore exact by construction, which makes the output usable as
   labelled training data for period-finding methods.

You can also pass your own observation times (`times=`), in which case errors are drawn from the
template star's error distribution and the noise is Gaussian (residuals cannot be aligned with other
times). Each returned curve records the noise mode applied under `"noise"`.

## Surveys and training data

The model is trained on phase-folded profiles of catalogued periodic variables from the training
split of the [NN_Periodogram](https://github.com/nialljmiller/NN_Periodogram) data set:

| Survey token | Source | Band | Absolute magnitudes |
|---|---|---|---|
| `OGLE-I` | OGLE-IV Bulge, disk, LMC and SMC variables | I | yes |
| `VVV-Ks` | VVV / PRIMVS variables | Ks | yes |
| `ASAS-SN-V`, `ASAS-SN-g` | ASAS-SN light curves of VSX variables | V, g | no (relative) |
| `Kepler-Kp` | Kepler eclipsing binaries | Kp | no (relative) |
| `TESS-T` | TESS eclipsing binaries | T | no (relative) |

For surveys without absolute magnitudes the generated curves are in relative magnitudes (mean 0).
Periods follow each catalogue's own convention (eclipsing binaries: orbital period).
Only stars whose every entry is in the NN_Periodogram training split are used, so synthetic curves
never carry information about NN_Periodogram's validation or test stars.

## Installation

```bash
pip install -e .            # core: numpy + torch
pip install -e ".[plot]"    # adds matplotlib for --plot
```

## Training your own model

```bash
# 1. Build the profile and template bank from NN_Periodogram shards
python training/build_bank.py --help

# 2. Train (single GPU; bf16 on CUDA). Useful options: --min-snr 5 (drop low-S/N profiles),
#    --max-repeats 2 (drop labels that look like multiples of the true period), --init CKPT (fine-tune)
python training/train.py --bank BANK_DIR --out RUN_DIR --min-snr 5

# 3. Evaluate: sample grids (grid), period/amplitude recovery with residual and Gaussian noise
#    (period), real-vs-synthetic classifier in shape and full-pipeline modes with profile and
#    summary-statistic features (classifier), guidance / DDIM-step sweep (sweep)
python training/evaluate.py --help
```

`RUN_DIR/lc_ddpm.pt` together with `templates.npz` is everything `Generator.from_pretrained(RUN_DIR)`
needs. The design and file formats are documented in [docs/DESIGN.md](docs/DESIGN.md).

## History

Versions before 2.0 (2025 to early 2026) tiled light curves into 2-D images and trained a 2-D
U-Net (DDPM, later stable flow matching) on PRIMVS light curves. That code is in the git history
(last commit `263dad6`).

## Citation and licence

See [CITATION.cff](CITATION.cff). MIT licence, copyright Niall Miller.
