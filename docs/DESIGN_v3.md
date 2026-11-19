# LC_DDPM v3 design: generating measured light curves (draft, 2026-11-18)

Status: approved direction (Nill, 2026-11-18/19: generate measured light curves; "do what you want").
The choices marked *(choice, revisable)* below were made by the implementer, not by Nill.

## Goal

`generate(survey, period, amplitude, mag=None, cls=None, n=1, seed=None)` returns light curves made of
individual measurements: observation times, magnitudes and per-point errors. Every one of them is generated
by the model. Nothing is copied from a real star, and the model is not reduced to one folded cycle.

This is what the pre-2.0 LC_DDPM did (PRIMVS curves; channels time, mag and err; 2-D U-Net and later
flow matching). v3 keeps that idea and changes four things:
- it is a 1-D sequence model instead of a sequence tiled into a 128x128 image;
- it accepts variable length instead of exactly 70 randomly thinned points;
- it covers all NN_Periodogram surveys instead of VVV only;
- amplitude, magnitude and time span keep their physical units instead of per-star standardisation.

v2 (the phase-profile model with real template stars) stays in the repository as a baseline for comparison.

## What one training example is

One star from the NN_Periodogram shards, training split only. The same rule as v2 applies: a star is used
only if every entry of it, in every band, is in the training split. Each example has:

- **Points.** Up to `N_MAX = 1024` measurements (t_i, m_i, e_i), sorted by time *(choice, revisable)*.
  Window rule *(choice, revisable)*:
  1. n <= N_MAX: the whole light curve.
  2. Otherwise pick a random start and take N_MAX consecutive points. If that run spans at least
     3 periods, use it (real cadence kept exactly; this is the usual case for OGLE, ASAS-SN and
     short-period Kepler/TESS stars).
  3. Otherwise take a random time window spanning min(3 P, full span) and, if it holds more than
     N_MAX points, keep N_MAX of them chosen at random (thinned, but still covering 3 cycles).
  4. Stars whose whole light curve spans less than 1.5 periods are not used.
  Shorter examples are padded and masked. The window span S and the number of points n are
  recorded as conditions, so the model knows how much of the star it is generating.
- **Conditions.** The same as v2: survey token, class family, log10 period (label), log10 amplitude, median
  magnitude (NaN when unknown). Amplitude and class come from bank v2.2. New conditions: log10 time
  span S of the window and log10 number of points n (cadence model only). Every condition can be
  dropped (classifier-free guidance). At generation time, S and n default to a draw from the
  training-split stars of the requested survey.

## Representation (all channels about unit scale)

| channel | definition | back to physical units |
|---|---|---|
| time | t_i - t_0 in days, divided by the window span S, so values lie in [0, 1] | t = S * x (S is a condition) |
| mag | (m_i - median m) / A, with A the amplitude | m = median + A * x |
| err | log10(e_i / A) | e = A * 10^x |

Because mag and err are scaled by the requested amplitude, the amplitude and the magnitude keep their meaning
at generation time.

## Model

Two stages, both diffusion models, trained separately:

1. **Cadence model**, p(times | survey, number of points, span). Learns the survey's sampling pattern
   (gaps, nightly and seasonal structure). It works on the sorted time gaps, which are positive, so it uses
   log gaps.
2. **Measurement model**, p(mag, err | times, conditions). For every point the network sees its time and
   phase features sin and cos(2 pi k t / P), k = 1..4, computed exactly from the given times and the
   requested period. With these features, periodicity is easy to learn. Without them a network would have
   to compute t mod P from noisy inputs, which it does poorly. Each point's value is still generated
   separately, so noise, outliers, correlated errors and changes from cycle to cycle come from the data.

Architecture for both: a transformer over the points (about 8 layers, width 256, PyTorch scaled_dot_product_attention so 1024 points fit a 6 GB GPU, padding mask, adaLN
conditioning on the diffusion step and the star conditions). The points are irregularly spaced and their
number varies, so a transformer fits better than a convolutional U-Net. Training uses v-prediction with a
cosine schedule; sampling uses DDIM. This is the same training setup as v2.

Splitting the work into two stages lets a user pass their own observation times (stage 2 only). It also lets
each stage be evaluated on its own.

## What changes for users

- `generate(...)` returns time, mag and err for each point. All three are generated.
- `times=` is optional. When it is given, only stage 2 runs.
- The period and amplitude are no longer exact by construction. The model is asked for them; the evaluation
  measures how often they come out (below). For NN_Periodogram training data, the label is the requested
  period.
- Length is at most 1024 points per call. Longer curves, for example a full Kepler quarter, are out of scope
  for v3.0.

## Evaluation (mostly reused from v2)

- **Real-versus-synthetic classifier.** Real validation stars against generated curves with the same
  conditions, scored on profile, noise statistics and cadence features.
- **Requested-period recovery.** Lomb-Scargle on generated curves: the fraction within 1% of the requested
  period, with aliases accepted and not, next to the same numbers for real validation stars.
- **Requested-amplitude recovery.** The ratio of the folded amplitude to the requested amplitude.
- **Cadence realism.** Distributions of gaps, number of points and span against real stars of the survey.
- **Comparison with v2** on all of the above.
- **NN_Periodogram retrain,** scored on real validation stars only (the decisive test, as before).

## Data and compute

- Data: the NN_Periodogram shards and bank v2.2 labels on Aurora, plus the 34 GB copy. **No work on Aurora
  until Nill lifts the hold.**
- Compute: the Argus Slurm cluster as user njm (7 nodes, one RTX 3050 6 GB each). Data copied from the lab
  NAS (`/Lab_NAS/scratch/nillmill/{nnp,lcddpm}`) to `/scratch/njm/lcddpm/data`. Aurora is not used.
- Code is written and unit-tested on the Mac with the fake shards (`~/LC_DDPM_work/fakeshards`).

## Open decisions for Nill

Resolved by the implementer (revisable):
1. Cadence is learned (stage 1). `cadence="real"` (sample a training-split star's times) is kept as an
   option and as the baseline for evaluating stage 1.
2. N_MAX = 1024 with the window rule above (the 512 contiguous points of the draft would hold less than
   one cycle of many Kepler/TESS binaries).
3. Each catalogue's own period convention is kept (no conversion); the class token tells the model which
   convention applies. Revisit if NN_Periodogram results show ELL harmonic confusion.

## Use as NN_Periodogram training data

v3's period is not exact. Before any synthetic shard is written, the period-recovery evaluation (real vs
synthetic, per survey and class) is run; whole surveys or classes whose synthetic recovery falls clearly
below that of real stars are left out. Individual curves are never filtered by whether Lomb-Scargle
recovers them (that would keep only easy cases). Conditions for the synthetic shard are drawn only from
bank v2.2 rows (training-split stars).

## Implementation record (2026-11-19)

Facts of the implementation; the decisions above are unchanged.

**Code.** `lc_ddpm/data_v3.py` (window rule, representation, per-point inputs, training-file loader, batches),
`lc_ddpm/model.py` (`PointTransformer`), `lc_ddpm/diffusion.py` (`SeqDiffusion`, `ddim_step` shared with v2),
`lc_ddpm/api_v3.py` (`GeneratorV3`), `training/build_v3.py`, `training/train_v3.py`, `training/evaluate_v3.py`,
`tests/test_v3.py`.

**Window rule details.** Rule 4 is checked first on the whole (finite, time-sorted) curve. Rule 2 uses one
random start, no retries. Rule 3's window of width W = min(3 P, span) starts at a randomly chosen observation
time t_j with t_j - t_0 <= span - W, so it always holds at least one point. Examples with fewer than
`MIN_POINTS = 10` points after the rule are dropped. Windows are drawn once at build time, seeded per star
(`build_bank.seed_of(shard, index, salt=3)`); evaluation applies the same function and seed to real stars.
S is the span of the selected points (last minus first time), not W. Light curves are read with
`build_bank.load_star(..., subsample=False)` (no Kepler/TESS thinning), then sorted by time.

**Training file** (`build_v3.py --out V3_DIR`):

| file / key | dtype | meaning |
|---|---|---|
| `train_v3.npz` `time` | float64 (P,) | concatenated times, days from each window's first point |
| `obs_mag`, `obs_err` | float32 (P,) | measured magnitudes and errors |
| `offsets` | int64 (N+1,) | example i is `[offsets[i], offsets[i+1])` |
| `n_max` | int32 scalar | N_MAX used |
| `survey`, `cls`, `period`, `amplitude`, `mag`, `tier`, `snr`, `repeats`, `shard`, `index`, `sourceid` | as bank | copied from the bank v2.2 row |
| `bank_row` | int32 (N,) | row in the bank's profiles.npz |
| `rule` | int8 (N,) | window rule used (1, 2, 3) |
| `n_points`, `n_total` | int32 (N,) | points in the window / in the cleaned star |
| `span` | float64 (N,) | S, days |
| `cadences.npz` | | up to 2000 examples per survey: `time`, `offsets`, `survey`, `period`, `mag`, `span`, `n_points`, `sourceid` (cadence="real") |
| `vocab.json` | | copy of the bank's |
| `summary.json` | | counts per survey and rule, dropped stars by reason (`not_train`, `not_in_shards`, `short_span`, `too_few_points`) |

Guard: a bank row is also dropped if its shard entry is not `split == "train"` or its base ID is in the bank's
`excluded.json`.

**Model inputs.** Targets are standardised per channel with training-set mean/std (`norm` in the checkpoint):
measure = [(m - median m)/A, log10(e/A)], cadence = log10(dt_i / S) floored at 1e-9. Per-point inputs:
measure = [x = t/S, sin/cos(pi 2^j x) j < 10, standardised log gaps to previous and next point, sin/cos(2 pi k t/P)
k = 1..4] (31 columns; phase computed in float64 numpy); cadence = [sin/cos(pi 2^j i/N_MAX) j <= 10, i/(n-1)]
for gap index i (23 columns). Numeric conditions: measure (log10 P, log10 A, mag, log10 S), with survey and class;
cadence (log10 n, log10 S), with survey only. The phase features are zeroed whenever the period condition is
dropped, so the unconditional branch of guidance has no period information. Generated gaps are rescaled to
sum to S (span exact). No clamp on predicted x0 for v3.

**Network.** `PointTransformer`: linear embedding of [noisy channels, per-point inputs], 8 blocks of width 256,
8 heads, MLP ratio 4, adaLN-Zero conditioning on (diffusion step + condition embedding), key-padding mask in
`scaled_dot_product_attention`, zero output at padded points; about 10.0 M parameters per stage.
Batches are padded to the longest example rounded up to a multiple of 8.

**Run directory.** `RUN_DIR/lc_ddpm_cadence.pt`, `RUN_DIR/lc_ddpm_measure.pt` (keys `format = "lc_ddpm-3"`,
`stage`, `config`, `state_dict` (EMA), `vocab`, `norm`, `n_max`, `T`, `card`, and for measure `defaults`:
survey, period, n_points, span, mag of up to 20000 training examples per survey), `RUN_DIR/cadences.npz`
(symlink to the training file's), `RUN_DIR/<stage>/{log.jsonl, best.pt, ckpt_*.pt, samples_*.png}`.
