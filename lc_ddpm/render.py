"""Turn generated profiles into light curves on real cadences (template stars)."""
import numpy as np


def normalise_profiles(p):
    """Rescale each profile to min 0 / max 1 and roll its faintest point (maximum) to index 0."""
    p = np.asarray(p, dtype=np.float64)
    lo, hi = p.min(1, keepdims=True), p.max(1, keepdims=True)
    p = (p - lo) / np.where(hi > lo, hi - lo, 1.0)
    return np.stack([np.roll(row, -int(np.argmax(row))) for row in p])


def interp_profile(profile, phase):
    """Periodic linear interpolation of a profile sampled at phases (k + 0.5) / nbins."""
    n = len(profile)
    u = np.mod(phase, 1.0) * n - 0.5
    k = np.floor(u).astype(np.int64)
    f = u - k
    return (1 - f) * profile[k % n] + f * profile[(k + 1) % n]


NOISE_MODES = ("residual", "gaussian", None)


def noise_mode(noise):
    """Normalise a ``noise`` argument: "residual" (or True), "gaussian", or None / "none" / False."""
    if noise is True:
        return "residual"
    if noise is None or noise is False or noise == "none":
        return None
    if noise in NOISE_MODES:
        return noise
    raise ValueError(f"unknown noise mode {noise!r}; valid: residual, gaussian, none")


class TemplateBank:
    """Cadence + noise bank from ``templates.npz`` (training-split stars only).

    Bank v2 adds ``resid`` (the star's real residuals, aligned with ``time``); ``resid`` is None
    for a v1 bank, and residual noise then falls back to Gaussian. Bank v2.1 adds ``quality``
    (False = junk residuals); :meth:`pick` then uses only quality templates where a survey has any."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.time, self.err, self.offsets = z["time"], z["err"], z["offsets"]
            self.survey, self.mag, self.scatter = z["survey"], z["mag"], z["scatter"]
            self.sourceid = z["sourceid"]
            self.resid = z["resid"] if "resid" in z.files else None
            self.quality = z["quality"].astype(bool) if "quality" in z.files else None

    def mags_known(self, survey):
        return bool(np.isfinite(self.mag[self.survey == survey]).any())

    def pick(self, survey, mag, rng, k_nearest=10):
        """Template index for a survey: random among the k nearest in median magnitude when
        ``mag`` is finite and the survey has known magnitudes, otherwise uniformly random."""
        idx = np.flatnonzero(self.survey == survey)
        if len(idx) == 0:
            raise ValueError(f"no templates for survey index {survey}")
        if self.quality is not None and self.quality[idx].any():
            idx = idx[self.quality[idx]]
        if mag is not None and np.isfinite(mag):
            known = idx[np.isfinite(self.mag[idx])]
            if len(known):
                idx = known[np.argsort(np.abs(self.mag[known] - mag))[:k_nearest]]
        return int(rng.choice(idx))

    def get(self, j):
        s = slice(self.offsets[j], self.offsets[j + 1])
        scatter = float(self.scatter[j])
        return self.time[s].astype(np.float64), self.err[s].astype(np.float64), (scatter if np.isfinite(scatter) else 1.0)

    def get_resid(self, j):
        """Residual series of template ``j`` (None for a v1 bank)."""
        if self.resid is None:
            return None
        return self.resid[self.offsets[j]:self.offsets[j + 1]].astype(np.float64)


def residual_noise(resid, err, rng):
    """The template's (residual, error) pairs circularly shifted together by a random index, with a
    random sign. Returns (noise, err): every residual keeps its own error, so outlier fractions and
    rms / median(err) of the template are preserved exactly."""
    k = int(rng.integers(len(resid)))
    sign = 1.0 if rng.integers(2) else -1.0
    return sign * np.roll(resid, k), np.roll(err, k)


def render(profiles, period, amplitude, mean_mag, templates, template_idx, rng, times=None, survey=None, cls=None,
           noise="residual"):
    """Assemble light curves.

    mag(t) = mean_mag + amplitude * (P((t - t0) / period mod 1) - mean(P)) + noise

    profiles: (N, nbins), already normalised (see :func:`normalise_profiles`). period, amplitude,
    mean_mag: (N,) arrays. ``mean_mag`` must be resolved by the caller (the API uses the requested
    magnitude, else the template's median magnitude, else 0 = relative magnitudes for surveys
    without absolute magnitudes). template_idx: (N,) indices into ``templates``. times: None
    (template cadence), one array shared by all curves, or a list of N arrays; with user times the
    errors are resampled from the template's error distribution. t0 is uniform in [0, period).

    noise: "residual" adds the template star's real residual series, circularly shifted by a random
    index and randomly sign-flipped (its errors are shifted with it); "gaussian" adds
    N(0, (err_i * scatter)^2) with the template's scatter ratio; None / "none" gives noiseless curves.
    Residual noise needs template cadence and a v2 bank (``resid``); otherwise Gaussian is used.
    Each output dict records the mode actually applied under "noise".
    """
    mode = noise_mode(noise)
    out = []
    for i, prof in enumerate(profiles):
        j = int(template_idx[i])
        t, err, scatter = templates.get(j)
        P, A = float(period[i]), float(amplitude[i])
        t0 = rng.uniform(0, P)
        if times is not None:
            t = np.asarray(times[i] if isinstance(times, (list, tuple)) else times, dtype=np.float64)
            err = rng.choice(err, size=len(t))
        mag = mean_mag[i] + A * (interp_profile(prof, (t - t0) / P) - prof.mean())
        applied = mode
        if mode == "residual" and (times is not None or templates.resid is None):
            applied = "gaussian"
        if applied == "residual":
            dm, err = residual_noise(templates.get_resid(j), err, rng)
            mag = mag + dm
        elif applied == "gaussian":
            mag = mag + rng.normal(0.0, 1.0, len(t)) * err * scatter
        out.append({"time": t, "mag": mag, "err": err, "period": P, "amplitude": A, "mean_mag": float(mean_mag[i]),
                    "survey": survey, "cls": cls, "template_sourceid": str(templates.sourceid[j]), "t0": t0,
                    "scatter": scatter, "noise": applied or "none", "profile": np.asarray(prof, dtype=np.float32)})
    return out
