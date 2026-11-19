"""High-level API: ``Generator`` and ``generate``."""
import os
from pathlib import Path

import numpy as np
import torch

from .api_v3 import CADENCE_NAME, CADENCES_NAME, MEASURE_NAME, GeneratorV3  # noqa: F401 (re-exported)
from .data import encode_conditions
from .diffusion import Diffusion
from .model import UNet1D
from .render import TemplateBank, noise_mode, normalise_profiles, render

CHECKPOINT_NAME, TEMPLATES_NAME = "lc_ddpm.pt", "templates.npz"
# Placeholder: download location of the released weights + templates. Not published yet.
WEIGHTS_URL = None


def default_home():
    return Path(os.environ.get("LC_DDPM_HOME", Path.home() / ".cache" / "lc_ddpm"))


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_v3(path):
    return torch.load(path, map_location="cpu", weights_only=False).get("format") == "lc_ddpm-3"


def build_model(ckpt):
    model = UNet1D(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _lookup(name, names, kind):
    if name is None:
        return -1
    if name not in names:
        raise ValueError(f"unknown {kind} {name!r}; valid {kind} names: {', '.join(names)}")
    return names.index(name)


class Generator:
    """A trained shape model plus its template bank.

    checkpoint: path to ``lc_ddpm.pt`` (model config, weights, vocab, normalisation constants).
    templates: path to ``templates.npz``; default is the file next to the checkpoint.
    """

    def __init__(self, checkpoint, templates=None, device=None):
        checkpoint = Path(checkpoint)
        self.device = device or auto_device()
        self.ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = build_model(self.ckpt).to(self.device)
        self.vocab, self.norm = self.ckpt["vocab"], self.ckpt["norm"]
        self.diffusion = Diffusion(self.ckpt.get("T", 1000))
        self.templates = TemplateBank(templates or checkpoint.with_name(TEMPLATES_NAME))

    @classmethod
    def from_pretrained(cls, path_or_dir=None, templates=None, device=None):
        """Load from a checkpoint file or a directory. Default directory: $LC_DDPM_HOME, else ~/.cache/lc_ddpm/.

        A directory holding ``lc_ddpm_measure.pt`` (v3) gives a :class:`~lc_ddpm.api_v3.GeneratorV3`;
        otherwise ``lc_ddpm.pt`` + ``templates.npz`` give the v2 Generator. A checkpoint file is
        dispatched by its ``format`` key ("lc_ddpm-3"; absent or "lc_ddpm-2" = v2). For a v3 measure
        checkpoint file, ``lc_ddpm_cadence.pt`` and ``cadences.npz`` are looked up next to it."""
        path = Path(path_or_dir) if path_or_dir is not None else default_home()
        if path.is_dir() and (path / MEASURE_NAME).exists():
            return GeneratorV3.from_dir(path, device)
        if path.is_dir():
            path = path / CHECKPOINT_NAME
        if path.is_file() and is_v3(path):
            return GeneratorV3(path, path.with_name(CADENCE_NAME), path.with_name(CADENCES_NAME), device)
        if not path.exists():
            hint = f"download from {WEIGHTS_URL}" if WEIGHTS_URL else "no public download exists yet; train one with training/train.py"
            raise FileNotFoundError(f"no LC_DDPM weights at {path} (set LC_DDPM_HOME or pass a path; {hint})")
        return cls(path, templates, device)

    @property
    def surveys(self):
        return list(self.vocab["surveys"])

    @property
    def classes(self):
        return list(self.vocab["classes"])

    def sample_profiles(self, survey, cls, period, amplitude, mag, guidance=1.0, steps=50, seed=None, batch=1024):
        """Normalised profiles (N, nbins) for per-curve condition arrays (survey/cls int, -1 = missing;
        period/amplitude/mag float, NaN = missing)."""
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        out = []
        for s in range(0, len(period), batch):
            sl = slice(s, s + batch)
            cond = encode_conditions(survey[sl], cls[sl], period[sl], amplitude[sl], mag[sl], self.norm, self.device)
            x = self.diffusion.sample(self.model, cond, steps=steps, guidance=guidance, generator=gen,
                                      nbins=self.ckpt["config"]["nbins"])
            out.append(((x.cpu().numpy() + 1) / 2))
        return normalise_profiles(np.concatenate(out))

    def generate(self, survey, period, amplitude, mag=None, cls=None, n=1, seed=None, times=None, guidance=1.0,
                 steps=50, noise="residual"):
        """Generate ``n`` light curves of one survey.

        period (days), amplitude (peak-to-peak mag), mag (mean magnitude): scalars or arrays, broadcast
        to n (n grows to the array length when an array is longer).
        mag=None: the template star's median magnitude is used as mean and as the model condition;
        for surveys without absolute magnitudes (ASAS-SN, Kepler, TESS) the mean is 0, i.e. the
        curves are in relative magnitudes. cls=None leaves the class unconditioned.
        times: None (template cadence), one array for all curves, or a list of n arrays.
        noise: "residual" (default; the template star's real residuals, randomly shifted and sign-flipped),
        "gaussian" (N(0, (err * scatter)^2)) or None / "none" (noiseless). Residual noise falls back to
        Gaussian for a v1 template bank (no ``resid``) and for user ``times``.
        Returns a list of dicts {time, mag, err, period, amplitude, mean_mag, survey, cls,
        template_sourceid, t0, scatter, noise (mode applied), profile}.
        """
        s_idx = _lookup(survey, self.surveys, "survey")
        if s_idx < 0:
            raise ValueError(f"survey is required; valid survey names: {', '.join(self.surveys)}")
        c_idx = _lookup(cls, self.classes, "class")
        noise_mode(noise)  # fail before sampling on a bad value
        rng = np.random.default_rng(seed)
        n = max(n, *(np.size(v) for v in (period, amplitude, mag) if v is not None))
        period, amplitude = (np.broadcast_to(np.asarray(v, np.float64), (n,)).copy() for v in (period, amplitude))
        req = np.broadcast_to(np.asarray(np.nan if mag is None else mag, np.float64), (n,))
        tidx = np.array([self.templates.pick(s_idx, m, rng) for m in req])
        tmag = self.templates.mag[tidx].astype(np.float64)
        cond_mag = np.where(np.isfinite(req), req, tmag)
        mean_mag = np.nan_to_num(cond_mag, nan=0.0)
        prof = self.sample_profiles(np.full(n, s_idx), np.full(n, c_idx), period, amplitude, cond_mag, guidance,
                                    steps, seed=rng.integers(2 ** 62))
        return render(prof, period, amplitude, mean_mag, self.templates, tidx, rng, times=times, survey=survey,
                      cls=cls, noise=noise)

    def card(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"surveys": self.surveys, "classes": self.classes, "config": self.ckpt["config"], "params": n_params,
                "norm": self.norm, "card": self.ckpt.get("card", {}), "templates": len(self.templates.survey)}


_DEFAULT = {}


def generate(survey, period, amplitude, mag=None, cls=None, n=1, seed=None, times=None, guidance=1.0, steps=50,
             model=None, noise="residual", **v3):
    """Module-level shortcut for :meth:`Generator.generate` (v2) or :meth:`GeneratorV3.generate` (v3).
    ``model`` may be a Generator, a GeneratorV3, a checkpoint path or directory, or None (cached default
    from $LC_DDPM_HOME or ~/.cache/lc_ddpm). ``noise`` applies to v2 only; v3 keywords (n_points, span,
    cadence, batch) to v3 only."""
    if not isinstance(model, (Generator, GeneratorV3)):
        key = str(model)
        if key not in _DEFAULT:
            _DEFAULT[key] = Generator.from_pretrained(model)
        model = _DEFAULT[key]
    kw = v3 if isinstance(model, GeneratorV3) else {"noise": noise, **v3}
    return model.generate(survey, period, amplitude, mag=mag, cls=cls, n=n, seed=seed, times=times,
                          guidance=guidance, steps=steps, **kw)
