"""v3 generator (docs/DESIGN_v3.md): measured light curves from two diffusion models.

Stage 1 (cadence model) draws observation times for (survey, n, span); stage 2 (measurement model)
draws magnitudes and errors at those times for (survey, class, period, amplitude, magnitude, span).
A run directory holds ``lc_ddpm_measure.pt``, ``lc_ddpm_cadence.pt`` and optionally ``cadences.npz``.
"""
from pathlib import Path

import numpy as np
import torch

from .data_v3 import CHANNELS, encode_v3, measure_values, pad_stack, point_inputs, raw_numeric, times_from_log_gaps
from .diffusion import SeqDiffusion
from .model import PointTransformer

MEASURE_NAME, CADENCE_NAME, CADENCES_NAME = "lc_ddpm_measure.pt", "lc_ddpm_cadence.pt", "cadences.npz"
K_NEAREST = 50  # default n / span / mag: one of the K training examples of the survey nearest in log period


def _load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PointTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return ckpt, model.eval()


def _lookup(name, names, kind):
    if name is None:
        return -1
    if name not in names:
        raise ValueError(f"unknown {kind} {name!r}; valid {kind} names: {', '.join(names)}")
    return names.index(name)


class GeneratorV3:
    """measure: ``lc_ddpm_measure.pt`` (required); cadence: ``lc_ddpm_cadence.pt`` (needed unless times
    are given or cadence="real"); cadences: ``cadences.npz`` (needed for cadence="real")."""

    def __init__(self, measure, cadence=None, cadences=None, device=None):
        from .api import auto_device
        self.device = device or auto_device()
        self.mckpt, self.measure = _load(measure)
        self.measure.to(self.device)
        self.cckpt, self.cadence = _load(cadence) if cadence and Path(cadence).exists() else (None, None)
        if self.cadence is not None:
            self.cadence.to(self.device)
        self.cadences = dict(np.load(cadences)) if cadences and Path(cadences).exists() else None
        self.vocab, self.n_max = self.mckpt["vocab"], int(self.mckpt["n_max"])
        self.defaults = self.mckpt.get("defaults")
        self.diffusion = SeqDiffusion(self.mckpt.get("T", 1000))

    @classmethod
    def from_dir(cls, run_dir, device=None):
        d = Path(run_dir)
        return cls(d / MEASURE_NAME, d / CADENCE_NAME, d / CADENCES_NAME, device)

    @property
    def surveys(self):
        return list(self.vocab["surveys"])

    @property
    def classes(self):
        return list(self.vocab["classes"])

    def _sample(self, stage, survey, cls, num, times, period, guidance, steps, gen, batch):
        """Stage samples, de-standardised, one array per curve (curves batched by length)."""
        ckpt, model = (self.mckpt, self.measure) if stage == "measure" else (self.cckpt, self.cadence)
        norm = ckpt["norm"]
        out = [None] * len(times)
        order = np.argsort([len(t) for t in times], kind="stable")
        for s in range(0, len(order), batch):
            j = order[s:s + batch]
            feat, pad = pad_stack(point_inputs(stage, [times[i] for i in j], period[j], norm, self.n_max),
                                  ckpt["config"]["feat_dim"])
            cond = encode_v3(survey[j], cls[j], num[j], norm, self.device)
            cond.update(feat=torch.as_tensor(feat, device=self.device), pad=torch.as_tensor(pad, device=self.device))
            x = self.diffusion.sample(model, cond, CHANNELS[stage], steps=steps, guidance=guidance, generator=gen)
            x = x.cpu().numpy().astype(np.float64)
            for k, i in enumerate(j):
                n = int(pad[k].sum())
                if stage == "measure":
                    out[i] = x[k, :n] * np.asarray(norm["chan_std"]) + np.asarray(norm["chan_mean"])
                else:
                    out[i] = x[k, :n, 0] * norm["gap_std"] + norm["gap_mean"]
        return out

    def _reference(self, s_idx, period, rng):
        """Index into the defaults table: one of the K_NEAREST examples of the survey nearest in log period."""
        d = self.defaults
        if d is None:
            raise ValueError("this checkpoint has no defaults table; pass n_points and span (or times)")
        pool = np.flatnonzero(d["survey"] == s_idx)
        if len(pool) == 0:
            raise ValueError(f"no training examples of survey {self.surveys[s_idx]}; pass n_points and span")
        near = pool[np.argsort(np.abs(np.log10(d["period"][pool]) - np.log10(period)))[:K_NEAREST]]
        return int(rng.choice(near))

    def generate(self, survey, period, amplitude, mag=None, cls=None, n=1, seed=None, times=None, n_points=None,
                 span=None, cadence="model", guidance=1.0, steps=50, batch=64):
        """Generate ``n`` measured light curves of one survey (time, mag and err all generated).

        period (days), amplitude (peak-to-peak mag), mag (median magnitude), n_points, span (days):
        scalars or arrays broadcast to n. mag=None: the magnitude of the reference training example
        (below) is used for surveys with absolute magnitudes; for the others (ASAS-SN, Kepler, TESS) the
        curves are relative (median 0) and the magnitude condition is left out. cls=None leaves the
        class unconditioned.
        Times: ``times`` (one array for all curves or a list of n arrays, at most n_max points each) runs
        stage 2 only. Otherwise n_points and span default to those of a reference training example
        of the survey (drawn among the K_NEAREST nearest in log period), and cadence="model" samples
        the times from the cadence model, cadence="real" takes the reference example's own times
        (needs cadences.npz; n_points/span must then be None).
        Returns a list of dicts {time (days from the first point), mag, err, period, amplitude,
        mean_mag, survey, cls, n_points, span, cadence ("model", "real" or "user"), t0 (0)}.
        """
        s_idx = _lookup(survey, self.surveys, "survey")
        if s_idx < 0:
            raise ValueError(f"survey is required; valid survey names: {', '.join(self.surveys)}")
        c_idx = _lookup(cls, self.classes, "class")
        if cadence not in ("model", "real"):
            raise ValueError(f"cadence must be 'model' or 'real', not {cadence!r}")
        rng = np.random.default_rng(seed)
        per_curve = times is not None and np.ndim(times[0]) > 0
        n = max(n, len(times) if per_curve else 1, *(np.size(v) for v in (period, amplitude, mag, n_points, span)
                                                     if v is not None))
        bc = lambda v: np.broadcast_to(np.asarray(np.nan if v is None else v, np.float64), (n,)).copy()
        period, amplitude, req_mag, n_req, s_req = map(bc, (period, amplitude, mag, n_points, span))

        if times is not None:
            tl = [np.sort(np.asarray(t, np.float64)) for t in (times if per_curve else [times] * n)]
            if len(tl) != n:
                raise ValueError(f"times: expected one array or {n} arrays, got {len(tl)}")
            for t in tl:
                if not 2 <= len(t) <= self.n_max or t[-1] <= t[0]:
                    raise ValueError(f"times: each curve needs 2..{self.n_max} points spanning a positive time")
            tl = [t - t[0] for t in tl]
            mode = "user"
        ref = [self._reference(s_idx, p, rng) if self.defaults is not None else None for p in period]
        known = self.defaults is not None and np.isfinite(self.defaults["mag"][self.defaults["survey"] == s_idx]).any()
        if known:
            cond_mag = np.where(np.isfinite(req_mag), req_mag, [self.defaults["mag"][r] for r in ref])
        else:
            cond_mag = req_mag
        mean_mag = np.nan_to_num(cond_mag, nan=0.0)

        if times is None and cadence == "real":
            if self.cadences is None:
                raise ValueError("cadence='real' needs cadences.npz next to the checkpoints")
            if np.isfinite(n_req).any() or np.isfinite(s_req).any():
                raise ValueError("cadence='real' takes n_points and span from the real example; leave them None")
            c, off = self.cadences, self.cadences["offsets"]
            pool = np.flatnonzero(c["survey"] == s_idx)
            tl = []
            for p in period:
                near = pool[np.argsort(np.abs(np.log10(c["period"][pool]) - np.log10(p)))[:K_NEAREST]]
                j = int(rng.choice(near))
                tl.append(np.asarray(c["time"][off[j]:off[j + 1]], np.float64))
            mode = "real"
        elif times is None:
            if self.cadence is None:
                raise ValueError("no cadence model (lc_ddpm_cadence.pt); pass times or use cadence='real'")
            d = self.defaults
            n_pts = np.where(np.isfinite(n_req), n_req, [d["n_points"][r] if r is not None else np.nan for r in ref])
            spans = np.where(np.isfinite(s_req), s_req, [d["span"][r] if r is not None else np.nan for r in ref])
            n_pts = np.round(n_pts).astype(int)
            if not ((n_pts >= 2) & (n_pts <= self.n_max)).all() or not (spans > 0).all():
                raise ValueError(f"n_points must be 2..{self.n_max} and span > 0")
            num = raw_numeric("cadence", n=n_pts, span=spans)
            gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
            lg = self._sample("cadence", np.full(n, s_idx), np.full(n, -1), num, [np.zeros(k) for k in n_pts],
                              period, guidance, steps, gen, batch)
            tl = [times_from_log_gaps(g, S) for g, S in zip(lg, spans)]
            mode = "model"

        spans = np.array([t[-1] for t in tl])
        num = raw_numeric("measure", period, amplitude, cond_mag, spans)
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
        x = self._sample("measure", np.full(n, s_idx), np.full(n, c_idx), num, tl, period, guidance, steps, gen, batch)
        out = []
        for i in range(n):
            m, e = measure_values(x[i], mean_mag[i], amplitude[i])
            out.append({"time": tl[i], "mag": m, "err": e, "period": float(period[i]), "amplitude": float(amplitude[i]),
                        "mean_mag": float(mean_mag[i]), "survey": survey, "cls": cls, "n_points": len(tl[i]),
                        "span": float(spans[i]), "cadence": mode, "t0": 0.0})
        return out

    def card(self):
        params = {s: sum(p.numel() for p in m.parameters()) for s, m in (("measure", self.measure), ("cadence", self.cadence))
                  if m is not None}
        return {"format": "lc_ddpm-3", "surveys": self.surveys, "classes": self.classes, "n_max": self.n_max,
                "params": params, "config": {"measure": self.mckpt["config"], "cadence": (self.cckpt or {}).get("config")},
                "card": {"measure": self.mckpt.get("card", {}), "cadence": (self.cckpt or {}).get("card", {})},
                "cadences": None if self.cadences is None else int(len(self.cadences["survey"]))}
