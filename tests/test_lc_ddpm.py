import json

import numpy as np
import pytest
import torch

from lc_ddpm import Generator, generate
from lc_ddpm.cli import main as cli_main
from lc_ddpm.data import ProfileDataset, compute_norm, encode_conditions, load_bank, select_rows
from lc_ddpm.diffusion import Diffusion, drop_conditions
from lc_ddpm.model import UNet1D
from lc_ddpm.render import TemplateBank, normalise_profiles, render

SURVEYS = ["OGLE-I", "VVV-Ks", "ASAS-SN-V", "ASAS-SN-g", "Kepler-Kp", "TESS-T"]
CLASSES = ["RR", "EB", "SIN"]
MAG_KNOWN = {0: (13, 19), 1: (11, 17)}  # survey index -> magnitude range; others NaN


def fake_shapes(cls, n, rng):
    ph = (np.arange(128) + 0.5) / 128
    if cls == "RR":    # sawtooth: fast rise to max light, slow decline
        x = np.mod(ph[None] + rng.uniform(0, 1, (n, 1)), 1.0)
        rise = rng.uniform(0.1, 0.25, (n, 1))
        p = np.where(x < rise, 1 - x / rise, (x - rise) / (1 - rise))
    elif cls == "EB":  # two Gaussian dips (magnitudes: dips are maxima)
        w = rng.uniform(0.02, 0.06, (n, 1))
        d2 = rng.uniform(0.2, 0.9, (n, 1))
        p = np.exp(-0.5 * (np.minimum(ph, 1 - ph)[None] / w) ** 2)
        p += d2 * np.exp(-0.5 * ((ph[None] - 0.5) / w) ** 2)
    else:              # sine with a weak harmonic
        p = np.sin(2 * np.pi * ph)[None] + rng.uniform(0, 0.3, (n, 1)) * np.sin(4 * np.pi * ph)[None]
    return normalise_profiles(p + rng.normal(0, 0.01, p.shape))


def fake_resid(t, err, scatter, rng, outlier_frac=0.03):
    """Bank-v2-like residuals: white noise at err * scatter, a slow trend and a few large outliers."""
    r = rng.normal(0, 1, len(t)) * err * scatter + 0.02 * np.sin(2 * np.pi * t / rng.uniform(300, 900))
    out = rng.uniform(size=len(t)) < outlier_frac
    return r + out * rng.choice([-1, 1], len(t)) * rng.uniform(6, 15, len(t)) * err


def make_fake_bank(path, n=300, m=40, seed=0, v2=True):
    """Tiny bank in the docs/DESIGN.md format: profiles.npz, templates.npz, vocab.json.
    v2 adds ``resid``/``scatter_rms`` to templates and ``repeats`` to profiles."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    cls = rng.integers(0, len(CLASSES), n)
    survey = rng.integers(0, len(SURVEYS), n)
    prof = np.zeros((n, 128))
    for c in range(len(CLASSES)):
        prof[cls == c] = fake_shapes(CLASSES[c], int((cls == c).sum()), rng)
    logp = np.where(cls == 0, rng.uniform(-0.5, -0.1, n), np.where(cls == 1, rng.uniform(-0.5, 1, n), rng.uniform(-1, 0.5, n)))
    mag = np.full(n, np.nan)
    for s, (lo, hi) in MAG_KNOWN.items():
        mag[survey == s] = rng.uniform(lo, hi, (survey == s).sum())
    np.savez(path / "profiles.npz", profile=prof.astype(np.float32), period=10 ** logp,
             amplitude=(10 ** rng.uniform(-1.3, 0, n)).astype(np.float32), mag=mag.astype(np.float32),
             survey=survey.astype(np.int16), cls=cls.astype(np.int16), tier=(rng.uniform(size=n) < 0.3).astype(np.int8),
             snr=rng.uniform(5, 200, n).astype(np.float32), chi2r=np.ones(n, np.float32),
             order=np.full(n, 4, np.int16), n_points=np.full(n, 100, np.int32), shard=np.array(["fake"] * n),
             index=np.arange(n, dtype=np.int32), sourceid=np.array([f"s{i}" for i in range(n)]),
             **({"repeats": rng.choice([1, 2, 3, 4], n, p=[0.8, 0.1, 0.06, 0.04]).astype(np.int8)} if v2 else {}))
    ts, es, rs, off = [], [], [], [0]
    scatter = rng.uniform(0.8, 1.5, m)
    tsurv = np.arange(m) % len(SURVEYS)
    tmag = np.full(m, np.nan)
    for j in range(m):
        k = int(rng.integers(30, 200))
        ts.append(np.sort(rng.uniform(0, 1000, k))); es.append(rng.uniform(0.01, 0.05, k)); off.append(off[-1] + k)
        rs.append(fake_resid(ts[-1], es[-1], scatter[j], rng))
        if tsurv[j] in MAG_KNOWN:
            tmag[j] = rng.uniform(*MAG_KNOWN[tsurv[j]])
    v2_fields = {}
    if v2:
        mad = [1.4826 * np.median(np.abs(r - np.median(r))) / np.median(e) for r, e in zip(rs, es)]
        v2_fields = {"resid": np.concatenate(rs).astype(np.float32), "scatter_rms": scatter.astype(np.float32)}
        scatter = np.array(mad)
    np.savez(path / "templates.npz", time=np.concatenate(ts), err=np.concatenate(es).astype(np.float32),
             offsets=np.array(off, np.int64), survey=tsurv.astype(np.int16), mag=tmag.astype(np.float32),
             scatter=scatter.astype(np.float32), sourceid=np.array([f"t{j}" for j in range(m)]), **v2_fields)
    (path / "vocab.json").write_text(json.dumps({"surveys": SURVEYS, "classes": CLASSES, "class_map": {}, "nbins": 128}))
    return path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Fake bank plus a randomly initialised tiny model saved in the Generator checkpoint format."""
    d = make_fake_bank(tmp_path_factory.mktemp("bank"))
    arrays, vocab = load_bank(d)
    torch.manual_seed(0)
    model = UNet1D(len(SURVEYS), len(CLASSES), width=16, mults=(1, 2, 2, 2), n_res=1)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "vocab": vocab, "norm": compute_norm(arrays),
                "T": 1000}, d / "lc_ddpm.pt")
    return d


def cond_batch(B, mask_cols=(), nan_mag=False):
    c = encode_conditions(np.arange(B) % 6, np.arange(B) % 3, np.full(B, 0.5), np.full(B, 0.3),
                          np.full(B, np.nan if nan_mag else 15.0), {"mean": [0, 0, 15], "std": [1, 1, 2]})
    c["mask"][:, list(mask_cols)] = False
    return c


@pytest.mark.parametrize("B,width,shift", [(2, 16, 8), (5, 32, 16), (3, 16, 24)])
def test_model_shapes_and_circular_equivariance(B, width, shift):
    torch.manual_seed(0)
    m = UNet1D(6, 3, width=width).eval()
    torch.nn.init.normal_(m.out[-1].weight)  # zero-initialised by default
    x, t, c = torch.randn(B, 128), torch.randint(0, 1000, (B,)), cond_batch(B)
    with torch.no_grad():
        y = m(x, t, c["survey"], c["cls"], c["num"], c["mask"])
        y_roll = m(torch.roll(x, shift, 1), t, c["survey"], c["cls"], c["num"], c["mask"])
    assert y.shape == (B, 128) and m(x[:, None], t, c["survey"], c["cls"], c["num"], c["mask"]).shape == (B, 1, 128)
    # circular padding + stride-2 downsampling: a roll by a multiple of 8 commutes with the network
    torch.testing.assert_close(y_roll, torch.roll(y, shift, 1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mask_cols,nan_mag,guidance,eta", [
    ((), False, 2.0, 0.0), ((), True, 2.0, 0.0), ((1, 4), False, 1.0, 0.5), ((0, 1, 2, 3, 4), False, 3.0, 0.0)])
def test_loss_and_sampler(mask_cols, nan_mag, guidance, eta):
    torch.manual_seed(0)
    m, diff, B = UNet1D(6, 3, width=16, n_res=1), Diffusion(), 4
    c = cond_batch(B, mask_cols, nan_mag)
    loss = diff.loss(m, torch.rand(B, 128) * 2 - 1, drop_conditions(c), weight=torch.rand(B))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    x = diff.sample(m.eval(), c, steps=5, guidance=guidance, eta=eta, generator=torch.Generator().manual_seed(1))
    assert x.shape == (B, 128) and torch.isfinite(x).all() and x.abs().max() <= 1


@pytest.mark.parametrize("period,amplitude,mean_mag", [(0.5537, 0.42, 15.3), (12.3, 0.05, 0.0), (0.0712, 1.7, 9.1)])
def test_render_period_and_amplitude_exact(run_dir, period, amplitude, mean_mag):
    rng = np.random.default_rng(3)
    prof = normalise_profiles(np.roll(fake_shapes("RR", 1, rng), 17, 1))
    assert prof[0, 0] == 1 and prof.min() == 0
    tb = TemplateBank(run_dir / "templates.npz")
    c0 = render(prof, [period], [amplitude], [mean_mag], tb, [0], np.random.default_rng(0), noise=False)[0]
    t0 = c0["t0"]  # the same rng seed gives the same t0, so place times on the profile nodes
    cyc = np.random.default_rng(1).integers(0, 500, 128)
    times = t0 + period * (cyc + (np.arange(128) + 0.5) / 128)
    c = render(prof, [period], [amplitude], [mean_mag], tb, [0], np.random.default_rng(0), times=times, noise=False)[0]
    assert c["t0"] == t0
    np.testing.assert_allclose(c["mag"], mean_mag + amplitude * (prof[0] - prof[0].mean()), atol=1e-9)
    assert abs(c["mag"].max() - c["mag"].min() - amplitude) < 1e-9


@pytest.mark.parametrize("kwargs,match", [({"survey": "VVV-K"}, "valid survey names: OGLE-I"),
                                          ({"cls": "Mira"}, "valid class names: RR, EB, SIN")])
def test_vocab_validation(run_dir, kwargs, match):
    g = Generator.from_pretrained(run_dir, device="cpu")
    with pytest.raises(ValueError, match=match):
        g.generate(**{"survey": "VVV-Ks", "period": 1.0, "amplitude": 0.3, **kwargs})


def test_end_to_end(run_dir, tmp_path):
    g = Generator.from_pretrained(run_dir, device="cpu")
    a = generate("VVV-Ks", [0.5, 0.6, 0.7], 0.3, mag=14.0, cls="RR", n=3, seed=7, steps=4, model=g)
    b = generate("VVV-Ks", [0.5, 0.6, 0.7], 0.3, mag=14.0, cls="RR", n=3, seed=7, steps=4, model=g)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x["mag"], y["mag"])
    assert [c["period"] for c in a] == [0.5, 0.6, 0.7] and all(c["mean_mag"] == 14.0 for c in a)
    assert {"time", "mag", "err", "period", "amplitude", "mean_mag", "survey", "cls", "template_sourceid",
            "profile"} <= set(a[0])
    rel = g.generate("TESS-T", 2.0, 0.1, n=2, seed=1, steps=3, times=np.linspace(0, 27, 500))
    assert all(c["mean_mag"] == 0.0 and len(c["time"]) == 500 for c in rel)  # no absolute mags: relative
    ds = ProfileDataset(run_dir, min_snr=20, tier_weight=0.5)
    item = ds[0]
    assert item["x"].shape == (128,) and item["mask"].shape == (5,) and (ds.arrays["snr"][ds.rows] >= 20).all()
    out = tmp_path / "out.csv"
    cli_main(["generate", "--model", str(run_dir), "--survey", "OGLE-I", "--period", "0.55", "--amplitude", "0.4",
              "--mag", "16", "--cls", "EB", "-n", "2", "--seed", "0", "--steps", "3", "--device", "cpu", "--noise", "gaussian",
              "--out", str(out)])
    lines = out.read_text().splitlines()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert lines[0] == "id,time,mag,err" and len(meta) == 2 and meta[0]["noise"] == "gaussian"
    assert all(c["noise"] == "residual" for c in a) and all(c["noise"] == "gaussian" for c in rel)


def noise_of(tb, j, noise, seed, times=None):
    """Rendered curve, its added noise (rendered minus noiseless; t0 is drawn first, so it matches) and
    the curve's errors."""
    prof = normalise_profiles(fake_shapes("SIN", 1, np.random.default_rng(0)))
    args = (prof, [0.7], [0.3], [15.0], tb, [j])
    c = render(*args, np.random.default_rng(seed), times=times, noise=noise)[0]
    c0 = render(*args, np.random.default_rng(seed), times=times, noise=None)[0]
    return c, c["mag"] - c0["mag"]


@pytest.mark.parametrize("bank,noise,times,applied", [
    ("v2", "residual", None, "residual"), ("v2", True, None, "residual"), ("v2", "gaussian", None, "gaussian"),
    ("v2", "residual", "user", "gaussian"), ("v1", "residual", None, "gaussian"), ("v2", "none", None, "none"),
    ("v2", False, None, "none")])
def test_noise_modes(tmp_path, bank, noise, times, applied):
    tb = TemplateBank(make_fake_bank(tmp_path, n=20, m=12, v2=bank == "v2") / "templates.npz")
    j = 3
    t = np.linspace(0, 300, 4000) if times == "user" else None
    c, dm = noise_of(tb, j, noise, seed=5, times=t)
    assert c["noise"] == applied
    if applied == "none":
        assert not dm.any()
    elif applied == "residual":
        r, e = tb.get_resid(j), tb.get(j)[1]
        # a shifted, sign-flipped copy: rms, outliers and |resid| / err pairs are preserved exactly
        np.testing.assert_allclose(np.sort(np.abs(dm)), np.sort(np.abs(r)), atol=1e-6)
        np.testing.assert_allclose(np.sort(np.abs(dm) / c["err"]), np.sort(np.abs(r) / e), rtol=1e-6)
        assert abs(np.sqrt(np.mean(dm ** 2)) / np.sqrt(np.mean(r ** 2)) - 1) < 1e-6
        assert np.mean(np.abs(dm) / c["err"] > 5) == np.mean(np.abs(r) / e > 5) > 0
        k = [int(np.argmin([np.abs(np.roll(r, s) * sg - noise_of(tb, j, noise, sd)[1]).max()
                            for s in range(len(r)) for sg in (1, -1)])) for sd in range(12)]
        assert len({x // 2 for x in k}) > 6 and len({x % 2 for x in k}) == 2  # shifts vary, both signs occur
        np.testing.assert_array_equal(dm, noise_of(tb, j, noise, seed=5)[1])  # seeded
    else:  # gaussian: N(0, (err * scatter)^2), errors resampled from the template for user times
        z = dm / (c["err"] * c["scatter"])
        assert abs(z.std() - 1) < 0.15 and abs(z.mean()) < 0.15
        assert set(np.round(c["err"], 7)) <= set(np.round(tb.get(j)[1], 7))


@pytest.mark.parametrize("v2,max_repeats,expect_max", [(True, 2, 2), (True, None, 4), (True, 1, 1), (False, 2, None),
                                                      ("quality", None, None)])
def test_bank_filters(tmp_path, v2, max_repeats, expect_max):
    """max-repeats row filter (ignored for v1); "quality": TemplateBank.pick uses only quality templates
    when the bank has ``quality`` and falls back to all templates of a survey that has none."""
    if v2 == "quality":
        d = make_fake_bank(tmp_path, n=20, m=24)
        z = dict(np.load(d / "templates.npz"))
        q = np.arange(24) % 3 == 0
        q[z["survey"] == 5] = False  # survey 5 has no quality template
        np.savez(d / "templates.npz", **z, quality=q)
        tb, rng = TemplateBank(d / "templates.npz"), np.random.default_rng(0)
        for s in range(6):
            picked = {tb.pick(s, None, rng) for _ in range(60)}
            pool = np.flatnonzero((tb.survey == s) & (q if q[tb.survey == s].any() else True))
            assert picked == set(pool.tolist())
        return
    arrays, _ = load_bank(make_fake_bank(tmp_path, n=400, m=6, v2=v2))
    rows = select_rows(arrays, max_repeats=max_repeats)
    if expect_max is None:  # bank v1 has no repeats: the filter is ignored
        assert "repeats" not in arrays and len(rows) == 400
    else:
        assert arrays["repeats"][rows].max() == expect_max
        assert len(rows) == (arrays["repeats"] <= (max_repeats or 99)).sum()
