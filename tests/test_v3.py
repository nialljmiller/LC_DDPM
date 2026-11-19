"""v3 tests: window rule, representation, model masking, sequence diffusion, build -> train -> generate."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lc_ddpm import Generator, GeneratorV3, generate
from lc_ddpm.data_v3 import (MIN_CYCLES, PHASE_GATE, feat_dim, load_v3, log_gaps, make_batch, measure_channels,
                             measure_values, times_from_log_gaps, window)
from lc_ddpm.diffusion import SeqDiffusion, drop_conditions
from lc_ddpm.model import PointTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import build_bank  # noqa: E402
import build_v3  # noqa: E402
import train_v3  # noqa: E402


def make_fake_v3(path, n_ogle=40, n_kep=12, seed=0):
    """Matched fake NN_Periodogram shards and bank v2.2. ``ogle_blg_rrlyr`` (.npyd): 40-300 points over 1000 d
    with seasonal gaps; ``kepler_eb`` (.npz only): 30-min cadence over 90 d (4320 points). Star 0 of each
    shard is in the val split but has a bank row (the build must drop it); the last Kepler star has
    P = 80 d (span < 1.5 P, dropped)."""
    rng = np.random.default_rng(seed)
    shards, bank = path / "shards", path / "bank"
    shards.mkdir(parents=True, exist_ok=True); bank.mkdir(exist_ok=True)
    rows = []
    for shard, n, kepler in (("ogle_blg_rrlyr", n_ogle, False), ("kepler_eb", n_kep, True)):
        T, M, E, off, meta, lab = [], [], [], [0], [], []
        for i in range(n):
            if kepler:
                P = 80.0 if i == n - 1 else 10 ** rng.uniform(-0.5, 1.2)
                t = 54900 + np.arange(4320) / 48.0
            else:
                P = 10 ** rng.uniform(-0.5, -0.1)
                t = 56000 + rng.uniform(0, 1000, rng.integers(40, 300))
                t = t[np.mod(t, 365) < 200]
            t = rng.permutation(t)  # shards are not necessarily sorted
            A, e = 10 ** rng.uniform(-1, -0.3), rng.uniform(0.005, 0.03, len(t))
            m = 15 + A * np.sin(2 * np.pi * t / P) + rng.normal(0, 1, len(t)) * e
            T.append(t); M.append(m); E.append(e); off.append(off[-1] + len(t))
            meta.append({"sourceid": f"{shard[:3]}{i}", "split": "val" if i == 0 else "train",
                         **({"cls": "EA"} if kepler else {})})
            lab.append({"index": i, "freq": 1 / P})
            rows.append((shard, i, P, A, np.nan if kepler else 15.0, 4 if kepler else 0, 5 if kepler else 0))
        arrs = dict(time=np.concatenate(T), mag=np.concatenate(M).astype(np.float32),
                    err=np.concatenate(E).astype(np.float32), offsets=np.array(off, np.int64))
        np.savez(shards / f"{shard}.npz", **arrs)
        if not kepler:
            (shards / f"{shard}.npyd").mkdir(exist_ok=True)
            for k, v in arrs.items():
                np.save(shards / f"{shard}.npyd" / f"{k}.npy", v)
        json.dump(meta, open(shards / f"{shard}_meta.json", "w"))
        json.dump(lab, open(shards / f"{shard}_labels.json", "w"))
    col = lambda j, dt: np.array([r[j] for r in rows], dt)
    np.savez(bank / "profiles.npz", shard=col(0, "U"), index=col(1, np.int32), period=col(2, np.float64),
             amplitude=col(3, np.float32), mag=col(4, np.float32), survey=col(5, np.int16), cls=col(6, np.int16),
             tier=np.zeros(len(rows), np.int8), snr=np.full(len(rows), 50, np.float32),
             repeats=np.ones(len(rows), np.int8), sourceid=np.array([f"{r[0][:3]}{r[1]}" for r in rows]),
             profile=np.zeros((len(rows), 128), np.float32))
    json.dump({"surveys": build_bank.SURVEYS, "classes": build_bank.CLASSES, "class_map": {}, "nbins": 128},
              open(bank / "vocab.json", "w"))
    json.dump({}, open(bank / "excluded.json", "w"))
    return shards, bank


@pytest.mark.parametrize("n,cycles,n_max,rule,check", [
    (50, 2.0, 64, 1, "whole"),                 # n <= n_max: whole curve
    (50, MIN_CYCLES * 0.99, 64, None, ""),     # spans < 1.5 P: not used
    (80, MIN_CYCLES, 1024, 1, "whole"),        # exactly 1.5 P is kept
    (2000, 200.0, 64, 2, "consecutive"),       # 64 consecutive points span ~6 P
    (2000, 20.0, 64, 3, "3P"),                 # 64 consecutive span 0.6 P: 3 P window, thinned
    (2000, 2.0, 64, 3, "all_span"),            # 1.5 P <= span < 3 P: whole span, thinned
])
def test_window_rule(n, cycles, n_max, rule, check):
    P = 0.7
    t = np.sort(np.random.default_rng(1).uniform(0, cycles * P, n))
    t[-1] = t[0] + cycles * P
    w = window(t, P, np.random.default_rng(2), n_max)
    if rule is None:
        assert w is None
        return
    idx, r = w
    assert r == rule and len(idx) <= n_max and np.all(np.diff(idx) > 0)
    if check == "whole":
        assert len(idx) == n
    elif check == "consecutive":
        assert len(idx) == n_max and idx[-1] - idx[0] == n_max - 1 and t[idx[-1]] - t[idx[0]] >= 3 * P
    elif check == "3P":
        assert len(idx) == n_max and t[idx[-1]] - t[idx[0]] <= 3 * P and t[idx[-1]] - t[idx[0]] > 2.5 * P
    else:
        assert len(idx) == n_max and t[idx[-1]] - t[idx[0]] > 1.8 * P


@pytest.mark.parametrize("t", [np.array([0.0, 0.5, 0.5, 3.0, 10.0]), np.sort(np.random.default_rng(0).uniform(5, 900, 300))])
def test_representation_roundtrip(t):
    rng = np.random.default_rng(0)
    m, e, A = 14 + rng.normal(0, 0.2, len(t)), rng.uniform(0.01, 0.1, len(t)), 0.37
    mm, ee = measure_values(measure_channels(m, e, A), np.median(m), A)
    np.testing.assert_allclose(mm, m); np.testing.assert_allclose(ee, e)
    tt = times_from_log_gaps(log_gaps(t), t[-1] - t[0])
    np.testing.assert_allclose(tt, t - t[0], atol=1e-6 * (t[-1] - t[0]))
    assert tt[-1] == t[-1] - t[0]  # span exact


def tiny(stage, **kw):
    torch.manual_seed(0)
    m = PointTransformer(2 if stage == "measure" else 1, feat_dim(stage, 64), 6, 14 if stage == "measure" else 0,
                         4 if stage == "measure" else 2, width=32, depth=2, heads=4,
                         gate=PHASE_GATE if stage == "measure" else None, **kw)
    for p in m.parameters():  # adaLN-Zero starts at zero output: randomise for the tests
        torch.nn.init.normal_(p, std=0.1)
    return m


def seq_cond(stage, lengths, L, mask_cols=()):
    B = len(lengths)
    K = 4 if stage == "measure" else 2
    pad = torch.arange(L)[None] < torch.as_tensor(lengths)[:, None]
    mask = torch.ones(B, 2 + K, dtype=torch.bool)
    mask[:, list(mask_cols)] = False
    return {"survey": torch.arange(B) % 6, "cls": torch.arange(B) % 3, "num": torch.randn(B, K), "mask": mask,
            "feat": torch.randn(B, L, feat_dim(stage, 64)) * pad[..., None], "pad": pad}


@pytest.mark.parametrize("stage,lengths", [("measure", [5, 16, 9]), ("cadence", [3, 8])])
def test_transformer_padding(stage, lengths):
    """Outputs at real points do not depend on padded content or on the padded length; padded outputs are 0;
    the phase features only matter when the period condition is present."""
    m = tiny(stage).eval()
    C, L = (2 if stage == "measure" else 1), max(lengths)
    c = seq_cond(stage, lengths, L)
    x, t = torch.randn(len(lengths), L, C) * c["pad"][..., None], torch.randint(0, 1000, (len(lengths),))
    with torch.no_grad():
        y = m(x, t, c)
        junk = {**c, "feat": c["feat"] + 5 * ~c["pad"][..., None]}
        y2 = m(x + 7 * ~c["pad"][..., None], t, junk)
        longer = {**c, "feat": torch.cat([c["feat"], torch.randn(len(lengths), 8, c["feat"].shape[2])], 1),
                  "pad": torch.cat([c["pad"], torch.zeros(len(lengths), 8, dtype=torch.bool)], 1)}
        y3 = m(torch.cat([x, torch.randn(len(lengths), 8, C)], 1), t, longer)[:, :L]
    assert y.shape == x.shape and not y[~c["pad"]].any()
    torch.testing.assert_close(y2, y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y3, y, atol=1e-5, rtol=1e-5)
    if stage == "measure":
        a, b = PHASE_GATE[1:]
        f2 = c["feat"].clone(); f2[..., a:b] += 1
        nop = {**c, "mask": c["mask"].clone()}; nop["mask"][:, PHASE_GATE[0]] = False
        with torch.no_grad():
            assert not torch.allclose(m(x, t, {**c, "feat": f2}), y)
            torch.testing.assert_close(m(x, t, {**nop, "feat": f2}), m(x, t, nop))


@pytest.mark.parametrize("stage,mask_cols,guidance,eta,grad_ckpt", [
    ("measure", (), 2.0, 0.0, False), ("measure", (1, 4), 1.0, 0.5, True), ("cadence", (), 1.5, 0.0, False)])
def test_seq_loss_and_sampler(stage, mask_cols, guidance, eta, grad_ckpt):
    m, diff = tiny(stage, grad_ckpt=grad_ckpt), SeqDiffusion()
    C, lengths = (2 if stage == "measure" else 1), [4, 11, 7]
    c = seq_cond(stage, lengths, 16, mask_cols)
    x0 = torch.randn(3, 16, C) * c["pad"][..., None]
    loss = diff.loss(m.train(), x0, drop_conditions(c), weight=torch.rand(3))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    x0b = x0.clone(); x0b[~c["pad"]] = 99.0  # padded targets do not enter the loss
    g = lambda: torch.Generator().manual_seed(3)
    tt, nz = torch.randint(0, 1000, (3,), generator=g()), torch.randn(3, 16, C, generator=g())
    torch.testing.assert_close(diff.loss(m.eval(), x0, c, noise=nz, t=tt), diff.loss(m, x0b, c, noise=nz, t=tt))
    x = diff.sample(m, c, C, steps=4, guidance=guidance, eta=eta, generator=g())
    assert x.shape == (3, 16, C) and torch.isfinite(x).all() and not x[~c["pad"]].any()
    torch.testing.assert_close(x, diff.sample(m, c, C, steps=4, guidance=guidance, eta=eta, generator=g()))


@pytest.fixture(scope="module")
def v3_run(tmp_path_factory):
    """Fake shards + bank -> build_v3 (n_max 64) -> two tiny trained stages in one run directory."""
    d = tmp_path_factory.mktemp("v3")
    shards, bank = make_fake_v3(d)
    build_v3.main(["--shards", str(shards), "--bank", str(bank), "--out", str(d / "data"), "--workers", "2",
                   "--n-max", "64", "--chunk", "7"])
    for stage in ("cadence", "measure"):
        train_v3.main(["--data", str(d / "data"), "--out", str(d / "run"), "--stage", stage, "--steps", "3",
                       "--batch", "4", "--accum", "2", "--width", "32", "--depth", "2", "--heads", "4",
                       "--ckpt-every", "2", "--log-every", "1", "--val-frac", "0.1", "--device", "cpu"])
    return d


def test_build_v3(v3_run):
    arrays, vocab = load_v3(v3_run / "data")
    summary = json.loads((v3_run / "data" / "summary.json").read_text())
    assert summary["dropped"] == {"not_train": 2, "short_span": 1}
    assert set(arrays["sourceid"]).isdisjoint({"ogl0", "kep0", "kep11"})
    n = arrays["n_points"]
    assert (n <= 64).all() and (n >= 10).all() and arrays["offsets"][-1] == n.sum()
    kep = arrays["survey"] == build_bank.SURVEYS.index("Kepler-Kp")
    assert set(arrays["rule"][kep]) <= {2, 3} and (n[kep] == 64).all() and np.isnan(arrays["mag"][kep]).all()
    for r in range(len(n)):
        t = arrays["time"][arrays["offsets"][r]:arrays["offsets"][r + 1]]
        assert t[0] == 0 and np.all(np.diff(t) >= 0) and t[-1] == arrays["span"][r]
        assert t[-1] >= (3 * arrays["period"][r] if arrays["rule"][r] == 2 else MIN_CYCLES * arrays["period"][r] - 1e-9) \
            or arrays["rule"][r] == 3
    x0, c = make_batch(arrays, np.arange(3), "measure", train_v3.compute_norm_v3(arrays, np.arange(len(n)), "measure"))
    assert x0.shape[2] == 2 and x0.shape[1] % 8 == 0 and c["pad"].sum(1).tolist() == n[:3].tolist()
    log = [json.loads(s) for s in (v3_run / "run" / "measure" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3] and "val_loss" in log[1]


@pytest.mark.parametrize("kwargs,cadence,n_pts", [
    ({}, "model", None), ({"n_points": 30, "span": 12.5}, "model", 30), ({"cadence": "real"}, "real", None),
    ({"times": np.linspace(3, 40, 50)}, "user", 50), ({"times": [np.linspace(0, 9, 20), np.linspace(1, 2, 11)]}, "user", None)])
def test_generate_v3(v3_run, kwargs, cadence, n_pts):
    g = Generator.from_pretrained(v3_run / "run", device="cpu")
    assert isinstance(g, GeneratorV3)
    args = dict(survey="OGLE-I", period=0.5, amplitude=0.3, mag=16.0, cls="RR", n=2, seed=4, steps=3, model=g, **kwargs)
    a, b = generate(**args), generate(**args)
    assert len(a) == 2 and all(c["cadence"] == cadence for c in a)
    for c, d in zip(a, b):
        np.testing.assert_array_equal(c["mag"], d["mag"]); np.testing.assert_array_equal(c["time"], d["time"])
        assert len(c["time"]) == len(c["mag"]) == len(c["err"]) == c["n_points"] <= 64
        assert c["time"][0] == 0 and np.all(np.diff(c["time"]) >= 0) and abs(c["time"][-1] - c["span"]) < 1e-9
        assert (c["err"] > 0).all() and np.isfinite(c["mag"]).all() and c["period"] == 0.5
    if n_pts:
        assert all(c["n_points"] == n_pts for c in a)
    if "span" in kwargs:
        assert all(c["span"] == 12.5 for c in a)
    rel = g.generate("Kepler-Kp", 2.0, 0.1, n=1, seed=0, steps=2)[0]
    assert rel["mean_mag"] == 0.0  # no absolute magnitudes: relative curves


def test_v3_errors_and_dispatch(v3_run):
    g = GeneratorV3.from_dir(v3_run / "run", device="cpu")
    for kw, match in [({"survey": "VVV-K"}, "valid survey names"), ({"times": np.arange(100.0)}, "2..64 points"),
                      ({"cadence": "real", "n_points": 20}, "leave them None")]:
        with pytest.raises(ValueError, match=match):
            g.generate(**{"survey": "OGLE-I", "period": 1.0, "amplitude": 0.3, **kw})
    g2 = Generator.from_pretrained(v3_run / "run" / "lc_ddpm_measure.pt", device="cpu")
    assert isinstance(g2, GeneratorV3) and g2.cadence is not None and g.card()["format"] == "lc_ddpm-3"


def test_cli_v3(v3_run, tmp_path):
    from lc_ddpm.cli import main as cli_main
    out = tmp_path / "v3.csv"
    cli_main(["generate", "--model", str(v3_run / "run"), "--survey", "Kepler-Kp", "--period", "1.3", "--amplitude", "0.2",
              "-n", "2", "--seed", "0", "--steps", "2", "--device", "cpu", "--n-points", "40", "--span", "5", "--out", str(out)])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert len(out.read_text().splitlines()) == 81 and meta[1]["cadence"] == "model" and meta[1]["n_points"] == 40
    cli_main(["info", "--model", str(v3_run / "run")])


def test_evaluate_v3(v3_run, tmp_path):
    import evaluate_v3
    out = tmp_path / "ev.json"
    evaluate_v3.main(["--model", str(v3_run / "run"), "--shards", str(v3_run / "shards"), "--shard", "ogle_blg_rrlyr",
                      "--split", "train", "--n", "12", "--epochs", "5", "--period-n", "4", "--steps", "2", "--min-class", "6",
                      "--modes", "v3_full,v3_measure,v3_real_cadence", "--device", "cpu", "--out", str(out)])
    r = json.loads(out.read_text())["surveys"]["OGLE-I"]
    assert r["n_stars"] == 12 and set(r["v3_full"]["auc"]) == {"profile", "stats", "both", "cadence", "gaps"}
    assert r["v3_measure"]["n_points_q"] == r["real"]["n_points_q"] and "RR" in r["v3_full"]["per_class"]
