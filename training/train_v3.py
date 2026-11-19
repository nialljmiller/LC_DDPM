"""Train one stage of LC_DDPM v3 (docs/DESIGN_v3.md) on a training file from build_v3.py.

    python training/train_v3.py --data V3_DIR --out RUN_DIR --stage cadence [--batch 8 --accum 4 --steps 200000]
    python training/train_v3.py --data V3_DIR --out RUN_DIR --stage measure [--batch 8 --accum 4 --steps 200000]

Both stages share RUN_DIR. Each stage keeps its log.jsonl, ckpt_XXXXXXX.pt (last 3) and best.pt (by
validation loss) in RUN_DIR/<stage>/ and resumes from the newest ckpt there. The final EMA weights go to
RUN_DIR/lc_ddpm_<stage>.pt, and RUN_DIR/cadences.npz links to the training file's cadences.npz, so
``lc_ddpm.Generator.from_pretrained(RUN_DIR)`` loads the v3 generator once both stages exist.

Examples live in CPU memory; each step builds a padded batch (length = longest example, rounded up to
a multiple of 8) and moves it to the device. --accum micro-batches are accumulated per optimiser step
(effective batch = batch * accum). bf16 autocast on CUDA; --grad-ckpt recomputes block activations in
the backward pass (less memory, about 30 % slower). The measure stage drops rows with repeats >
--max-repeats (as v2) and samples tier-1 rows with --tier-weight; the cadence stage uses all rows.

Requires the package: run `pip install -e .` from the repo root first.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import lr_at  # noqa: E402
from lc_ddpm.api import auto_device  # noqa: E402
from lc_ddpm.data_v3 import (CHANNELS, NUM_NAMES, PHASE_GATE, STAGES, compute_norm_v3, example, feat_dim,  # noqa: E402
                             load_v3, make_batch, measure_values)
from lc_ddpm.diffusion import SeqDiffusion, drop_conditions  # noqa: E402
from lc_ddpm.model import PointTransformer  # noqa: E402

DEFAULTS_PER_SURVEY = 20000  # rows of the (survey, period, n, span, mag) table stored for generation defaults


def defaults_table(arrays, rows, seed=0):
    """Per-example (survey, period, n_points, span, mag) of up to DEFAULTS_PER_SURVEY training rows per
    survey: generate() draws n and span (and mag when not given) jointly from one of these."""
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([rng.permutation(rows[arrays["survey"][rows] == s])[:DEFAULTS_PER_SURVEY]
                                   for s in np.unique(arrays["survey"][rows])]))
    return {k: arrays[k][keep] for k in ("survey", "period", "n_points", "span", "mag")}


def export(ema, stage, vocab, norm, n_max, card, defaults=None):
    return {"format": "lc_ddpm-3", "stage": stage, "config": ema.config, "state_dict": ema.state_dict(),
            "vocab": vocab, "norm": norm, "n_max": n_max, "T": 1000, "card": card, "defaults": defaults}


@torch.no_grad()
def val_loss(model, diff, arrays, rows, stage, norm, n_max, batch, dev):
    """Deterministic validation loss: fixed noise and timesteps, all conditions present."""
    gen = torch.Generator().manual_seed(0)
    tot = 0.0
    for s in range(0, len(rows), batch):
        x0, cond = make_batch(arrays, rows[s:s + batch], stage, norm, dev, n_max)
        noise = torch.randn(x0.shape, generator=gen).to(dev)
        t = torch.randint(0, diff.T, (len(x0),), generator=gen).to(dev)
        tot += diff.loss(model, x0, cond, noise=noise, t=t).item() * len(x0)
    return tot / len(rows)


def sample_plot(model, diff, arrays, rows, stage, norm, n_max, vocab, dev, path, steps=50):
    """Real (grey) vs generated (blue): folded mag for measure (on the real times), time raster for cadence."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x0, cond = make_batch(arrays, rows, stage, norm, dev, n_max)
    x = diff.sample(model, cond, CHANNELS[stage], steps=steps, generator=torch.Generator().manual_seed(1)).cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(14, 9))
    for ax, r, xi in zip(axes.flat, rows, x):
        t, m, e = example(arrays, r)
        P, A = arrays["period"][r], arrays["amplitude"][r]
        if stage == "measure":
            g = xi[:len(t)] * np.asarray(norm["chan_std"]) + np.asarray(norm["chan_mean"])
            gm, _ = measure_values(g, np.median(m), A)
            ax.plot(np.mod(t / P, 1), m, ".", color="0.6", ms=2)
            ax.plot(np.mod(t / P, 1), gm, ".", color="C0", ms=2)
            ax.invert_yaxis()
        else:
            lg = xi[:len(t) - 1, 0] * norm["gap_std"] + norm["gap_mean"]
            c = np.concatenate([[0], np.cumsum(10.0 ** lg)])
            ax.eventplot([t, t[-1] * c / c[-1]], colors=["0.6", "C0"], lineoffsets=[1, 0], linelengths=0.8)
        ax.set_title(f"{vocab['surveys'][arrays['survey'][r]]} {vocab['classes'][arrays['cls'][r]]} "
                     f"P={P:.3g} n={len(t)}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="build_v3.py output directory")
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--steps", type=int, default=200_000, help="optimiser steps (schedule length)")
    p.add_argument("--max-steps", type=int, help="stop after this many steps (schedule still uses --steps)")
    p.add_argument("--batch", type=int, default=8, help="micro-batch size")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimiser step")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing per block")
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--max-repeats", type=int, default=2, help="measure: keep rows with repeats <= N; negative = all")
    p.add_argument("--tier-weight", type=float, default=0.5, help="measure: sampling weight of tier-1 rows")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--val-max", type=int, default=1024)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    run = Path(a.out); out = run / a.stage; out.mkdir(parents=True, exist_ok=True)
    dev = a.device or auto_device()
    arrays, vocab = load_v3(a.data)
    n_max = int(arrays["n_max"])
    keep = np.ones(len(arrays["period"]), bool)
    if a.stage == "measure" and a.max_repeats >= 0:
        keep &= arrays["repeats"] <= a.max_repeats
    rows = np.flatnonzero(keep)
    perm = np.random.default_rng(a.seed).permutation(rows)
    n_val = max(1, int(round(a.val_frac * len(rows))))
    val_rows, train_rows = np.sort(perm[:n_val][:a.val_max]), np.sort(perm[n_val:])

    ckpts = sorted(out.glob("ckpt_*.pt"))
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False) if ckpts else None
    norm = state["norm"] if state else compute_norm_v3(arrays, train_rows, a.stage)
    cfg = state["config"] if state else dict(
        in_ch=CHANNELS[a.stage], feat_dim=feat_dim(a.stage, n_max), n_surveys=len(vocab["surveys"]),
        n_classes=len(vocab["classes"]) if a.stage == "measure" else 0, n_num=len(NUM_NAMES[a.stage]),
        width=a.width, depth=a.depth, heads=a.heads, gate=PHASE_GATE if a.stage == "measure" else None)
    model = PointTransformer(**cfg, grad_ckpt=a.grad_ckpt).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step, best = 0, float("inf")
    if state:
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); opt.load_state_dict(state["opt"])
        step, best = state["step"], state["best"]
        print(f"resumed from {ckpts[-1].name} at step {step}")
    n_params = sum(q.numel() for q in model.parameters())
    print(f"stage {a.stage}: {len(train_rows)} train rows, {len(val_rows)} val rows, {n_params / 1e6:.2f} M params, "
          f"device {dev}, batch {a.batch} x {a.accum}")

    W = torch.ones(len(train_rows))
    if a.stage == "measure":
        W = torch.as_tensor(np.where(arrays["tier"][train_rows] == 1, a.tier_weight, 1.0), dtype=torch.float32)
    defaults = defaults_table(arrays, train_rows) if a.stage == "measure" else None
    diff = SeqDiffusion(1000)
    torch.manual_seed(a.seed + step)
    plot_rows = np.random.default_rng(1).choice(val_rows, min(16, len(val_rows)), replace=False)
    card = lambda: {"data": str(Path(a.data).resolve()), "stage": a.stage, "steps": step,
                    "train_rows": int(len(train_rows)), "max_repeats": a.max_repeats, "tier_weight": a.tier_weight,
                    "batch": a.batch * a.accum, "best_val_loss": best, "params": n_params,
                    "date": time.strftime("%Y-%m-%d")}
    log = open(out / "log.jsonl", "a")
    t_start, run_loss, run_n = time.time(), 0.0, 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    last = min(a.steps, a.max_steps or a.steps)
    model.train()
    while step < last:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            idx = train_rows[torch.multinomial(W, a.batch, replacement=True).numpy()]
            x0, cond = make_batch(arrays, idx, a.stage, norm, dev, n_max)
            cond = drop_conditions(cond)
            with amp:
                loss = diff.loss(model, x0, cond) / a.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        run_loss += tot; run_n += 1

        if step % a.log_every == 0 or step % a.ckpt_every == 0 or step == last:
            rec = {"step": step, "loss": run_loss / run_n, "lr": lr, "time": round(time.time() - t_start, 1)}
            if dev.startswith("cuda"):
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if step % a.ckpt_every == 0 or step == last:
                with amp:
                    rec["val_loss"] = val_loss(ema, diff, arrays, val_rows, a.stage, norm, n_max, a.batch, dev)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]
                    torch.save(export(ema, a.stage, vocab, norm, n_max, card(), defaults), out / "best.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best": best, "norm": norm, "config": ema.config, "args": vars(a)},
                           out / f"ckpt_{step:07d}.pt")
                for old in sorted(out.glob("ckpt_*.pt"))[:-3]:
                    old.unlink()
                sample_plot(ema, diff, arrays, plot_rows, a.stage, norm, n_max, vocab, dev, out / f"samples_{step:07d}.png")
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_n = 0.0, 0

    torch.save(export(ema, a.stage, vocab, norm, n_max, card(), defaults), run / f"lc_ddpm_{a.stage}.pt")
    link = run / "cadences.npz"
    if not os.path.lexists(link) and (Path(a.data) / "cadences.npz").exists():
        link.symlink_to((Path(a.data) / "cadences.npz").resolve())
    print(f"wrote {run / f'lc_ddpm_{a.stage}.pt'}")


if __name__ == "__main__":
    main()
