"""Train the LC_DDPM shape model on a profile bank (see docs/DESIGN.md).

    python training/train.py --bank BANK_DIR --out RUN_DIR [--steps 200000 --batch 512 --width 64]

The whole bank is loaded onto the device and batches are indexed directly. Checkpoints
(``ckpt_XXXXXXX.pt``, last 3 kept, plus ``best.pt`` by validation loss) go to --out every
--ckpt-every steps; a rerun resumes from the newest one. The final EMA weights are written
to ``--out/lc_ddpm.pt`` (the format ``lc_ddpm.Generator`` loads) with a ``templates.npz``
symlink to the bank's, so ``Generator.from_pretrained(RUN_DIR)`` works.

--max-repeats N (default 2) drops bank rows with ``repeats > N`` (bank v2; ignored for v1).
--init CKPT starts from the weights of an exported ``lc_ddpm.pt`` (or a training ``ckpt_*.pt``) for
fine-tuning; model config and condition normalisation are taken from it. Resuming from a
``ckpt_*.pt`` in --out takes precedence over --init.

Requires the package: run `pip install -e .` from the repo root first.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lc_ddpm.api import auto_device
from lc_ddpm.data import compute_norm, encode_conditions, load_bank, row_weights, select_rows
from lc_ddpm.diffusion import Diffusion, drop_conditions
from lc_ddpm.model import UNet1D
from lc_ddpm.render import normalise_profiles


def export(ema, vocab, norm, card):
    return {"format": "lc_ddpm-2", "config": ema.config, "state_dict": ema.state_dict(), "vocab": vocab,
            "norm": norm, "T": 1000, "card": card}


def lr_at(step, a):
    warm = min(1.0, (step + 1) / a.warmup)
    return a.lr * warm * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1.0)))


@torch.no_grad()
def val_loss(model, diff, X, cond, batch):
    """Deterministic validation loss: fixed noise and timesteps, all conditions present."""
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(X.shape, generator=gen).to(X.device)
    t = torch.randint(0, diff.T, (len(X),), generator=gen).to(X.device)
    tot = 0.0
    for s in range(0, len(X), batch):
        sl = slice(s, s + batch)
        tot += diff.loss(model, X[sl], {k: v[sl] for k, v in cond.items()}, noise=noise[sl], t=t[sl]).item() * len(X[sl])
    return tot / len(X)


def sample_grid(model, diff, X, cond, vocab, path, steps=50):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    gen = torch.Generator().manual_seed(1)
    x = normalise_profiles((diff.sample(model, cond, steps=steps, generator=gen).cpu().numpy() + 1) / 2)
    fig, axes = plt.subplots(4, 6, figsize=(15, 8), sharex=True)
    ph = (np.arange(x.shape[1]) + 0.5) / x.shape[1]
    for i, ax in enumerate(axes.flat[:len(x)]):
        ax.plot(ph, (X[i].cpu().numpy() + 1) / 2, color="0.6", lw=1, label="real")
        ax.plot(ph, x[i], color="C0", lw=1, label="sample")
        ax.invert_yaxis()
        s, c = vocab["surveys"][int(cond["survey"][i])], vocab["classes"][int(cond["cls"][i])]
        ax.set_title(f"{s} {c}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--min-snr", type=float)
    p.add_argument("--max-repeats", type=int, default=2,
                   help="keep rows with repeats <= N (bank v2 only); negative = no filter")
    p.add_argument("--init", help="checkpoint to initialise weights from (fine-tuning)")
    p.add_argument("--tier-weight", type=float, default=0.5, help="sampling weight of tier-1 rows (tier 0 = 1)")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--val-max", type=int, default=8192, help="rows used for the validation loss")
    p.add_argument("--ckpt-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device or auto_device()
    arrays, vocab = load_bank(a.bank)
    rows = select_rows(arrays, a.min_snr, max_repeats=a.max_repeats if a.max_repeats >= 0 else None)
    perm = np.random.default_rng(a.seed).permutation(rows)
    n_val = max(1, int(round(a.val_frac * len(rows))))
    val_rows, train_rows = np.sort(perm[:n_val][:a.val_max]), np.sort(perm[n_val:])

    ckpts = sorted(out.glob("ckpt_*.pt"))
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False) if ckpts else None
    init = None
    if a.init and state:
        print(f"--init {a.init} ignored: resuming from {ckpts[-1].name}")
    elif a.init:
        init = torch.load(a.init, map_location="cpu", weights_only=False)
        if "vocab" in init and (init["vocab"]["surveys"] != vocab["surveys"] or init["vocab"]["classes"] != vocab["classes"]):
            raise SystemExit(f"--init {a.init}: survey/class vocabulary differs from the bank's vocab.json")
    norm = (state or init or {}).get("norm") or compute_norm({k: arrays[k][train_rows]
                                                              for k in ("period", "amplitude", "mag")})

    def tensors(r):
        X = torch.as_tensor(2 * arrays["profile"][r] - 1, dtype=torch.float32, device=dev)
        cond = encode_conditions(arrays["survey"][r], arrays["cls"][r], arrays["period"][r], arrays["amplitude"][r],
                                 arrays["mag"][r], norm, dev)
        return X, cond

    X, cond = tensors(train_rows)
    Xv, condv = tensors(val_rows)
    W = torch.as_tensor(row_weights(arrays, train_rows, a.tier_weight), device=dev)

    cfg = (state or init or {}).get("config")  # resumed or initialised runs keep their architecture
    if cfg:
        model = UNet1D(**cfg)
    else:
        model = UNet1D(len(vocab["surveys"]), len(vocab["classes"]), width=a.width, nbins=X.shape[1])
    if init:  # exported lc_ddpm.pt / best.pt hold EMA weights in "state_dict"; ckpt_*.pt in "ema"
        model.load_state_dict(init.get("state_dict") or init["ema"])
        print(f"initialised weights from {a.init}")
    model = model.to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step, best = 0, float("inf")
    if state:
        model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); opt.load_state_dict(state["opt"])
        step, best = state["step"], state["best"]
        print(f"resumed from {ckpts[-1].name} at step {step}")
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{len(train_rows)} train rows, {len(val_rows)} val rows, {n_params / 1e6:.2f} M params, device {dev}")

    diff = Diffusion(1000)
    torch.manual_seed(a.seed + step)
    grid_idx = torch.as_tensor(np.random.default_rng(1).choice(len(Xv), min(24, len(Xv)), replace=False), device=dev)
    card = lambda: {"bank": str(Path(a.bank).resolve()), "steps": step, "train_rows": int(len(train_rows)),
                    "min_snr": a.min_snr, "max_repeats": a.max_repeats,
                    "init": str(Path(a.init).resolve()) if a.init else None, "tier_weight": a.tier_weight,
                    "best_val_loss": best, "params": n_params,
                    "date": time.strftime("%Y-%m-%d")}
    log = open(out / "log.jsonl", "a")
    t_start, run_loss, run_n = time.time(), 0.0, 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.startswith("cuda") else torch.autocast("cpu", enabled=False)
    model.train()
    while step < a.steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.multinomial(W, a.batch, replacement=True)
        c = drop_conditions({k: v[idx] for k, v in cond.items()})
        with amp:
            loss = diff.loss(model, X[idx], c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = min(a.ema, (1 + step) / (10 + step))  # EMA warm-up
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        run_loss += loss.item(); run_n += 1

        if step % a.log_every == 0 or step % a.ckpt_every == 0 or step == a.steps:
            rec = {"step": step, "loss": run_loss / run_n, "lr": lr, "time": round(time.time() - t_start, 1)}
            if step % a.ckpt_every == 0 or step == a.steps:
                rec["val_loss"] = val_loss(ema, diff, Xv, condv, a.batch)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]
                    torch.save(export(ema, vocab, norm, card()), out / "best.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best": best, "norm": norm, "config": ema.config, "args": vars(a)}, out / f"ckpt_{step:07d}.pt")
                for old in sorted(out.glob("ckpt_*.pt"))[:-3]:
                    old.unlink()
                sample_grid(ema, diff, Xv[grid_idx], {k: v[grid_idx] for k, v in condv.items()}, vocab,
                            out / f"samples_{step:07d}.png")
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_n = 0.0, 0

    torch.save(export(ema, vocab, norm, card()), out / "lc_ddpm.pt")
    link = out / "templates.npz"
    if not link.exists():
        link.symlink_to(Path(a.bank).resolve() / "templates.npz")
    print(f"wrote {out / 'lc_ddpm.pt'}")


if __name__ == "__main__":
    main()
