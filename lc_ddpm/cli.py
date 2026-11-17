"""Command line: ``lc-ddpm generate`` and ``lc-ddpm info``."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def plot_curves(curves, path, max_curves=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = min(len(curves), max_curves)
    cols = min(k, 4)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, c in zip(axes.flat, curves):
        ph = np.mod((c["time"] - c["t0"]) / c["period"], 1.0)
        ax.errorbar(ph, c["mag"], c["err"], fmt=".", ms=2, lw=0.3, alpha=0.6)
        ax.invert_yaxis()
        ax.set_title(f"P={c['period']:.4g} d  A={c['amplitude']:.3g}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    fig.supxlabel("phase"); fig.supylabel("mag")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def cmd_generate(a):
    from .api import Generator
    g = Generator.from_pretrained(a.model, templates=a.templates, device=a.device)
    curves = g.generate(a.survey, a.period, a.amplitude, mag=a.mag, cls=a.cls, n=a.n, seed=a.seed,
                        guidance=a.guidance, steps=a.steps)
    out = Path(a.out)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "time", "mag", "err"])
        for i, c in enumerate(curves):
            for row in zip(c["time"], c["mag"], c["err"]):
                w.writerow([i, *(f"{v:.8g}" for v in row)])
    meta = [{"id": i, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()
                         if k not in ("time", "mag", "err")}} for i, c in enumerate(curves)]
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    if a.plot:
        plot_curves(curves, a.plot)
    print(f"wrote {len(curves)} curves to {out} (+ {out.with_suffix('.json').name})")


def cmd_info(a):
    from .api import Generator
    print(json.dumps(Generator.from_pretrained(a.model, templates=a.templates, device="cpu").card(), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="lc-ddpm", description="Generate variable-star light curves.")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="checkpoint file or directory (default $LC_DDPM_HOME or ~/.cache/lc_ddpm)")
    common.add_argument("--templates", help="templates.npz (default: next to the checkpoint)")
    g = sub.add_parser("generate", parents=[common], help="generate light curves to CSV")
    g.add_argument("--survey", required=True)
    g.add_argument("--period", type=float, required=True, help="days")
    g.add_argument("--amplitude", type=float, required=True, help="peak-to-peak, mag")
    g.add_argument("--mag", type=float, help="mean magnitude")
    g.add_argument("--cls", help="class family")
    g.add_argument("-n", type=int, default=1)
    g.add_argument("--out", required=True, help="CSV path; metadata goes to the same name with .json")
    g.add_argument("--plot", help="PNG of folded curves (needs matplotlib)")
    g.add_argument("--seed", type=int)
    g.add_argument("--guidance", type=float, default=2.0)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--device")
    g.set_defaults(func=cmd_generate)
    i = sub.add_parser("info", parents=[common], help="print vocabulary and model card")
    i.set_defaults(func=cmd_info)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
