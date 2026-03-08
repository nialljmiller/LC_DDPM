"""
infer.py — Conditional light curve generation.

Usage examples
--------------
# Generate with specific catalog features (values in physical units):
python infer.py \
    --milestone 750000 \
    --true_period 1.5 \
    --amplitude 0.4 \
    --mean_mag 14.2 \
    --skewness -0.3 \
    --kurtosis 0.8 \
    --ad_pvalue 0.01 \
    --stetson_k 0.85 \
    --color_jk 1.2 \
    --color_hk 0.3

# Omit any feature to use its training-set mean (i.e. the most typical value).
# Unconditional checkpoint (no catalog_fits during training):
python infer.py --milestone 750000 --batches 10
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_flow_matching import (
    Unet, StableFlowMatching, Trainer, LightcurveDataset,
    DEFAULT_COND_FEATURES,
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPATIAL_SIZE = 128
CHANNELS     = 3

CONFIGS = {
    "primvs": {"logdir": "./logs/"},
    "probes": {"logdir": "./logs/probes/"},
    "sdss":   {"logdir": "./logs/sdss/"},
}


# --------------------------------------------------------------------------- #
#  Feature handling                                                            #
# --------------------------------------------------------------------------- #

def load_feature_stats(logdir):
    """Load normalisation stats written by Trainer at the start of training."""
    path = Path(logdir) / "feature_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        stats = json.load(f)
    print(f"Loaded feature stats from {path}")
    print(f"  Features: {stats['names']}")
    return stats


def build_cond_tensor(feature_stats, user_values, batch_size, device):
    """
    Normalise user-supplied physical feature values and broadcast to (batch_size, F).

    Args:
        feature_stats : dict with "names", "mean", "std"
        user_values   : dict of {feature_name: physical_value}  (may be partial)
        batch_size    : int
        device        : torch.device

    Returns:
        torch.Tensor (batch_size, F), normalised, on `device`
    """
    names = feature_stats["names"]
    means = np.array(feature_stats["mean"])
    stds  = np.array(feature_stats["std"])
    F     = len(names)

    # Start from zeros in normalised space (= training-set mean in physical space)
    vec = np.zeros(F, dtype=np.float32)

    for i, fname in enumerate(names):
        if fname in user_values:
            val = float(user_values[fname])
            # Apply the same transform used at training time
            if "period" in fname.lower():
                val = np.log1p(abs(val))
            vec[i] = (val - means[i]) / stds[i]

    cond = torch.from_numpy(vec).unsqueeze(0).expand(batch_size, F).to(device)
    return cond


def plot_lightcurves(samples, outdir, start_index, feature_names=None,
                     feature_stats=None, cond_tensor=None):
    """
    Save individual light curve plots (unfolded + phase-folded side-by-side)
    for each sample.
    """
    samples_np = samples.detach().cpu().numpy()

    cond_denorm = None
    if cond_tensor is not None and feature_stats and feature_names:
        cond_np     = cond_tensor.detach().cpu().numpy()
        means       = np.array(feature_stats["mean"])
        stds        = np.array(feature_stats["std"])
        cond_denorm = cond_np * stds + means
        for j, fname in enumerate(feature_names):
            if "period" in fname.lower():
                cond_denorm[:, j] = np.expm1(cond_denorm[:, j])

    for k, sample in enumerate(samples_np):
        mag   = sample[0].flatten().astype(np.float64)
        err   = np.abs(sample[1].flatten().astype(np.float64))
        phase = sample[2].flatten().astype(np.float64)

        stride     = max(1, len(mag) // 500)
        idx        = np.arange(0, len(mag), stride)
        mag_s, err_s, phase_s = mag[idx], err[idx], phase[idx]

        fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(10, 4))

        ax_u.errorbar(np.arange(len(mag_s)), mag_s, yerr=err_s,
                      fmt="o", ms=3, alpha=0.7,
                      color="steelblue", ecolor="lightsteelblue",
                      elinewidth=0.8, capsize=0)
        ax_u.invert_yaxis()
        ax_u.set_xlabel("Index")
        ax_u.set_ylabel("Magnitude")
        ax_u.set_title("Unfolded")

        order = np.argsort(phase_s)
        ax_p.errorbar(phase_s[order], mag_s[order], yerr=err_s[order],
                      fmt="o", ms=3, alpha=0.7,
                      color="tomato", ecolor="lightsalmon",
                      elinewidth=0.8, capsize=0)
        ax_p.invert_yaxis()
        ax_p.set_xlabel("Phase")
        ax_p.set_ylabel("Magnitude")
        ax_p.set_title("Phase-Folded")

        # Annotation from denormalised features
        if cond_denorm is not None and feature_names:
            row   = cond_denorm[k] if k < len(cond_denorm) else cond_denorm[0]
            lines = [f"{n}: {row[j]:.4g}" for j, n in enumerate(feature_names)]
            fig.text(0.5, -0.02, "   |   ".join(lines),
                     ha="center", fontsize=7, style="italic")

        fig.suptitle(f"Generated Light Curve #{start_index + k}")
        fig.tight_layout()
        out_path = outdir / f"{start_index + k:05d}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser("Stable Flow Matching — Conditional Inference")
    parser.add_argument("--dataset",          default="primvs", choices=list(CONFIGS.keys()))
    parser.add_argument("--milestone",        default=750000,   type=int)
    parser.add_argument("--batches",          default=10,       type=int,
                        help="Number of batches to generate")
    parser.add_argument("--batch_size",       default=16,       type=int)
    parser.add_argument("--num_sample_steps", default=100,      type=int)
    parser.add_argument("--lambda_z",         default=2.0,      type=float)
    parser.add_argument("--lambda_tau",       default=1.0,      type=float)
    parser.add_argument("--use_midpoint",     action="store_true",
                        help="Use midpoint (RK2) integrator for better quality")
    parser.add_argument("--outdir",           default="inferred/", type=str)
    parser.add_argument("--save_npy",         action="store_true",
                        help="Also save raw .npy tensors alongside plots")

    # ---- One argument per conditioning feature (physical units) ------------
    # Any omitted feature defaults to its training-set mean.
    for fname in DEFAULT_COND_FEATURES:
        parser.add_argument(f"--{fname}", default=None, type=float,
                            help=f"Physical value for '{fname}' "
                                 "(default: training-set mean)")

    args = parser.parse_args()

    cfg    = CONFIGS[args.dataset]
    logdir = Path(cfg["logdir"])

    # ---- Load feature stats ------------------------------------------------
    feature_stats = load_feature_stats(logdir)
    feature_names = feature_stats["names"] if feature_stats else None
    num_cond      = len(feature_names) if feature_names else 0

    # ---- Collect user-supplied feature values ------------------------------
    user_values = {}
    if feature_names:
        for fname in feature_names:
            val = getattr(args, fname, None)
            if val is not None:
                user_values[fname] = val
        if user_values:
            print(f"User-specified features: {user_values}")
        else:
            print("No features specified — using training-set means for all.")

    # ---- Build model -------------------------------------------------------
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=CHANNELS,
        num_cond_features=num_cond,
    ).to(DEVICE)

    flow = StableFlowMatching(
        model,
        spatial_size=SPATIAL_SIZE,
        channels=CHANNELS,
        lambda_z=args.lambda_z,
        lambda_tau=args.lambda_tau,
        num_sample_steps=args.num_sample_steps,
    ).to(DEVICE)

    dummy_ds = LightcurveDataset(
        [np.zeros((CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE))]
    )
    trainer = Trainer(
        flow, dummy_ds,
        logdir=cfg["logdir"],
        spatial_size=SPATIAL_SIZE,
        train_batch_size=16,
        train_lr=2e-5,
        train_num_steps=750001,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        num_workers=0,
        rank=[0],
    )
    trainer.load(args.milestone)

    # ---- Build conditioning tensor -----------------------------------------
    cond = None
    if feature_names:
        cond = build_cond_tensor(
            feature_stats, user_values, args.batch_size, DEVICE
        )
        # Print denormalised values actually used
        means       = np.array(feature_stats["mean"])
        stds        = np.array(feature_stats["std"])
        cond_np     = cond[0].cpu().numpy()
        denorm      = cond_np * stds + means
        for j, fname in enumerate(feature_names):
            if "period" in fname.lower():
                denorm[j] = np.expm1(denorm[j])
        print("\nConditioning values used:")
        for j, fname in enumerate(feature_names):
            marker = "  ← user" if fname in user_values else "  (mean)"
            print(f"  {fname:20s} = {denorm[j]:.4g}{marker}")

    # ---- Generate ----------------------------------------------------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_fn  = flow.sample_midpoint if args.use_midpoint else flow.sample
    total      = 0

    for _ in range(args.batches):
        batch = sample_fn(batch_size=args.batch_size, cond=cond)

        plot_lightcurves(
            batch, outdir, start_index=total,
            feature_names=feature_names,
            feature_stats=feature_stats,
            cond_tensor=cond,
        )

        if args.save_npy:
            for s in batch.detach().cpu().numpy():
                np.save(outdir / f"{int(time.time())}_{total:05d}.npy", s)

        total += len(batch)

    print(f"\nGenerated {total} light curves → {outdir}")


if __name__ == "__main__":
    main()
