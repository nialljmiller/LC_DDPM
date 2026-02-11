import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

from stable_flow_matching import (
    Unet, StableFlowMatching, Trainer, LightcurveDataset, delete_rand_items,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Config ----------------------------------------------------------------
LC_SIZE = 70
DATA_DIR = "/beegfs/car/njm/Periodic_Variables/Best/LC/*.csv"
CHANNELS = 3  # mag, magerr, phase


# ---- Load and preprocess data ---------------------------------------------
def load_sequences(pattern, lc_size, max_files=None):
    """Load CSV light curves into fixed-size numpy arrays."""
    files = glob.glob(pattern)
    if max_files is not None:
        files = files[:max_files]

    sequences = []
    for fi in tqdm(files, desc="Loading data"):
        try:
            mag, magerr, phase, time_arr = np.genfromtxt(
                fi, dtype="float", comments="#", delimiter=","
            ).T
        except Exception:
            continue

        if len(mag) <= lc_size:
            continue

        mag, magerr, time_arr, phase = delete_rand_items(
            mag, magerr, time_arr, phase, len(mag) - lc_size
        )

        sequence = np.stack(
            (np.tile(mag, (2, 2)),
             np.tile(magerr, (2, 2)),
             np.tile(phase, (2, 2))),
            axis=0,
        )

        if not np.any(np.isnan(sequence)):
            sequences.append(sequence)

    print(f"Loaded {len(sequences)} valid light curves.")
    return sequences


# ---- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Stable Flow Matching — Light Curve Training")
    parser.add_argument("--milestone", default=0, type=int, help="Checkpoint step to resume from")
    parser.add_argument("--lambda_z", default=2.0, type=float,
                        help="Spatial eigenvalue (controls pull toward data)")
    parser.add_argument("--lambda_tau", default=1.0, type=float,
                        help="Pseudo-time eigenvalue (controls τ convergence)")
    parser.add_argument("--num_sample_steps", default=100, type=int,
                        help="ODE integration steps during sampling")
    parser.add_argument("--loss_type", default="l2", choices=["l1", "l2"],
                        help="Loss function type")
    args = parser.parse_args()

    print(f"Config: λ_z={args.lambda_z}, λ_τ={args.lambda_tau}, "
          f"ratio={args.lambda_z / args.lambda_tau:.2f}")
    if args.lambda_z == args.lambda_tau:
        print("  → ratio=1.0 recovers OT-FM (Corollary 4.12)")

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=CHANNELS).to(DEVICE)

    flow = StableFlowMatching(
        model,
        lc_size=LC_SIZE,
        channels=CHANNELS,
        lambda_z=args.lambda_z,
        lambda_tau=args.lambda_tau,
        num_sample_steps=args.num_sample_steps,
        loss_type=args.loss_type,
    ).to(DEVICE)

    sequences = load_sequences(DATA_DIR, LC_SIZE, max_files=100)
    dataset = LightcurveDataset(sequences)

    trainer = Trainer(
        flow,
        dataset,
        logdir="./logs/",
        lc_size=LC_SIZE,
        train_batch_size=56,
        train_lr=2e-5,
        train_num_steps=750001,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        num_workers=32,
        rank=[0, 1, 2],
    )

    if args.milestone != 0:
        trainer.load(args.milestone)

    trainer.train()


if __name__ == "__main__":
    main()
