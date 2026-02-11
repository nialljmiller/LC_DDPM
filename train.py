import argparse
import torch

from stable_flow_matching import (
    Unet, StableFlowMatching, Trainer, LightcurveDataset,
    load_sequences_from_primvs,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Config ----------------------------------------------------------------
SPATIAL_SIZE = 128   # H and W of the 2D representation (must be divisible by 16)
CHANNELS = 3         # mag, err, phase
LC_SIZE = 70         # minimum number of datapoints per lightcurve


# ---- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Stable Flow Matching — Light Curve Training")
    parser.add_argument("--milestone", default=0, type=int, help="Checkpoint step to resume from")
    parser.add_argument("--data_dir", default="/media/bigdata/PRIMVS/light_curves/", type=str,
                        help="PRIMVS data directory")
    parser.add_argument("--fits_file", default=None, type=str,
                        help="FITS file with source IDs (optional; if omitted, discovers all CSVs in data_dir)")
    parser.add_argument("--fits_id_column", default="sourceid", type=str,
                        help="Column name for source IDs in the FITS file")
    parser.add_argument("--band", default="Ks", type=str, help="Photometric band to use")
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

    # ---- Resolve source IDs ------------------------------------------------
    if args.fits_file is not None:
        from astropy.table import Table
        tbl = Table.read(args.fits_file, hdu=1)
        source_ids = tbl[args.fits_id_column].data
        print(f"Read {len(source_ids)} source IDs from {args.fits_file}")
    else:
        from pathlib import Path
        csv_paths = list(Path(args.data_dir).glob("**/*.csv"))
        source_ids = [int(p.stem) for p in csv_paths]
        print(f"Discovered {len(source_ids)} sources in {args.data_dir}")

    # ---- Load data via PRIMVS API ------------------------------------------
    sequences = load_sequences_from_primvs(
        source_ids, args.data_dir, LC_SIZE,
        spatial_size=SPATIAL_SIZE, band=args.band,
    )
    dataset = LightcurveDataset(sequences)

    # ---- Model -------------------------------------------------------------
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=CHANNELS).to(DEVICE)

    flow = StableFlowMatching(
        model,
        spatial_size=SPATIAL_SIZE,
        channels=CHANNELS,
        lambda_z=args.lambda_z,
        lambda_tau=args.lambda_tau,
        num_sample_steps=args.num_sample_steps,
        loss_type=args.loss_type,
    ).to(DEVICE)

    # ---- Train -------------------------------------------------------------
    trainer = Trainer(
        flow,
        dataset,
        logdir="./logs/",
        spatial_size=SPATIAL_SIZE,
        train_batch_size=56,
        train_lr=2e-5,
        train_num_steps=750001,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        num_workers=32,
        rank=list(range(torch.cuda.device_count())),
    )

    if args.milestone != 0:
        trainer.load(args.milestone)

    trainer.train()


if __name__ == "__main__":
    main()
