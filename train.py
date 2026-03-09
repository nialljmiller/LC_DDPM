import argparse
import torch

from stable_flow_matching import (
    Unet, StableFlowMatching, Trainer, LazyPrimvsDataset,
    DEFAULT_COND_FEATURES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Config ----------------------------------------------------------------
SPATIAL_SIZE = 128   # H and W of the 2D representation (must be divisible by 16)
CHANNELS     = 3     # mag, err, phase
LC_SIZE      = 70    # minimum number of datapoints per light curve


# ---- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Stable Flow Matching — Light Curve Training")
    parser.add_argument("--milestone",        default=0,    type=int)
    parser.add_argument("--data_dir",         default="/media/bigdata/PRIMVS/light_curves/", type=str)
    parser.add_argument("--fits_file",        default=None, type=str,
                        help="FITS file with source IDs (if omitted, discovers all CSVs in data_dir)")
    parser.add_argument("--fits_id_column",   default="sourceid", type=str)
    parser.add_argument("--catalog_fits",     default=None, type=str,
                        help="PRIMVS catalog FITS file containing variability features "
                             "used for conditioning (e.g. PRIMVS_P.fits). "
                             "If omitted the model trains unconditionally.")
    parser.add_argument("--cond_features",    default=None, nargs="+",
                        help="Catalog column names to use as conditioning features. "
                             "Defaults to DEFAULT_COND_FEATURES when --catalog_fits is set.")
    parser.add_argument("--band",             default="Ks",  type=str)
    parser.add_argument("--lambda_z",         default=2.0,   type=float)
    parser.add_argument("--lambda_tau",       default=1.0,   type=float)
    parser.add_argument("--num_sample_steps", default=100,   type=int)
    parser.add_argument("--loss_type",        default="l2",  choices=["l1", "l2"])
    args = parser.parse_args()

    print(f"Config: λ_z={args.lambda_z}, λ_τ={args.lambda_tau}, "
          f"ratio={args.lambda_z / args.lambda_tau:.2f}")
    if args.lambda_z == args.lambda_tau:
        print("  → ratio=1.0 recovers OT-FM (Corollary 4.12)")

    # ---- Resolve source IDs ------------------------------------------------
    if args.fits_file is not None:
        from astropy.table import Table
        tbl        = Table.read(args.fits_file, hdu=1)
        source_ids = tbl[args.fits_id_column].data
        print(f"Read {len(source_ids)} source IDs from {args.fits_file}")
    else:
        from pathlib import Path
        csv_paths  = list(Path(args.data_dir).glob("**/*.csv"))
        source_ids = [int(p.stem) for p in csv_paths]
        print(f"Discovered {len(source_ids)} sources in {args.data_dir}")

    # ---- Conditioning feature names ----------------------------------------
    cond_feature_names = None
    if args.catalog_fits:
        cond_feature_names = args.cond_features or DEFAULT_COND_FEATURES
        print(f"Conditioning on {len(cond_feature_names)} features: {cond_feature_names}")
    else:
        print("No catalog FITS supplied — training unconditionally.")

    num_cond = len(cond_feature_names) if cond_feature_names else 0

    # ---- Dataset -----------------------------------------------------------
    dataset = LazyPrimvsDataset(
        source_ids,
        data_dir=args.data_dir,
        lc_size=LC_SIZE,
        spatial_size=SPATIAL_SIZE,
        band=args.band,
        catalog_fits=args.catalog_fits,
        cond_feature_names=cond_feature_names,
    )

    # ---- Model -------------------------------------------------------------
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
        loss_type=args.loss_type,
    ).to(DEVICE)

    # ---- Train -------------------------------------------------------------
    trainer = Trainer(
        flow,
        dataset,
        logdir="./logs/",
        spatial_size=SPATIAL_SIZE,
        train_batch_size=512,
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
