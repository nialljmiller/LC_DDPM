import argparse
import time
import numpy as np
import torch
from pathlib import Path

from stable_flow_matching import (
    Unet, StableFlowMatching, Trainer, LightcurveDataset,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "probes": {"logdir": "./logs/probes/"},
    "sdss":   {"logdir": "./logs/sdss/"},
}


def main():
    parser = argparse.ArgumentParser("Stable Flow Matching — Inference")
    parser.add_argument("--dataset", default="probes", choices=list(CONFIGS.keys()))
    parser.add_argument("--milestone", default=750000, type=int)
    parser.add_argument("--batches", default=105, type=int)
    parser.add_argument("--batch_size", default=96, type=int)
    parser.add_argument("--num_sample_steps", default=100, type=int)
    parser.add_argument("--lambda_z", default=2.0, type=float)
    parser.add_argument("--lambda_tau", default=1.0, type=float)
    parser.add_argument("--use_midpoint", action="store_true",
                        help="Use midpoint (RK2) integrator for better quality")
    parser.add_argument("--outdir", default="inferred/", type=str)
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(DEVICE)

    flow = StableFlowMatching(
        model,
        lc_size=256,
        lambda_z=args.lambda_z,
        lambda_tau=args.lambda_tau,
        num_sample_steps=args.num_sample_steps,
    ).to(DEVICE)

    # Trainer is only used here to load the checkpoint — pass a dummy dataset
    dummy_dataset = LightcurveDataset([np.zeros((3, 256, 256))])
    trainer = Trainer(
        flow,
        dummy_dataset,
        logdir=cfg["logdir"],
        lc_size=256,
        train_batch_size=16,
        train_lr=2e-5,
        train_num_steps=750001,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        num_workers=0,
        rank=[0],
    )

    trainer.load(args.milestone)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_fn = flow.sample_midpoint if args.use_midpoint else flow.sample

    i = 0
    for _ in range(args.batches):
        sampled_batch = sample_fn(batch_size=args.batch_size)

        for sample in sampled_batch.detach().cpu().numpy():
            np.save(outdir / f"{int(time.time())}_{i:05d}.npy", sample)
            i += 1

    print(f"Generated {i} samples in {outdir}")


if __name__ == "__main__":
    main()
