#!/bin/sh -f
#PBS -N StableFM
#PBS -m abe
#PBS -q gpu

# Stable Flow Matching for light curves
# --lambda_z / --lambda_tau controls interpolation rate (Figure 3 of paper)
#   ratio=1.0 → OT-FM (Corollary 4.12)
#   ratio>1.0 → sharper convergence, better stability

python3 /beegfs/car/njm/DDPM/train.py \
    --milestone 0 \
    --data_dir /media/bigdata/PRIMVS/light_curves/ \
    --band Ks \
    --lambda_z 2.0 \
    --lambda_tau 1.0 \
    --loss_type l2
