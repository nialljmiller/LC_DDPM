#!/bin/bash
#SBATCH --account=galacticbulge
#SBATCH --job-name=stable_fm
#SBATCH --partition=mb-gpu
#SBATCH --gres=gpu:4          # Request ALL 4 GPUs on the node
#SBATCH --nodes=1             # DataParallel cannot handle >1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Give the GPUs plenty of CPU power
#SBATCH --mem=0               # Request ALL available RAM on the node
#SBATCH --time=24:00:00
#SBATCH --exclusive           # Lock the node so only YOU use it

zenv
# Point to your specific source list to avoid the "scanning billions of files" hang
# You MUST have this fits file, or the job will never start.
python train.py \
    --milestone 0 \
    --data_dir "/project/<YOUR_PROJECT>/PRIMVS/light_curves/" \
    --fits_file "my_source_list.fits" \
    --fits_id_column "sourceid" \
    --band Ks \
    --lambda_z 2.0 \
    --lambda_tau 1.0 \
    --loss_type l2
