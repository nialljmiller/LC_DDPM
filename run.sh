#!/bin/bash
#SBATCH --account=galacticbulge
#SBATCH --job-name=stable_fm
#SBATCH --partition=mb-h100   # CORRECTED: Use 'mb-h100', 'mb-l40s', or 'mb-a30'
#SBATCH --gres=gpu:8          # MAX CAPACITY: These nodes have 8 GPUs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48    # H100 nodes have 96 cores (48 per socket). 
#SBATCH --time=24:00:00
#SBATCH --exclusive           # Gives you the full node resources.
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

zenv

# Run the code
# REMINDER: Ensure train.py has "rank=list(range(torch.cuda.device_count())),"
python train.py \
    --milestone 0 \
    --data_dir "/project/galacticbulge/PRIMVS/light_curves/" \
    --fits_file "my_source_list.fits" \
    --fits_id_column "sourceid" \
    --band Ks \
    --lambda_z 2.0 \
    --lambda_tau 1.0 \
    --loss_type l2
