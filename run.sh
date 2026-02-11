#!/bin/bash
#SBATCH --account=galacticbulge
#SBATCH --job-name=stable_fm
#SBATCH --partition=mb-gpu
#SBATCH --gres=gpu:4          # Request 4 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclusive           # Gives you ALL memory and CPUs automatically
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
