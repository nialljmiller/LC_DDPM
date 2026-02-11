#!/bin/sh -f
#PBS -N DDPM
#PBS -m abe
#PBS -q gpu

python3 /beegfs/car/njm/DDPM/train.py --dataset LC --milestone 177000

