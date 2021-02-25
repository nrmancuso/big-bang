#!/bin/bash

# A P100 node on the GPU-small partition has: 2 GPUs, 2 16-core CPUs,
# 8 TB on-node storage.  In this configuration, 1 node is requested.
# 
# 16 OpenMP threads, 2 GPU's
#SBATCH -p GPU-small
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:p100:2
# echo commands to stdout
set -x
#./big_bang8.exec big_bang8.anim 0.50
#./big_bang8.exec big_bang8.anim 0.55
#./big_bang8.exec big_bang8.anim 0.60
#./big_bang8.exec big_bang8.anim 0.65
#./big_bang8.exec big_bang8.anim 0.70
#./big_bang8.exec big_bang8.anim 0.75
#./big_bang8.exec big_bang8.anim 0.80
#./big_bang8.exec big_bang8.anim 0.85
./big_bang8.exec big_bang8.anim 0.90
./big_bang8.exec big_bang8.anim 0.95
./big_bang8.exec big_bang8.anim 1.0