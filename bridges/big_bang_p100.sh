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
./big_bang1.exec big_bangp1.anim 1.0
./big_bang2.exec big_bangp2.anim 1.0
./big_bang3.exec big_bangp3.anim 1.0
./big_bang4.exec big_bangp4.anim 1.0
./big_bang5.exec big_bangp5.anim 1.0
./big_bang6.exec big_bangp6.anim 1.0
./big_bang7.exec big_bangp7.anim 1.0
./big_bang8.exec big_bangp8.anim 1.0