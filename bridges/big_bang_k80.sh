#!/bin/bash
# A k80 node on the GPU partition has: 4 GPUs, 2 14-core CPUs,
# 8 TB on-node storage.  In this configuration, 1 node is requested.
# 
# 14 OpenMP threads, 4 GPU's
#SBATCH -p GPU
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:k80:4
# echo commands to stdout
set -x
./big_bang1.exec big_bangk1.anim 1.0
./big_bang2.exec big_bangk2.anim 1.0
./big_bang3.exec big_bangk3.anim 1.0
./big_bang4.exec big_bangk4.anim 1.0
./big_bang5.exec big_bangk5.anim 1.0
./big_bang6.exec big_bangk6.anim 1.0
./big_bang7.exec big_bangk7.anim 1.0
./big_bang8.exec big_bangk8.anim 1.0