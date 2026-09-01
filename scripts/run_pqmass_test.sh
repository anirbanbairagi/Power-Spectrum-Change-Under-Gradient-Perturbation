#!/bin/bash
#SBATCH --account bdne-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH -J pqmass
#SBATCH -o /u/nchartier/scratch/pqmass_%x_%A.out
#SBATCH -e /u/nchartier/scratch/pqmass_%x_%A.out

source /u/nchartier/Modules/miniconda3/bin/activate pkcorr

SIM=${1:?usage: sbatch run_pqmass_test.sh <sim> <which_b> <case> [num_refs]}
WHICH_B=${2:?usage: sbatch run_pqmass_test.sh <sim> <which_b> <case> [num_refs]}
CASE=${3:?usage: sbatch run_pqmass_test.sh <sim> <which_b> <case> [num_refs]}
NUM_REFS=${4:-50}


# eg sbatch run_pqmass_test.sh picola 3 octant for iteration 3 and octant split
python /u/nchartier/PkCorr/run_pqmass_test.py --sim "$SIM" --which_b "$WHICH_B" \
    --case "$CASE" --num_refs "$NUM_REFS"
