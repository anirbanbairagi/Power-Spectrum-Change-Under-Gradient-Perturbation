#!/bin/bash
#SBATCH --account bdne-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=51-100
#SBATCH --time=00:50:00
#SBATCH -J quijote_delta
#SBATCH -o /u/nchartier/scratch/quijote_delta_%A_%a.out
#SBATCH -e /u/nchartier/scratch/quijote_delta_%A_%a.out

module load PrgEnv-gnu/8.6.0 cray-fftw/3.3.10.10 gsl/2.8-gcc13.3.1
export LD_LIBRARY_PATH="${GSL_HOME}/lib:${LD_LIBRARY_PATH}"
source /u/nchartier/Modules/miniconda3/bin/activate pkcorr

python /u/nchartier/PkCorr/compute_quijote_delta.py --seed "$SLURM_ARRAY_TASK_ID"
