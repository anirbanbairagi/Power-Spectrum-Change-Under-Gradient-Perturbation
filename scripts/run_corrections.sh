#!/bin/bash
#SBATCH --account bdne-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --array=0-49
#SBATCH --time=03:00:00
#SBATCH -J qcorr
#SBATCH -o /u/nchartier/scratch/qcorr_%x_%A_%a.out
#SBATCH -e /u/nchartier/scratch/qcorr_%x_%A_%a.out

module load PrgEnv-gnu/8.6.0 cray-fftw/3.3.10.10 gsl/2.8-gcc13.3.1
export LD_LIBRARY_PATH="${GSL_HOME}/lib:${LD_LIBRARY_PATH}"
source /u/nchartier/Modules/miniconda3/bin/activate pkcorr

SIM=${1:?usage: sbatch run_qcorr.sh picola|fastpm}
python /u/nchartier/PkCorr/run_Q_corrections.py --sim "$SIM" --seed "$SLURM_ARRAY_TASK_ID"
