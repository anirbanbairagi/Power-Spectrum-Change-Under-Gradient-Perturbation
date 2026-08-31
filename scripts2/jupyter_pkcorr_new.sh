#!/bin/bash
#SBATCH --account bdne-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:30:00
#SBATCH -J pkcorr_nb
#SBATCH -o /u/nchartier/scratch/myjob.o%j
#SBATCH -e /u/nchartier/scratch/myjob.e%j

# NOT NEEDED: module by default of cray system. I only need to use source .. activate for conda
# instead of conda activate pkcorr  since miniforge messes up the conda env list, and I compiled mpi4py
# against the default module list
#module purge
#module load <PrgEnv-xxx / cray-mpich modules from `module list`>
#module load miniforge3-python

export port=8283
node=$(hostname -s)
user=$(whoami)

source /u/nchartier/Modules/miniconda3/bin/activate pkcorr
echo "CONDA ENVIRONMENT LOADED"

cd /u/nchartier
jupyter-lab --no-browser --port=${port} --ip=${node}
