#!/bin/bash
#SBATCH --account bdne-delta-cpu    # allocation name
#SBATCH --partition=cpu
#SBATCH --nodes=1       # Total # of nodes (must be 1 for OpenMP jobs)
#SBATCH --ntasks-per-node=1     # Total # of MPI tasks per node
#SBATCH --cpus-per-task=16      # cpu-cores per task (default value is 1, >1 for multi-threaded tasks)
#SBATCH --mem=64G           # cpu memory per node, "mem-per-cpu" for memory per cpu-core
#SBATCH --time=03:30:00        # Total run time limit (hh:mm:ss)
#SBATCH -J pkcorr_notebook          # Job name
#SBATCH -o /u/nchartier/scratch/myjob.o%j          # Name of stdout output file
#SBATCH -e /u/nchartier/scratch/myjob.e%j          # Name of stderr error file

# get tunneling info
export port=8283 # CAREFUL: same port as in jupyter_notebook_config.py file
node=$(hostname -s)
user=$(whoami)

module purge
source /u/nchartier/.bashrc
conda activate pkcorr # previously cmass_env with out of data ltu-ili
echo "CONDA ENVIRONMENT LOADED"

#cd /home/x-nchartier/LTU_ILI
cd /u/nchartier

# run jupyter notebook
#jupyter-notebook --no-browser --port=${port} --ip=${node}
jupyter-lab --no-browser --port=${port} --ip=${node}

