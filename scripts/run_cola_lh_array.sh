#!/bin/bash
#SBATCH --account bdne-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --array=6,7,13,14,17,18,23,26,30,33,35,37,44,47,48
#SBATCH --time=10:00:00
#SBATCH -J lpicola_lh
#SBATCH -o /u/nchartier/scratch/lpicola_lh_%A_%a.out
#SBATCH -e /u/nchartier/scratch/lpicola_lh_%A_%a.out

module load PrgEnv-gnu/8.6.0 cray-fftw/3.3.10.10 gsl/2.8-gcc13.3.1
export LD_LIBRARY_PATH="${GSL_HOME}/lib:${LD_LIBRARY_PATH}"

EXEC=/u/nchartier/Modules/l-picola/L-PICOLA
ROOTDIR=/u/nchartier/PkCorr/COLA_source

k=$SLURM_ARRAY_TASK_ID
paramfile=${ROOTDIR}/${k}/Cola${k}_512_20steps_nLPTpos0p5.dat
outdir=$(awk '$1=="OutputDir"{print $2}' "$paramfile")
filebase=$(awk '$1=="FileBase"{print $2}' "$paramfile")

if ls "${outdir}${filebase}_z0p500."* >/dev/null 2>&1; then
    echo "Realization $k already has a z=0.5 snapshot, skipping"
    exit 0
fi

echo "=== Realization $k: $paramfile ==="
srun -n 32 "$EXEC" "$paramfile"

rm -f "${outdir}${filebase}_z127p000."* "${outdir}${filebase}_z1p000."* "${outdir}${filebase}_z0p000."*
