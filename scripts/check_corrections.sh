#!/bin/bash

OUTROOT=/work/hdd/bdne/nchartier/Q_correction
NREAL=50
N_ITER=3

for sim in picola fastpm; do
    echo "=== $sim ==="
    incomplete=""
    for ((k=0; k<NREAL; k++)); do
        d="${OUTROOT}/${sim}/seed${k}"
        missing=""

        for f in pk_nb.npy pk_pm.npy \
                 delta_nb_grid256.npy delta_nb_grid512.npy \
                 delta_pm_grid256.npy delta_pm_grid512.npy \
                 bk_nb.npy bk_pm.npy; do
            [ -f "${d}/${f}" ] || missing="${missing} ${f}"
        done

        for ((it=1; it<=N_ITER; it++)); do
            for f in "pk_corrected_it${it}.npy" \
                     "delta_corrected_it${it}_grid256.npy" \
                     "delta_corrected_it${it}_grid512.npy" \
                     "bk_corrected_it${it}.npy"; do
                [ -f "${d}/${f}" ] || missing="${missing} ${f}"
            done
        done

        if [ ! -d "$d" ]; then
            echo "seed ${k}: MISSING ENTIRE OUTPUT DIRECTORY"
            incomplete="${incomplete} ${k}"
        elif [ -n "$missing" ]; then
            echo "seed ${k}: missing -${missing}"
            incomplete="${incomplete} ${k}"
        fi
    done

    if [ -z "$incomplete" ]; then
        echo "All $NREAL seeds complete for $sim"
    else
        echo "Incomplete seeds for $sim:${incomplete}"
    fi
    echo
done
