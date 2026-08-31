#!/bin/bash
START=${1:?usage: delete_raw_quijote_batch.sh start end}
END=${2:?usage: delete_raw_quijote_batch.sh start end}
QUIJOTE_ROOT=/work/hdd/bdne/nchartier/quijote_lh

for ((k=START; k<=END; k++)); do
    if [ ! -f "$QUIJOTE_ROOT/$k/bk.npy" ]; then
        echo "seed $k: bk.npy missing, skipping (derived products incomplete)"
        continue
    fi
    echo "seed $k: deleting raw snapshot"
    rm -f "$QUIJOTE_ROOT/$k"/snap_003*
done
