#!/bin/bash
# usage: delete_quijote_ics.sh <start> <end>
START=${1:?usage: delete_quijote_ics.sh start end}
END=${2:?usage: delete_quijote_ics.sh start end}
QUIJOTE_ROOT=/work/hdd/bdne/nchartier/quijote_lh

for ((k=START; k<=END; k++)); do
    d="$QUIJOTE_ROOT/$k"
    if ls "$d"/ics.*.hdf5 >/dev/null 2>&1; then
        echo "seed $k: deleting ics.*.hdf5"
        rm -f "$d"/ics.*.hdf5
    fi
done
