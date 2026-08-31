#!/bin/bash
# usage: check_quijote_snapshots.sh [start] [end]
QUIJOTE_ROOT=/work/hdd/bdne/nchartier/quijote_lh
START=${1:-0}
END=${2:-100}

incomplete=""
for ((k=START; k<=END; k++)); do
    d="$QUIJOTE_ROOT/$k"
    if [ ! -d "$d" ]; then
        echo "seed $k: MISSING DIRECTORY"
        incomplete="$incomplete $k"
        continue
    fi

    missing=""
    for ((i=0; i<8; i++)); do
        f="$d/snap_003.$i.hdf5"
        if [ ! -s "$f" ]; then
            missing="$missing snap_003.$i.hdf5"
        fi
    done

    if [ -n "$missing" ]; then
        echo "seed $k: missing/empty -$missing"
        incomplete="$incomplete $k"
    fi
done

echo
if [ -z "$incomplete" ]; then
    echo "All seeds from $START to $END have all 8 snapshot files"
else
    echo "Incomplete seeds:$incomplete"
fi
