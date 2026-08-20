import os
import argparse
import numpy as np
import readgadget
from pypower import CatalogMesh


def load_quijote(seed):
    path = "/work/hdd/bdne/nchartier/quijote_lh/%d/snap_003" % seed
    header = readgadget.header(path)
    BoxSize = header.boxsize / 1e3
    pos = readgadget.read_block(path, "POS ", [1]) / 1e3
    return pos, BoxSize


def compute_delta(positions, BoxSize, grid):
    mesh = CatalogMesh(data_positions=positions, boxsize=BoxSize, nmesh=grid,
                        resampler='cic', interlacing=0, position_type='pos', dtype='f4')
    delta = np.array(mesh.to_mesh(compensate=False))
    return (delta / delta.mean() - 1.0).astype('float32')


def run_seed(seed, grids=(256, 512)):
    outdir = "/work/hdd/bdne/nchartier/quijote_lh/%d" % seed
    done_marker = os.path.join(outdir, "delta_grid%d.npy" % grids[-1])
    if os.path.exists(done_marker):
        print(f"[seed {seed}] already complete, skipping")
        return

    pos, BoxSize = load_quijote(seed)
    for grid in grids:
        delta = compute_delta(pos, BoxSize, grid)
        np.save(os.path.join(outdir, "delta_grid%d.npy" % grid), delta)
        print(f"[seed {seed}] saved delta_grid{grid}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_seed(args.seed)
