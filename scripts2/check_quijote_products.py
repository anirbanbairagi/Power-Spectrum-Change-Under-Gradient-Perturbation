import os

QUIJOTE_ROOT = "/work/hdd/bdne/nchartier/quijote_lh"
start, end = 450, 500  # set per batch

expected = ["k.npy", "pk.npy", "delta_grid256.npy", "delta_grid512.npy", "bk.npy", "ks_bispectrum.npy"]
bad_seeds = []
for i in range(start, end + 1):
    d = os.path.join(QUIJOTE_ROOT, str(i))
    missing = [f for f in expected if not os.path.exists(os.path.join(d, f))]
    if missing:
        print(f"seed {i}: missing {missing}")
        bad_seeds.append(i)

print("bad seeds:", bad_seeds)
