import os, time, json, argparse, warnings
import numpy as np
from scipy.stats import chi2 as chi2_dist
from pqm import pqm_chi2


def load_field(qroot, quijote_root, sim, seed, which, grid):
    if which == "nb":
        return np.load(os.path.join(quijote_root, str(seed), "delta_grid%d.npy" % grid))
    if which == "pm":
        return np.load(os.path.join(qroot, sim, "seed%d" % seed, "delta_pm_grid%d.npy" % grid))
    it = int(which)
    return np.load(os.path.join(qroot, sim, "seed%d" % seed,
                                 "delta_corrected_it%d_grid%d.npy" % (it, grid)))


def split_into_octants(field):
    g = field.shape[0]
    h = g // 2
    return [field[i:i+h, j:j+h, k:k+h]
            for i in (0, h) for j in (0, h) for k in (0, h)]


def build_samples(fields, case):
    if case == "raw":
        return np.stack([f.reshape(-1) for f in fields])
    elif case == "octant":
        vecs = [oct_.reshape(-1) for f in fields for oct_ in split_into_octants(f)]
        return np.stack(vecs)
    raise ValueError("case must be 'raw' or 'octant', got %r" % case)


def run_test(qroot, quijote_root, sim, which_a, which_b, grid, seeds, case,
             num_refs, re_tessellation, outroot):
    start = time.time()
    fields_a = [load_field(qroot, quijote_root, sim, s, which_a, grid) for s in seeds]
    fields_b = [load_field(qroot, quijote_root, sim, s, which_b, grid) for s in seeds]
    print(f"loaded {len(fields_a)}+{len(fields_b)} fields in {time.time()-start:.1f}s")

    x = build_samples(fields_a, case)
    y = build_samples(fields_b, case)
    print(f"x: {x.shape}, y: {y.shape}")

    result = dict(sim=sim, which_a=which_a, which_b=which_b, grid=grid, case=case,
                  num_refs=num_refs, re_tessellation=re_tessellation,
                  n_seeds=len(seeds), x_shape=list(x.shape), y_shape=list(y.shape))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            t0 = time.time()
            chi2_list = pqm_chi2(x, y, num_refs=num_refs, re_tessellation=re_tessellation)
            chi2_arr = np.asarray(chi2_list)
            pval_arr = chi2_dist.sf(chi2_arr, num_refs - 1)
            result.update(chi2_mean=float(chi2_arr.mean()), chi2_std=float(chi2_arr.std()),
                           pval_mean=float(pval_arr.mean()), pval_std=float(pval_arr.std()),
                           chi2_list=chi2_arr.tolist(), pval_list=pval_arr.tolist(),
                           failed=False, runtime_s=time.time() - t0)
        except ValueError as e:
            result.update(failed=True, error=str(e))
        result["warnings"] = [str(warning.message) for warning in w]

    os.makedirs(outroot, exist_ok=True)
    outname = f"pqm_{case}_{sim}_{which_a}_vs_{which_b}_grid{grid}.json"
    with open(os.path.join(outroot, outname), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["picola", "fastpm"], required=True)
    parser.add_argument("--which_a", default="nb")
    parser.add_argument("--which_b", required=True)   # "pm" or an iteration number, e.g. "3"
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--case", choices=["raw", "octant"], required=True)
    parser.add_argument("--num_refs", type=int, default=50)
    parser.add_argument("--re_tessellation", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--qroot", default="/work/hdd/bdne/nchartier/Q_correction")
    parser.add_argument("--quijote_root", default="/work/hdd/bdne/nchartier/quijote_lh")
    parser.add_argument("--outroot", default="/work/hdd/bdne/nchartier/PQMass_results")
    args = parser.parse_args()

    run_test(args.qroot, args.quijote_root, args.sim, args.which_a, args.which_b,
              args.grid, list(range(args.n_seeds)), args.case,
              args.num_refs, args.re_tessellation, args.outroot)
