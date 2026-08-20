import os, sys, time, argparse
import numpy as np
import pyfftw
from numba import njit, prange
from scipy.interpolate import interp1d
import readgadget
from pypower import CatalogFFTPower, CatalogMesh
import PolyBin3D as pb

_trapz = getattr(np, "trapz", None) or np.trapezoid

# ---------------------------------------------------------------------------
# Reused from the notebook: numba interpolation / displacement helpers
# ---------------------------------------------------------------------------

@njit
def trilinear_interp_numba(data, points, x_coords, y_coords, z_coords):
    nx, ny, nz = data.shape
    num_points = points.shape[0]
    interpolated_values = np.empty(num_points, dtype=data.dtype)
    dx_inv = 1.0 / (x_coords[1] - x_coords[0])
    dy_inv = 1.0 / (y_coords[1] - y_coords[0])
    dz_inv = 1.0 / (z_coords[1] - z_coords[0])
    for i in prange(num_points):
        x_query, y_query, z_query = points[i, :]
        x0_idx = np.searchsorted(x_coords, x_query, side="left") - 1
        y0_idx = np.searchsorted(y_coords, y_query, side="left") - 1
        z0_idx = np.searchsorted(z_coords, z_query, side="left") - 1
        x0_idx, y0_idx, z0_idx = x0_idx % nx, y0_idx % ny, z0_idx % nz
        x1_idx, y1_idx, z1_idx = (x0_idx + 1) % nx, (y0_idx + 1) % ny, (z0_idx + 1) % nz
        x0, y0, z0 = x_coords[x0_idx], y_coords[y0_idx], z_coords[z0_idx]
        xd = (x_query - x0) * dx_inv
        yd = (y_query - y0) * dy_inv
        zd = (z_query - z0) * dz_inv
        c000 = data[x0_idx, y0_idx, z0_idx]; c100 = data[x1_idx, y0_idx, z0_idx]
        c010 = data[x0_idx, y1_idx, z0_idx]; c001 = data[x0_idx, y0_idx, z1_idx]
        c110 = data[x1_idx, y1_idx, z0_idx]; c101 = data[x1_idx, y0_idx, z1_idx]
        c011 = data[x0_idx, y1_idx, z1_idx]; c111 = data[x1_idx, y1_idx, z1_idx]
        interpolated_values[i] = (
            c000*(1-xd)*(1-yd)*(1-zd) + c100*xd*(1-yd)*(1-zd) +
            c010*(1-xd)*yd*(1-zd) + c001*(1-xd)*(1-yd)*zd +
            c110*xd*yd*(1-zd) + c101*xd*(1-yd)*zd +
            c011*(1-xd)*yd*zd + c111*xd*yd*zd
        )
    return interpolated_values


@njit
def displace_positions_interp(positions, new_positions, grad_x, grad_y, grad_z,
                                x_c, y_c, z_c, max_length, threshold=None):
    nabla_x = trilinear_interp_numba(grad_x, positions, x_c, y_c, z_c)
    nabla_y = trilinear_interp_numba(grad_y, positions, x_c, y_c, z_c)
    nabla_z = trilinear_interp_numba(grad_z, positions, x_c, y_c, z_c)
    if threshold is not None:
        for i in prange(nabla_x.shape[0]):
            mag = np.sqrt(nabla_x[i]**2 + nabla_y[i]**2 + nabla_z[i]**2)
            if mag > threshold:
                scale = threshold / mag
                nabla_x[i] *= scale
                nabla_y[i] *= scale
                nabla_z[i] *= scale
    new_positions[:, 0] = np.mod(positions[:, 0] + nabla_x, max_length)
    new_positions[:, 1] = np.mod(positions[:, 1] + nabla_y, max_length)
    new_positions[:, 2] = np.mod(positions[:, 2] + nabla_z, max_length)
    return new_positions, nabla_x, nabla_y, nabla_z


def fourier_shells_k(res, box_size, fshift=True):
    scale = 2*np.pi*res/box_size
    if fshift:
        k_x = np.array(scale*np.fft.fftshift(np.fft.fftfreq(res)), dtype='float32')
    else:
        k_x = np.array(scale*np.fft.fftfreq(res), dtype='float32')
    k_y, k_z = k_x.copy(), k_x.copy()
    kx, ky, kz = np.meshgrid(k_x, k_y, k_z, indexing='ij')
    k_radius_grid = np.sqrt(kx**2 + ky**2 + kz**2)
    return k_x, k_y, k_z, k_radius_grid


@njit
def fill_fourier_grid(weights_grid, k3D_radius, k_bins, weights1D, zero_above_kmax=True):
    kmax = k_bins[-1]
    n_zeroed = 0
    for ix in prange(k3D_radius.shape[0]):
        for iy in prange(k3D_radius.shape[1]):
            for iz in prange(k3D_radius.shape[2]):
                k_val = k3D_radius[ix, iy, iz]
                if k_val > kmax:
                    if zero_above_kmax:
                        weights_grid[ix, iy, iz] = 0.0
                        n_zeroed += 1
                    else:
                        weights_grid[ix, iy, iz] = weights1D[-1]
                else:
                    weights_grid[ix, iy, iz] = np.interp(k_val, k_bins, weights1D)
    return weights_grid, n_zeroed

# ---------------------------------------------------------------------------
# Reused: Q-matrix solve, phi/accel generation, iterative correction
# ---------------------------------------------------------------------------

def build_Q_matrix(k, Pk, n_y=201):
    n_k = len(k)
    P_interp = interp1d(k, Pk, kind="linear", bounds_error=False,
                         fill_value=(Pk[0], Pk[-1]))
    y = np.linspace(-1, 1, n_y)
    I_mat = np.zeros((n_k, n_k))
    for i, ki in enumerate(k):
        for j, qj in enumerate(k):
            kq = np.sqrt(np.clip(ki**2 + qj**2 - 2*ki*qj*y, 0, None))
            I_mat[i, j] = _trapz(y**2 * P_interp(kq), y)
    dk = np.empty(n_k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2
    dk[0] = (k[1] - k[0]) / 2
    dk[-1] = (k[-1] - k[-2]) / 2
    prefac = (k[:, None]**2 / (2*np.pi)**2) * (k[None, :]**4) * dk[None, :]
    N1 = prefac * I_mat
    N2 = prefac * (Pk[:, None] * 2. / 3.)
    N = np.diag(k**4) + N1 - N2
    return N, N1, N2


def get_phi_and_accel_Q(Q_sol, k, grid, BoxSize, seed=None, n_threads=1,
                          zero_above_kmax=True, verbose=True):
    kx_v, ky_v, kz_v, k_radius = fourier_shells_k(grid, BoxSize, fshift=True)
    prec_in, prec_out = "complex64", "complex64"
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(0., 1., size=(grid, grid, grid)).astype(prec_in)

    a_in = pyfftw.empty_aligned((grid, grid, grid), dtype=prec_in)
    a_out = pyfftw.empty_aligned((grid, grid, grid), dtype='complex64')
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0, 1, 2), flags=('FFTW_ESTIMATE',),
                              direction='FFTW_FORWARD', threads=n_threads)
    a_in[:] = white_noise
    white_noise_fourier_shift = np.fft.fftshift(fftw_plan(a_in, a_out), axes=(0, 1, 2))

    cell_volume = (BoxSize / grid)**3
    Q_safe = np.clip(Q_sol, 0., None)
    weights1D = np.sqrt(Q_safe / cell_volume)

    weights_grid = np.empty(white_noise_fourier_shift.shape)
    weights_grid, n_zeroed = fill_fourier_grid(weights_grid, k_radius, k, weights1D,
                                                 zero_above_kmax=zero_above_kmax)
    if verbose:
        frac = n_zeroed / k_radius.size
        print(f"  fill_fourier_grid: zero_above_kmax={zero_above_kmax} -> "
              f"{n_zeroed}/{k_radius.size} grid points ({frac:.1%}) beyond "
              f"k_max={k[-1]:.4f} h/Mpc set to zero weight.")

    phi_k_Q = weights_grid * white_noise_fourier_shift
    grad_x_fourier = 1.0j * kx_v[:, None, None] * phi_k_Q
    grad_y_fourier = 1.0j * ky_v[None, :, None] * phi_k_Q
    grad_z_fourier = 1.0j * kz_v[None, None, :] * phi_k_Q

    arr_in = pyfftw.empty_aligned((grid, grid, grid), dtype=prec_out)
    arr_out = pyfftw.empty_aligned((grid, grid, grid), dtype=prec_in)
    plan = pyfftw.FFTW(arr_in, arr_out, axes=(0, 1, 2), flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=n_threads)
    arr_in[:] = phi_k_Q.astype(prec_out)
    phi_r_Q = plan(np.fft.ifftshift(arr_in), arr_out).real

    accel = []
    for grad_fourier in (grad_x_fourier, grad_y_fourier, grad_z_fourier):
        arr_in = pyfftw.empty_aligned((grid, grid, grid), dtype=prec_out)
        arr_out = pyfftw.empty_aligned((grid, grid, grid), dtype=prec_in)
        plan = pyfftw.FFTW(arr_in, arr_out, axes=(0, 1, 2), flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=n_threads)
        arr_in[:] = grad_fourier.astype(prec_out)
        accel.append(plan(np.fft.ifftshift(arr_in), arr_out).real)

    return phi_k_Q, phi_r_Q, accel, n_zeroed


def correct_particles_with_Q(positions, BoxSize, grid, k_ref, Pk_ref,
                               measure_power_func, n_y=201, threshold=None, seed=None,
                               n_threads=None, zero_above_kmax=True,
                               remeasure=True, verbose=True,
                               k_before=None, Pk_before=None):
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    if k_before is None or Pk_before is None:
        k_before, Pk_before = measure_power_func(positions, BoxSize, grid)
    elif verbose:
        print("  reusing previous iteration's P(k) as this iteration's starting spectrum")

    Pk_ref_i = np.interp(k_before, k_ref, Pk_ref)
    N, N1, N2 = build_Q_matrix(k_before, Pk_before, n_y=n_y)
    Q_sol = np.linalg.solve(N, Pk_ref_i - Pk_before)
    Q_clipped = np.clip(Q_sol, 0., None)

    phi_k, phi_r, accel, n_zeroed = get_phi_and_accel_Q(
        Q_sol, k_before, grid, BoxSize, seed=seed, n_threads=n_threads,
        zero_above_kmax=zero_above_kmax, verbose=verbose)

    cell = BoxSize / grid
    length_grid = np.arange(0., BoxSize, cell)
    pos_corrected = np.empty(positions.shape, dtype="float32")
    pos_corrected, *_ = displace_positions_interp(
        positions.astype("float32"), pos_corrected,
        accel[0].astype("float32"), accel[1].astype("float32"), accel[2].astype("float32"),
        length_grid, length_grid, length_grid, BoxSize, threshold=threshold)

    out = dict(positions_corrected=pos_corrected, Q=Q_sol, Q_clipped=Q_clipped,
               k=k_before, Pk_before=Pk_before, n_zeroed_above_kmax=n_zeroed,
               phi_k=phi_k, phi_r=phi_r)
    if remeasure:
        k_after, Pk_after = measure_power_func(pos_corrected, BoxSize, grid)
        out['k_after'], out['Pk_after'] = k_after, Pk_after
    return out


def iterate_correction_with_Q(positions, BoxSize, grid, k_ref, Pk_ref, measure_power_func,
                                n_iterations, n_y=201, threshold=None, seed=None, n_threads=None,
                                zero_above_kmax=True, verbose=True):
    if seed is None:
        seeds = [None] * n_iterations
    else:
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31 - 1, size=n_iterations).tolist()

    history = []
    current_positions = positions
    k_before, Pk_before = None, None
    for it in range(n_iterations):
        if verbose:
            print(f"=== iteration {it + 1}/{n_iterations} ===")
        result = correct_particles_with_Q(
            current_positions, BoxSize, grid, k_ref, Pk_ref, measure_power_func,
            n_y=n_y, threshold=threshold, seed=seeds[it], n_threads=n_threads,
            zero_above_kmax=zero_above_kmax, remeasure=True, verbose=verbose,
            k_before=k_before, Pk_before=Pk_before,
        )
        history.append(result)
        current_positions = result['positions_corrected']
        k_before, Pk_before = result['k_after'], result['Pk_after']
    return dict(positions=current_positions, history=history)

# ---------------------------------------------------------------------------
# P(k) measurement (pypower)
# ---------------------------------------------------------------------------

def measure_power_pypower(positions, BoxSize, grid, kedges, resampler='cic',
                           interlacing=2, dtype='f4', ells=(0,),
                           remove_shotnoise=True, position_type='pos'):
    result = CatalogFFTPower(data_positions1=positions, boxsize=BoxSize, nmesh=grid,
                              resampler=resampler, edges=kedges, ells=ells,
                              interlacing=interlacing, position_type=position_type,
                              dtype=dtype)
    poles = result.poles
    k, Pk = poles(ell=ells[0], return_k=True, complex=False, remove_shotnoise=remove_shotnoise)
    mask = ~np.isnan(k) & ~np.isnan(Pk)
    return k[mask], Pk[mask]

# ---------------------------------------------------------------------------
# Diagnostics: density fields (CatalogMesh) and bispectra (PolyBin3D)
# ---------------------------------------------------------------------------

def compute_delta(positions, BoxSize, grid):
    mesh = CatalogMesh(data_positions=positions, boxsize=BoxSize, nmesh=grid,
                        resampler='cic', interlacing=0, position_type='pos', dtype='f4')
    delta = np.array(mesh.to_mesh(compensate=False))
    return (delta / delta.mean() - 1.0).astype('float32')

# ---------------------------------------------------------------------------
# Snapshot loaders
# ---------------------------------------------------------------------------

def load_reference(seed):
    path = "/work/hdd/bdne/nchartier/quijote_lh/%d/snap_003" % seed
    header = readgadget.header(path)
    BoxSize = header.boxsize / 1e3
    pos = readgadget.read_block(path, "POS ", [1]) / 1e3
    return pos, BoxSize


def load_pm(sim, seed):
    if sim == "fastpm":
        import bigfile
        path = "/work/hdd/bdne/maho3/cmass-ili/quijotelike_for_nicolas/fastpm/L1000-N128/%d/fastpm_B2_0.6667" % seed
        infile = bigfile.File(path)
        return infile['1/Position'][:]
    elif sim == "picola":
        path = "/work/hdd/bdne/nchartier/cola_lh/%d/Cola%d_512_20stepsnLPTpos0p5_z0p500" % (seed, seed)
        return readgadget.read_block(path, "POS ", [1]) / 1e3
    else:
        raise ValueError("sim must be 'fastpm' or 'picola', got %r" % sim)

# ---------------------------------------------------------------------------
# Per-realization pipeline
# ---------------------------------------------------------------------------

def run_realization(sim, seed, grid, n_iterations, n_threads, outroot):
    outdir = os.path.join(outroot, sim, "seed%d" % seed)
    os.makedirs(outdir, exist_ok=True)

    done_marker = os.path.join(outdir, "pk_corrected_it%d.npy" % n_iterations)
    if os.path.exists(done_marker):
        print(f"[seed {seed}] already complete, skipping")
        return

    pos_nb, BoxSize = load_reference(seed)
    pos_pm = load_pm(sim, seed)

    grid_msh = grid if grid < 512 else int(np.floor(grid / 2))
    kFund = 2 * np.pi / BoxSize
    kNyquist = np.pi * grid / BoxSize
    num_bins = 50
    kedges = np.linspace(kFund, kNyquist, num_bins)[:-1]

    measure_power_func = lambda pos, B, g: measure_power_pypower(pos, B, g, kedges)

    start = time.time()
    k_nb, Pk_nb = measure_power_func(pos_nb, BoxSize, grid)
    k_pm, Pk_pm = measure_power_func(pos_pm, BoxSize, grid)
    print(f"[seed {seed}] pm/nb P(k) measured in {time.time()-start:.1f}s")

    np.save(os.path.join(outdir, "k_nb.npy"), k_nb)
    np.save(os.path.join(outdir, "pk_nb.npy"), Pk_nb)
    np.save(os.path.join(outdir, "k_pm.npy"), k_pm)
    np.save(os.path.join(outdir, "pk_pm.npy"), Pk_pm)

    base = pb.PolyBin3D([BoxSize, BoxSize, BoxSize], [grid_msh, grid_msh, grid_msh],
                         boxcenter=[0, 0, 0], pixel_window='cic', backend='fftw',
                         nthreads=n_threads, sightline='global')
    k_msh = kedges[kedges <= np.pi * grid_msh / BoxSize]
    bspec = pb.BSpec(base, k_msh, lmax=0)

    for label, pos in (("pm", pos_pm), ("nb", pos_nb)):
        for g in (256, 512):
            np.save(os.path.join(outdir, f"delta_{label}_grid{g}.npy"),
                    compute_delta(pos, BoxSize, g))
        delta_msh = compute_delta(pos, BoxSize, grid_msh)
        bk = bspec.Bk_ideal(delta_msh, discreteness_correction=False)["b0"]
        np.save(os.path.join(outdir, f"bk_{label}.npy"), bk)
    np.save(os.path.join(outdir, "ks_bispectrum.npy"), bspec.get_ks())

    loop_result = iterate_correction_with_Q(
        pos_pm, BoxSize, grid, k_nb, Pk_nb, measure_power_func,
        n_iterations=n_iterations, n_y=201, seed=seed, n_threads=n_threads,
        zero_above_kmax=True, verbose=True,
    )

    for it, h in enumerate(loop_result['history'], start=1):
        pos_it = h['positions_corrected']
        np.save(os.path.join(outdir, f"k_corrected_it{it}.npy"), h['k_after'])
        np.save(os.path.join(outdir, f"pk_corrected_it{it}.npy"), h['Pk_after'])
        for g in (256, 512):
            np.save(os.path.join(outdir, f"delta_corrected_it{it}_grid{g}.npy"),
                    compute_delta(pos_it, BoxSize, g))
        delta_msh = compute_delta(pos_it, BoxSize, grid_msh)
        bk_it = bspec.Bk_ideal(delta_msh, discreteness_correction=False)["b0"]
        np.save(os.path.join(outdir, f"bk_corrected_it{it}.npy"), bk_it)

    print(f"[seed {seed}] done in {time.time()-start:.1f}s total")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["picola", "fastpm"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid", type=int, default=512)
    parser.add_argument("--n_iterations", type=int, default=3)
    parser.add_argument("--outroot", default="/work/hdd/bdne/nchartier/Q_correction")
    args = parser.parse_args()

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    run_realization(args.sim, args.seed, args.grid, args.n_iterations, n_threads, args.outroot)
