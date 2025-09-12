import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # ensure BLAS/OpenMP libs don't oversubscribe!

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
from datetime import timedelta
import multiprocessing as mp
from numba import njit

# complex <-> real basis unitary (rows are real orbitals in order: d_xy,d_yz,d_z2,d_xz,d_x2-y2) using Condon-Shortley phase convention
U_num = np.array([
    [ 1j/np.sqrt(2), 0, 0, 0, -1j/np.sqrt(2) ],
    [ 0, 1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0 ],
    [ 0, 0, 1, 0, 0 ],
    [ 0, -1/np.sqrt(2), 0, 1/np.sqrt(2), 0 ],
    [ 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2) ]
], dtype=np.complex128)

# ---------------- numba-compiled helpers -----------------
@njit(fastmath=True)
def d_matrix_l2_numeric_nb(beta):
    """
    Wigner small d-matrix for l=2 (m = [2,1,0,-1,-2]) using half-angle formulas.

    beta: Euler angle in radians, Z(alpha)-Y(beta)-Z(gamma) convention.

    returns: full Wigner small d-matrix for given beta
    """
    c = np.cos(beta/2.0)
    s = np.sin(beta/2.0)
    out = np.empty((5,5), dtype=np.float64)
    out[0,0] = c**4
    out[0,1] = -2.0*c**3*s
    out[0,2] = np.sqrt(6.0)*c**2*s**2
    out[0,3] = -2.0*c*s**3
    out[0,4] = s**4

    out[1,0] = 2.0*c**3*s
    out[1,1] = c**2*(2.0*c**2 - 1.0)
    out[1,2] = -np.sqrt(3.0/2.0)*np.sin(beta)*np.cos(beta)
    out[1,3] = s**2*(1.0 - 2.0*s**2)
    out[1,4] = -2.0*c*s**3

    out[2,0] = np.sqrt(6.0)*c**2*s**2
    out[2,1] = np.sqrt(3.0/2.0)*np.sin(beta)*np.cos(beta)
    out[2,2] = 0.5*(3.0*np.cos(beta)**2 - 1.0)
    out[2,3] = -np.sqrt(3.0/2.0)*np.sin(beta)*np.cos(beta)
    out[2,4] = np.sqrt(6.0)*c**2*s**2

    out[3,0] = 2.0*c*s**3
    out[3,1] = s**2*(2.0*s**2 - 1.0)
    out[3,2] = np.sqrt(3.0/2.0)*np.sin(beta)*np.cos(beta)
    out[3,3] = c**2*(1.0 - 2.0*c**2)
    out[3,4] = 2.0*c**3*s

    out[4,0] = s**4
    out[4,1] = 2.0*c*s**3
    out[4,2] = np.sqrt(6.0)*c**2*s**2
    out[4,3] = 2.0*c**3*s
    out[4,4] = c**4
    return out

@njit(fastmath=True)
def D_complex_from_euler_nb(alpha, beta, gamma):
    """
    Build complex Wigner D (5x5) in |m=2..-2> basis using d_matrix and e^{ - i m alpha } and e^{ - i m gamma } phases.

    alpha, beta, gamma: Euler angles in radians, Z(alpha)-Y(beta)-Z(gamma) convention.

    Returns full Wigner D-matrix as complex128 5x5 array.
    """
    m = np.array([2.0,1.0,0.0,-1.0,-2.0], dtype=np.float64)
    exp_alpha = np.empty(5, dtype=np.complex128)
    exp_gamma = np.empty(5, dtype=np.complex128)
    for i in range(5):
        exp_alpha[i] = np.exp(-1j * m[i] * alpha)
        exp_gamma[i] = np.exp(-1j * m[i] * gamma)
    d = d_matrix_l2_numeric_nb(beta)
    Dc = np.empty((5,5), dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            Dc[i,j] = exp_alpha[i] * d[i,j] * exp_gamma[j]
    return Dc

@njit(fastmath=True)
def D_real_numeric_nb(alpha, beta, gamma, Uflat):
    """
    Compute the 5x5 real Wigner D-matrix for l=2 in the real d-orbital basis.

    alpha, beta, gamma: Euler angles in radians, Z(alpha)-Y(beta)-Z(gamma) convention.
    Uflat: flattened (25,) complex128 array representing U_num in row-major form.

    returns full Wigner D-matrix as real 5x5 array
    """
    # reconstruct U_num from Uflat
    U = np.empty((5,5), dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            U[i,j] = Uflat[i*5 + j]

    Dc = D_complex_from_euler_nb(alpha, beta, gamma)  # complex (5,5)
    # Dr = U @ Dc @ U.conj().T  (complex) then take real part
    temp = np.empty((5,5), dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            s = 0+0j
            for k in range(5):
                s += U[i,k] * Dc[k,j]
            temp[i,j] = s
    Drc = np.empty((5,5), dtype=np.complex128)
    for i in range(5):
        for j in range(5):
            s = 0+0j
            for k in range(5):
                s += temp[i,k] * np.conjugate(U[j,k])
            Drc[i,j] = s
    # Dr should be real orthogonal; return real part
    Dr = np.empty((5,5), dtype=np.float64)
    for i in range(5):
        for j in range(5):
            Dr[i,j] = Drc[i,j].real
    return Dr

@njit(fastmath=True)
def compute_row_numba(nrow, num_points, init_proj_diag, alpha, beta, gamma, orbital_num, Uflat):
    """
    helper function for parallelising the rotation of population

    Compute one row (nrow) of rotated_proj and rotated_proj_residuals.
    init_proj_diag: shape (5,) diagonal elements of initial projection P
    alpha, beta: 2D arrays shape (num_points,num_points)
    gamma: 1D array shape (num_points,)

    Returns rotated_proj_row (num_points,num_points) and rotated_proj_resid_row (num_points,num_points)
    """
    row_proj = np.zeros((num_points, num_points), dtype=np.float64)
    row_resid = np.zeros((num_points, num_points), dtype=np.float64)

    # temporary matrices
    Dr = np.empty((5,5), dtype=np.float64)
    # exploit that init_proj is diagonal: P_{ij} = init_proj_diag[i] if i==j else 0

    for ncol in range(num_points):
        a = alpha[nrow, ncol]
        b = beta[nrow, ncol]
        for nrot in range(num_points):
            g = gamma[nrot]
            # compute rotation matrix
            Dr = D_real_numeric_nb(a, b, g, Uflat)
            # compute M = Dr @ P @ Dr.T, with P diagonal -> M_{ij} = sum_k Dr[i,k] * init_proj_diag[k] * Dr[j,k]
            # only need diagonal elements M_ii
            diagM = np.zeros(5, dtype=np.float64)
            for i in range(5):
                s = 0.0
                for k in range(5):
                    s += Dr[i,k] * init_proj_diag[k] * Dr[i,k]
                diagM[i] = s
            new_population_orb = diagM[orbital_num]
            total = 0.0
            for kk in range(5):
                total += diagM[kk]
            residual = total - new_population_orb
            row_proj[ncol, nrot] = new_population_orb
            row_resid[ncol, nrot] = residual
    return row_proj, row_resid

# ---------------- high-level worker wrapper (picklable) -----------------

def compute_row_worker(args):
    """Top-level wrapper that will be pickled and executed in worker processes."""
    (nrow, num_points, init_proj_diag, alpha, beta, gamma, orbital_num, Uflat) = args
    # numba-compiled function takes numpy arrays; ensure types are as expected
    row_proj, row_resid = compute_row_numba(nrow, num_points, init_proj_diag, alpha, beta, gamma, orbital_num, Uflat)
    return (nrow, row_proj, row_resid)

# ---------------- original helper functions kept for plotting & IO -----------------

def get_euler_angles(z2, z3):
    """
    For a given rotated frame evaluate proper Euler angles using ZYZ convention (ignoring gamma rotation, i.e. rotation about new Z'' axis) 
    Inputs:
      - z2, z3: y and z coordinates of  rotated z axis in original frame
    """
    alpha = np.arccos(z2/np.sqrt(1-z3**2))
    beta = np.arccos(z3)
    return (alpha,beta)

def get_rotation_matrix(alpha, beta, gamma):
    """
    Euler angles alpha,beta,gamma in radians, Z(alpha)-Y(beta)-Z(gamma) convention.
    """
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cc = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sc = np.sin(gamma)
    return np.array([
        [ca*cb*cc - sa*sc, -cc*sa - ca*cb*sc, ca*sb],
        [ca*sc + cb*cc*sa, ca*cc - cb*sa*sc, sa*sb],
        [-cc*sb, sb*sc, cb]
    ])

def rotate_real_procar(P_real, alpha, beta, gamma, Uflat):
    """
    Rotate a 5x5 PROCAR projection matrix given in the real d-orbital basis.
    Inputs:
      - P_real: (5,5) diagonal real projection matrix in order [d_xy,d_yz,d_z2,d_xz,d_x2-y2]
      - alpha,beta,gamma: Euler angles in radians (Z(alpha)-Y(beta)-Z(gamma) convention)
      - Uflat: flattened (25,) complex128 array representing U_num in row-major form.
    Returns:
      - P_rot: rotated (5,5) real matrix = D_real @ P_real @ D_real.T
    """
    P = np.asarray(P_real, dtype=float)
    if P.shape != (5,5):
        raise ValueError("P_real must be 5x5")
    Dr = D_real_numeric_nb(alpha, beta, gamma, Uflat)
    return Dr @ P @ Dr.T


# ---------------- main script -----------------
if __name__ == '__main__':
    mp_start = 'fork'
    try:
        mp.set_start_method(mp_start)
    except RuntimeError:
        # start method already set
        pass

    n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('NUM_PROCS', '1')))
    print(f'running on {n_cores} CPUs.', file=sys.stdout)

    # initial projection
    # octahedral sites
    initial_proj_Fe1 = np.array([0.953, 0.951, 0.987, 0.955, 0.978])
    initial_proj_Fe2 = np.array([0.953, 0.951, 0.987, 0.955, 0.978])
    initial_proj_Fe3 = np.array([0.953, 0.954, 0.986, 0.957, 0.978])
    initial_proj_Fe4 = np.array([0.953, 0.954, 0.986, 0.957, 0.978])
    initial_proj_Fe5 = np.array([0.082, 0.081, 0.274, 0.091, 0.233])
    initial_proj_Fe6 = np.array([0.082, 0.081, 0.273, 0.091, 0.234])
    initial_proj_Fe7 = np.array([0.082, 0.081, 0.274, 0.091, 0.233])
    initial_proj_Fe8 = np.array([0.081, 0.081, 0.273, 0.092, 0.234])
    # teterahedral sites
    initial_proj_Fe9 =  np.array([0.132, 0.197, 0.153, 0.165, 0.254])
    initial_proj_Fe10 = np.array([0.132, 0.197, 0.153, 0.165, 0.254])
    initial_proj_Fe11 = np.array([0.198, 0.130, 0.203, 0.173, 0.200])
    initial_proj_Fe12 = np.array([0.198, 0.130, 0.203, 0.173, 0.200])
    initial_proj_Fe13 = np.array([0.962, 0.962, 0.964, 0.960, 0.965])
    initial_proj_Fe14 = np.array([0.962, 0.962, 0.963, 0.960, 0.965])
    initial_proj_Fe15 = np.array([0.962, 0.963, 0.961, 0.960, 0.972])
    initial_proj_Fe16 = np.array([0.962, 0.964, 0.961, 0.960, 0.972])

    select_init_proj = initial_proj_Fe1
    
    # [d_xy,d_yz,d_z2,d_xz,d_x2-y2]
    select_orbital = 0 
    site_label = r'$\mathrm{Fe_{ohd}}$ '

    # Number of data points for each euler angle
    surface_resolution = 500

    # Plotting parameters 
    plot_filename = 'Fe1_dxy_rotations.png'
    # (elevation, azimuth, roll)
    viewing_orientation = (12, -30, 0)
    # d_xy: elev=12, azim=-30, roll=0, cbar left
    # d_yz: elev=12, azim=45, roll=0, cbar right
    # d_z^2: elev=12, azim=-30, roll=0, cbar left
    # d_xz: elev=12, azim=45, roll=0, cbar right
    # d_x^2-y^2: elev=12, azim=-30, roll=0, cbar left
    cbar_location = 'left'

    # Which quantity to plot on sphere surface: orbital projection [P], population residuals [M], or modified residuals [M]
    plot_quantity = 'M'

    DEBUG = True


    # build sphere and angles
    # TODO check if inversion symmetry can be used to only calculate upper hemisphere
    u = np.linspace(1e-6, 2*np.pi - 1e-6, surface_resolution)
    v = np.linspace(1e-6, np.pi - 1e-6, surface_resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    alpha, beta = get_euler_angles(y, z)
    gamma = np.linspace(0, 2*np.pi, surface_resolution)

    print(f'alpha range: [{np.min(np.rad2deg(alpha)):.3f}, {np.max(np.rad2deg(alpha)):.3f}]', file=sys.stdout)
    print(f'beta range: [{np.min(np.rad2deg(beta)):.3f}, {np.max(np.rad2deg(beta)):.3f}]', file=sys.stdout)
    print(f'gamma range: [{np.min(np.rad2deg(gamma)):.3f}, {np.max(np.rad2deg(gamma)):.3f}]', file=sys.stdout)
    print(f'calculating a total of {surface_resolution}x{surface_resolution}x{surface_resolution} = {surface_resolution**3:.1e} projections...')

    # Prepare flattened U_num for passing into numba
    Uflat = U_num.ravel()
    
    # Create tasks: one task per nrow
    tasks = [
        (nrow, surface_resolution, select_init_proj, alpha, beta, gamma, select_orbital, Uflat)
        for nrow in range(surface_resolution)
    ]

    rotated_proj = np.zeros((surface_resolution, surface_resolution, surface_resolution), dtype=np.float64)
    rotated_proj_residuals = np.zeros_like(rotated_proj)

    loop_start = time.time()
    # Use a process pool over imap_unordered
    with mp.Pool(processes=n_cores) as pool:
        for _j, res in enumerate(pool.imap_unordered(compute_row_worker, tasks)):
            nrow, row_proj, row_resid = res
            rotated_proj[nrow, :, :] = row_proj
            rotated_proj_residuals[nrow, :, :] = row_resid

    loop_end = time.time()
    if DEBUG: print(f'[DEBUG] total time for calculating new projections: {timedelta(seconds=(loop_end-loop_start))}', file=sys.stdout)

    # Determine indices at which the population (i.e. projection of selected orbtal onto itself) is maximised
    opt_start = time.time()
    projections_max_ind = np.unravel_index(np.argmax(rotated_proj, axis=None), rotated_proj.shape)
    residuals_min_ind = np.unravel_index(np.argmin(rotated_proj_residuals, axis=None), rotated_proj_residuals.shape)
    mod_proj_max_ind = np.unravel_index(np.argmin(rotated_proj_residuals-rotated_proj, axis=None), rotated_proj_residuals.shape)

    # Find optimal rotation by maximising orbital projection (option [P])
    alpha_opt1 = np.mod(alpha[projections_max_ind[:2]], np.pi)
    beta_opt1 = np.mod(beta[projections_max_ind[:2]],np.pi)
    gamma_opt1 = np.mod(gamma[projections_max_ind[2]],2 * np.pi)
    print(f'orientation of projection maximum {rotated_proj[projections_max_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt1):.1f}, {np.rad2deg(beta_opt1):.1f}, {np.rad2deg(gamma_opt1):.1f})', 
          np.round(np.diag(rotate_real_procar(np.diag(select_init_proj), alpha_opt1, beta_opt1, gamma_opt1, Uflat)), decimals=3), file=sys.stdout)
    # Get the new Z-axis wrt. initial coordinate system
    orientation_vec1 = get_rotation_matrix(alpha_opt1, beta_opt1, gamma_opt1) @ np.array([0,0,1])
    print('new z axis: ', np.round(orientation_vec1, decimals=2), file=sys.stdout)

    # Find optimal rotation by minimising residuals (option [R])
    alpha_opt2 = np.mod(alpha[residuals_min_ind[:2]], np.pi)
    beta_opt2 = np.mod(beta[residuals_min_ind[:2]], np.pi)
    gamma_opt2 = np.mod(gamma[residuals_min_ind[2]],2 * np.pi)
    print(f'orientation of residual minimum {rotated_proj_residuals[residuals_min_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt2):.1f}, {np.rad2deg(beta_opt2):.1f}, {np.rad2deg(gamma_opt2):.1f})', 
          np.round(np.diag(rotate_real_procar(np.diag(select_init_proj), alpha_opt2, beta_opt2, gamma_opt2, Uflat)), decimals=3), file=sys.stdout)
    # Get the new Z-axis wrt. initial coordinate system
    orientation_vec2 = get_rotation_matrix(alpha_opt2, beta_opt2, gamma_opt2) @ np.array([0,0,1])
    print('new z axis: ', np.round(orientation_vec2, decimals=2), file=sys.stdout)

    # Find optimal projection by minimising (residuals - projection) (option [M])
    alpha_opt3 = np.mod(alpha[mod_proj_max_ind[:2]], np.pi)
    beta_opt3 = np.mod(beta[mod_proj_max_ind[:2]], np.pi)
    gamma_opt3 = np.mod(gamma[mod_proj_max_ind[2]],2 * np.pi)
    print(f'orientation of residuals-projection minimum {(rotated_proj_residuals-rotated_proj)[mod_proj_max_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt3):.1f}, {np.rad2deg(beta_opt3):.1f}, {np.rad2deg(gamma_opt3):.1f})', 
          np.round(np.diag(rotate_real_procar(np.diag(select_init_proj), alpha_opt3, beta_opt3, gamma_opt3, Uflat)), decimals=3), file=sys.stdout)
    orientation_vec3 = get_rotation_matrix(alpha_opt3, beta_opt3, gamma_opt3) @ np.array([0,0,1])
    print('new z axis: ', np.round(orientation_vec3, decimals=2), file=sys.stdout)
    
    opt_end = time.time()
    if DEBUG: print(f'[DEBUG] total time for finding optimum orbital orientation: {timedelta(seconds=(opt_end-opt_start))}', file=sys.stdout)
    
    # Plot the population of the selected rotated orbital on a spherical surface
    plot_start = time.time()
    if plot_quantity == 'P':
        color_dimension = rotated_proj[:][:][projections_max_ind[2]]
        orientation_vec = orientation_vec1
        gamma_opt = gamma_opt1
        cbar_label = rf'$|\langle Y_{{2,{select_orbital-2}}}^{{\alpha}} | \phi_{{n\bf{{k}}}} \rangle |^2$'
    elif plot_quantity == 'R':
        color_dimension = rotated_proj_residuals[:][:][residuals_min_ind[2]]
        orientation_vec = orientation_vec2
        gamma_opt = gamma_opt2
        cbar_label = rf'$\sum_{{m \neq {select_orbital-2}}}|\langle Y_{{2,m}}^{{\alpha}} | \phi_{{n\bf{{k}}}} \rangle |^2$'
    elif plot_quantity == 'M':
        color_dimension = (rotated_proj_residuals-rotated_proj)[:][:][mod_proj_max_ind[2]]
        orientation_vec = orientation_vec3
        gamma_opt = gamma_opt3
        cbar_label = rf'$\sum_{{m \neq {select_orbital-2}}}|\langle Y_{{2,m}}^{{\alpha}} | \phi_{{n\bf{{k}}}} \rangle |^2 - |\langle Y_{{2,{select_orbital-2}}}^{{\alpha}} | \phi_{{n\bf{{k}}}} \rangle |^2$'

    minn, maxx = color_dimension.min(), color_dimension.max()
    if DEBUG: print(f'projection numbers range for plotting: {minn:.5f}, {maxx:.5f}', file=sys.stdout)
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.set_aspect('equal')
    # Set the viewing orientation
    ax.view_init(elev = viewing_orientation[0], 
                 azim = viewing_orientation[1],
                 roll = viewing_orientation[2])
    
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    ax.set_xlabel('x', labelpad=0.00001)
    ax.set_xticklabels([])
    ax.set_ylabel('y', labelpad=0.00001)
    ax.set_yticklabels([])
    ax.set_zlabel('z', labelpad=0.00001)
    ax.set_zticklabels([])

    # plot the new Z'' axis as an arrow 
    ax.quiver(
    -orientation_vec[0]*1.3, -orientation_vec[1]*1.3, -orientation_vec[2]*1.3, # <-- starting point of vector
    orientation_vec[0]*2.6, orientation_vec[1]*2.6, orientation_vec[2]*2.6, # <-- directions of vector
    color = 'black', alpha = .8, lw = 2, arrow_length_ratio=0.05
    )

    fig.colorbar(m, 
                 shrink=0.6, 
                 aspect=10, 
                 ax=ax, 
                 label=cbar_label, 
                 pad=0.01, 
                 location=cbar_location)
    
    d_orbital_labels = {
        0 : r'$\mathrm{d_{xy}}$',
        1 : r'$\mathrm{d_{yz}}$',
        2 : r'$\mathrm{d_{z^2}}$',
        3 : r'$\mathrm{d_{xz}}$',
        4 : r'$\mathrm{d_{x^2-y^2}}$'
    }

    ax.set_title(site_label + d_orbital_labels[select_orbital] + '-projection' + '\n' + rf"$z'=({orientation_vec[0]:.1f},{orientation_vec[1]:.1f},{orientation_vec[2]:.1f})$ ($\gamma = {np.rad2deg(gamma_opt):.1f}^{{\circ}}$)", y=0.95)

    plt.savefig(plot_filename, dpi=400)

    plot_end = time.time()
    print(f'[DEBUG] total time for plotting: {timedelta(seconds=(plot_end-plot_start))}', file=sys.stdout)
    print(f'\n[DEBUG] total execution time: {timedelta(seconds=(plot_end-loop_start))}', file=sys.stdout)