import os
os.environ["OMP_NUM_THREADS"] = "32"
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
from datetime import timedelta
import multiprocessing as mp

n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
pool = mp.Pool(processes=n_cores)
print(f'running on {n_cores} CPUs with {int(os.environ.get("OMP_NUM_THREADS"))} OpenMP threads.', file=sys.stdout)


def d_matrix_l2_numeric(beta):
    """Wigner small-d matrix for l=2 (m = [2,1,0,-1,-2]) using half-angle formulas."""
    c = np.cos(beta/2.0)
    s = np.sin(beta/2.0)
    return np.array([
        [c**4, -2*c**3*s, np.sqrt(6)*c**2*s**2, -2*c*s**3, s**4],
        [2*c**3*s, c**2*(2*c**2-1), -np.sqrt(3/2)*np.sin(beta)*np.cos(beta), s**2*(1-2*s**2), -2*c*s**3],
        [np.sqrt(6)*c**2*s**2, np.sqrt(3/2)*np.sin(beta)*np.cos(beta), 0.5*(3*np.cos(beta)**2-1), -np.sqrt(3/2)*np.sin(beta)*np.cos(beta), np.sqrt(6)*c**2*s**2],
        [2*c*s**3, s**2*(2*s**2-1), np.sqrt(3/2)*np.sin(beta)*np.cos(beta), c**2*(1-2*c**2), 2*c**3*s],
        [s**4, 2*c*s**3, np.sqrt(6)*c**2*s**2, 2*c**3*s, c**4]
    ], dtype=float)

# ----- 2. Wigner D^l using scipy -----
def wigner_D_l2(alpha, beta, gamma):
    """Return complex 5x5 Wigner D^2 matrix in |m=2..-2> basis."""
    return R.from_euler('zyz', [alpha, beta, gamma]).as_matrix(l=2)  # needs SciPy >=1.8


# complex <-> real basis unitary (rows are real orbitals in order: d_xy,d_yz,d_z2,d_xz,d_x2-y2) using Condon-Shortley phase convention
U_num = np.array([
    [ 1j/np.sqrt(2), 0, 0, 0, -1j/np.sqrt(2) ],
    [ 0, 1j/np.sqrt(2), 0, 1j/np.sqrt(2), 0 ],
    [ 0, 0, 1, 0, 0 ],
    [ 0, -1/np.sqrt(2), 0, 1/np.sqrt(2), 0 ],
    [ 1/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2) ]
], dtype=complex)


def D_real_numeric(alpha, beta, gamma):
    """
    Return the 5x5 real rotation matrix for l=2 in the real d-orbital basis.
    Euler angles alpha,beta,gamma in radians, Z(alpha)-Y(beta)-Z(gamma) convention.
    """
    m = np.array([2,1,0,-1,-2])
    exp_alpha = np.exp(-1j*m*alpha)  # shape (5,)
    exp_gamma = np.exp(-1j*m*gamma)  # shape (5,)
    d = d_matrix_l2_numeric(beta)    # shape (5,5), real
    # build complex Wigner D in |m> basis
    Dc = (exp_alpha[:,None] * d) * exp_gamma[None,:]  # elementwise multiply
    Dr = U_num @ Dc @ U_num.conj().T
    # numerical round-off: Dr should be real orthogonal; remove tiny imag parts
    Dr = np.real_if_close(Dr, tol=1e-9)
    return np.real(Dr)


def rotate_real_procar(P_real, alpha, beta, gamma):
    """
    Rotate a 5x5 PROCAR projection matrix given in the real d-orbital basis.
    Inputs:
      - P_real: (5,5) array-like real projection matrix in order [d_xy,d_yz,d_z2,d_xz,d_x2-y2]
      - alpha,beta,gamma: Euler angles in radians (Z(alpha)-Y(beta)-Z(gamma) convention)
    Returns:
      - P_rot: rotated (5,5) real matrix = D_real @ P_real @ D_real.T
    """
    P = np.asarray(P_real, dtype=float)
    if P.shape != (5,5):
        raise ValueError("P_real must be 5x5")
    Dr = D_real_numeric(alpha, beta, gamma)
    return Dr @ P @ Dr.T

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

def compute_row(nrow, num_points, init_proj, alpha, beta, gamma, orbital_num, rotate_procar):
    """
    helper function for parallelising the rotation of population
    each row in the outer loop (i.e. each row index in alpha and beta arrays) runs on separate CPUs
    """
    # inner loop
    rotated_proj_row = np.zeros((num_points, num_points))
    rotated_proj_resid_row = np.zeros((num_points, num_points))
    for ncol in range(num_points):
        for nrot in range(num_points):
            new_population = np.diag(rotate_procar(init_proj, alpha[nrow][ncol], beta[nrow][ncol], gamma[nrot]))
            rotated_proj_row[ncol][nrot] = new_population[orbital_num]
            rotated_proj_resid_row[ncol][nrot] = np.sum(np.delete(new_population, orbital_num))
    return (nrow, rotated_proj_row, rotated_proj_resid_row)

if __name__ == '__main__':
    
    # octahedral sites
    initial_proj_Fe1 = np.diag([0.953, 0.951, 0.987, 0.955, 0.978])
    initial_proj_Fe2 = np.diag([0.953, 0.951, 0.987, 0.955, 0.978])
    initial_proj_Fe3 = np.diag([0.953, 0.954, 0.986, 0.957, 0.978])
    initial_proj_Fe4 = np.diag([0.953, 0.954, 0.986, 0.957, 0.978])
    initial_proj_Fe5 = np.diag([0.082, 0.081, 0.274, 0.091, 0.233])
    initial_proj_Fe6 = np.diag([0.082, 0.081, 0.273, 0.091, 0.234])
    initial_proj_Fe7 = np.diag([0.082, 0.081, 0.274, 0.091, 0.233])
    initial_proj_Fe8 = np.diag([0.081, 0.081, 0.273, 0.092, 0.234])
    # teterahedral sites
    initial_proj_Fe9 =  np.diag([0.132, 0.197, 0.153, 0.165, 0.254])
    initial_proj_Fe10 = np.diag([0.132, 0.197, 0.153, 0.165, 0.254])
    initial_proj_Fe11 = np.diag([0.198, 0.130, 0.203, 0.173, 0.200])
    initial_proj_Fe12 = np.diag([0.198, 0.130, 0.203, 0.173, 0.200])
    initial_proj_Fe13 = np.diag([0.962, 0.962, 0.964, 0.960, 0.965])
    initial_proj_Fe14 = np.diag([0.962, 0.962, 0.963, 0.960, 0.965])
    initial_proj_Fe15 = np.diag([0.962, 0.963, 0.961, 0.960, 0.972])
    initial_proj_Fe16 = np.diag([0.962, 0.964, 0.961, 0.960, 0.972])

    # [d_xy,d_yz,d_z2,d_xz,d_x2-y2]
    select_orbital = 0
    select_init_proj = initial_proj_Fe1
    site_label = r'$\mathrm{Fe_{ohd}}$ '
    d_orbital_labels = {
        0 : r'$\mathrm{d_{xy}}$',
        1 : r'$\mathrm{d_{yz}}$',
        2 : r'$\mathrm{d_{z^2}}$',
        3 : r'$\mathrm{d_{xz}}$',
        4 : r'$\mathrm{d_{x^2-y^2}}$'
    }

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


    # Define unit sphere
    # TODO check if inversion symmetry can be used to only calculate upper hemisphere
    u = np.linspace(1e-6, 2 * np.pi - 1e-6, surface_resolution)
    v = np.linspace(1e-6, np.pi - 1e-6, surface_resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Calculate euler angles for each point on sphere surface
    alpha, beta = get_euler_angles(y,z)
    gamma = np.linspace(0, 2 * np.pi,surface_resolution)
    print(f'alpha range: [{np.min(np.rad2deg(alpha)):.3f}, {np.max(np.rad2deg(alpha)):.3f}]', file=sys.stdout)
    print(f'beta range: [{np.min(np.rad2deg(beta)):.3f}, {np.max(np.rad2deg(beta)):.3f}]', file=sys.stdout)
    print(f'gamma range: [{np.min(np.rad2deg(gamma)):.3f}, {np.max(np.rad2deg(gamma)):.3f}]', file=sys.stdout)
    print(f'calculating a total of {surface_resolution}x{surface_resolution}x{surface_resolution} = {surface_resolution**3:.1e} projections...')
    
    # For each sampling point calculate the rotated population
    loop_start = time.time()

    rotated_proj = np.zeros((surface_resolution, surface_resolution, surface_resolution))
    rotated_proj_residuals = np.zeros_like(rotated_proj)

    # use multiprocessing Pool to distribute outer loop over available cores
    with mp.Pool(processes=n_cores) as pool:
        results = pool.starmap(
            compute_row,
            [(nrow, surface_resolution, select_init_proj, alpha, beta, gamma, select_orbital, rotate_real_procar)
             for nrow in range(surface_resolution)]
        )

    # Reassemble results
    for nrow, row_proj, row_resid in results:
        rotated_proj[nrow, :, :] = row_proj
        rotated_proj_residuals[nrow, :, :] = row_resid

    loop_end = time.time()
    if DEBUG: print(f'[DEBUG] total time for calculating new projections: {timedelta(seconds=(loop_end-loop_start))}', file=sys.stdout)

    
    # Determine indices at which the population (i.e. projection of selected orbtal onto itself) is maximised
    opt_start = time.time()
    projections_max_ind = np.unravel_index(np.argmax(rotated_proj, axis=None), rotated_proj.shape)
    residuals_min_ind = np.unravel_index(np.argmin(rotated_proj_residuals, axis=None), rotated_proj_residuals.shape)
    mod_proj_max_ind = np.unravel_index(np.argmin(rotated_proj_residuals-rotated_proj, axis=None), rotated_proj_residuals.shape)

    # Find optimal rotation by maximising orbital projection
    alpha_opt1 = np.mod(alpha[projections_max_ind[:2]], np.pi)
    beta_opt1 = np.mod(beta[projections_max_ind[:2]],np.pi)
    gamma_opt1 = np.mod(gamma[projections_max_ind[2]],2 * np.pi)
    print(f'orientation of projection maximum {rotated_proj[projections_max_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt1):.1f}, {np.rad2deg(beta_opt1):.1f}, {np.rad2deg(gamma_opt1):.1f})', np.round(np.diag(rotate_real_procar(select_init_proj, alpha_opt1, beta_opt1, gamma_opt1)), decimals=3), file=sys.stdout)
    # Get the new Z-axis wrt. initial coordinate system
    orientation_vec1 = get_rotation_matrix(alpha_opt1, beta_opt1, gamma_opt1) @ np.array([0,0,1])
    print('new z axis: ', np.round(orientation_vec1, decimals=2), file=sys.stdout)

    # Find optimal rotation by minimising residuals
    alpha_opt2 = np.mod(alpha[residuals_min_ind[:2]], np.pi)
    beta_opt2 = np.mod(beta[residuals_min_ind[:2]], np.pi)
    gamma_opt2 = np.mod(gamma[residuals_min_ind[2]],2 * np.pi)
    print(f'orientation of residual minimum {rotated_proj_residuals[residuals_min_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt2):.1f}, {np.rad2deg(beta_opt2):.1f}, {np.rad2deg(gamma_opt2):.1f})', np.round(np.diag(rotate_real_procar(select_init_proj, alpha_opt2, beta_opt2, gamma_opt2)), decimals=3), file=sys.stdout)
    # Get the new Z-axis wrt. initial coordinate system
    orientation_vec2 = get_rotation_matrix(alpha_opt2, beta_opt2, gamma_opt2) @ np.array([0,0,1])
    print('new z axis: ', np.round(orientation_vec2, decimals=2), file=sys.stdout)

    # Find optimal projection by minimising (residuals - projection)
    alpha_opt3 = np.mod(alpha[mod_proj_max_ind[:2]], np.pi)
    beta_opt3 = np.mod(beta[mod_proj_max_ind[:2]], np.pi)
    gamma_opt3 = np.mod(gamma[mod_proj_max_ind[2]],2 * np.pi)
    print(f'orientation of residuals-projection minimum {(rotated_proj_residuals-rotated_proj)[mod_proj_max_ind]:.5f} (in euler angles): ({np.rad2deg(alpha_opt3):.1f}, {np.rad2deg(beta_opt3):.1f}, {np.rad2deg(gamma_opt3):.1f})', np.round(np.diag(rotate_real_procar(select_init_proj, alpha_opt3, beta_opt3, gamma_opt3)), decimals=3), file=sys.stdout)
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
    print(f'projection numbers range for plotting: {minn:.5f}, {maxx:.5f}', file=sys.stdout)
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Set the viewing orientation
    ax.view_init(elev = viewing_orientation[0], 
                 azim = viewing_orientation[1],
                 roll = viewing_orientation[2])

    # Plot the surface
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
    ax.set_aspect('equal')
    ax.set_title(site_label + d_orbital_labels[select_orbital] + '-projection' + '\n' + rf"$z'=({orientation_vec[0]:.1f},{orientation_vec[1]:.1f},{orientation_vec[2]:.1f})$ ($\gamma = {np.rad2deg(gamma_opt):.1f}^{{\circ}}$)", y=0.95)
    
    plt.savefig(plot_filename, dpi=400)
    plot_end = time.time()
    if DEBUG: print(f'[DEBUG] total time for plotting: {timedelta(seconds=(plot_end-plot_start))}', file=sys.stdout)
    if DEBUG: print(f'\n[DEBUG] total execution time: {timedelta(seconds=(plot_end-loop_start))}', file=sys.stdout)
    #plt.show()

