import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    select_orbital = 4
    select_init_proj = initial_proj_Fe1
    site_label = r'$\mathrm{Fe_{ohd}}$ '
    d_orbital_labels = {
        0 : r'$\mathrm{d_{xy}}$',
        1 : r'$\mathrm{d_{yz}}$',
        2 : r'$\mathrm{d_{z^2}}$',
        3 : r'$\mathrm{d_{xz}}$',
        4 : r'$\mathrm{d_{x^2-y^2}}$'
    }

    # number of data points for each euler angle
    surface_resolution = 100

    # define unit sphere
    u = np.linspace(1e-6, 2 * np.pi - 1e-6, surface_resolution)
    v = np.linspace(1e-6, np.pi - 1e-6, surface_resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # calculate euler angles for each point on sphere surface
    alpha, beta = get_euler_angles(y,z)
    gamma = np.deg2rad(0.0) # set gamma to zero for now
    gamma_trial = np.linspace(0, 2 * np.pi,surface_resolution)
    print(f'alpha range: [{np.min(np.rad2deg(alpha)):.3f}, {np.max(np.rad2deg(alpha)):.3f}]')
    print(f'beta range: [{np.min(np.rad2deg(beta)):.3f}, {np.max(np.rad2deg(beta)):.3f}]')

    rotated_proj = np.zeros(shape=(surface_resolution,surface_resolution))
    # TODO add 3rd dimension to rotated_proj and fill with rotated projections wrt gamma_trial
    # TODO determine projection_max_ind by looking at minimum of residuals (or residuals - new projection)
    # TODO use this optimised gamma angle to plot the surface and find the new orientation

    # for each sampling point calculate the rotated population
    for nrow in range(surface_resolution):
        for ncol in range(surface_resolution):
            rotated_proj[nrow][ncol] = np.diag(rotate_real_procar(select_init_proj, alpha[nrow][ncol], beta[nrow][ncol], gamma))[select_orbital]

    

    color_dimension = rotated_proj

    # Determine indices at which the population (i.e. projection of selected orbtal onto itself) is maximised
    projections_max_ind = np.unravel_index(np.argmax(color_dimension, axis=None), color_dimension.shape)
    print(f'orientation of projection extremum {color_dimension[projections_max_ind]:.5f} (in euler angles): ({np.rad2deg(alpha[projections_max_ind]):.3f}, {np.rad2deg(beta[projections_max_ind]):.3f}, {np.rad2deg(gamma):.3f})')
    
    # Get the new Z-axis wrt. initial coordinate system
    orientation_vec = get_rotation_matrix(alpha[projections_max_ind], beta[projections_max_ind], gamma) @ np.array([0,0,1])
    print('new z axis of projection extremum: ', np.round(orientation_vec, decimals=2))

    # For the optimised (wrt. alpha and beta) population check which angle gamma minimises the projection residuals 
    new_populations = np.zeros(shape=(len(gamma_trial),5))
    residuals = np.zeros(len(gamma_trial))
    for i,g in enumerate(gamma_trial):
        new_pop = np.round(np.diag(rotate_real_procar(select_init_proj, alpha[projections_max_ind], beta[projections_max_ind], g)), decimals=3)
        resid = np.sum(np.delete(new_pop,select_orbital))
        new_populations[i] = new_pop
        residuals[i] = resid
        #print(f'new populations (gamma = {np.rad2deg(g):.1f}): ', new_pop, f'R={resid:.3f}')

    min_resid_population = new_populations[np.argmin(residuals)]
    print('Population that minimises residuals: ', min_resid_population, f'gamma={np.rad2deg(gamma_trial[np.argmin(residuals)]):.2f}, R={residuals[np.argmin(residuals)]:.3f}')

    # Plot the population of the selected rotated orbital on a spherical surface
    minn, maxx = color_dimension.min(), color_dimension.max()
    print(f'projection numbers range: {minn:.5f}, {maxx:.5f}')
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Plot the surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    ax.set_xlabel('x')
    ax.set_xticklabels([])
    ax.set_ylabel('y')
    ax.set_yticklabels([])
    ax.set_zlabel('z')
    ax.set_zticklabels([])

    # plot the new Z'' axis as an arrow 
    ax.quiver(
    -orientation_vec[0]*1.3, -orientation_vec[1]*1.3, -orientation_vec[2]*1.3, # <-- starting point of vector
    orientation_vec[0]*2.6, orientation_vec[1]*2.6, orientation_vec[2]*2.6, # <-- directions of vector
    color = 'black', alpha = .8, lw = 2, arrow_length_ratio=0.05
    )

    fig.colorbar(m, shrink=0.5, aspect=7, ax=ax, label=rf'$|\langle Y_{{2,{select_orbital-2}}}^{{\alpha}} | \phi_{{n\bf{{k}}}} \rangle |^2$', pad=0.005)
    ax.set_aspect('equal')
    ax.set_title(site_label + d_orbital_labels[select_orbital] + '-projection' + '\n' + rf"$z'=({orientation_vec[0]:.1f},{orientation_vec[1]:.1f},{orientation_vec[2]:.1f})$", y=0.93)
    #plt.subplots_adjust(top=0.8)

    # set the voewing orientation
    ax.view_init(elev=12, azim=65, roll=0)
    
    #plt.savefig('Fe1_dx2-y2_rotations.png', dpi=400)
    plt.show()

