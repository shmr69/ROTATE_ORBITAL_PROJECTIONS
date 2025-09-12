# ROTATE_ORBITAL_PROJECTIONS

This post-processing tool is aimed at dealing with the issue when the reference coordinate system (usually the unit cell cartesian axes) are not aligned with the cartesian axes of the given coordination environment. This will usually lead to mixing between d-orbital populations (i.e. projection of DFT wave functions onto atomic orbitals).\
The idea is that the spherical harmonics used in the population analysis are rotated until an optimal projection is found, and the associated rotation determines the new orientation of atomic orbitals in the given coordination environment.

## Requirements

- Python 3.9+
- Numpy 1.21+
- Numba 0.55+
- Matplotlib 3.5+

The file `environment.yml` includes a list of all the requirements and one can install this package with all dependencies inside a conda virtual environment: 
```
conda env create -f environment.yml
conda activate orbital-rotation
```


## Theory
The population numbers that are outputted by most available DFT codes (e.g. VASP, CASTEP,...) are projections of Kohn-Sham orbitals $\phi_{n\mathbf{k}}$ onto *real* spherical harmonics $Y_{lm}^{\alpha}$ for every ion with index $\alpha$. For d-orbitals, a $5\times5$ *real* projection matrix $\mathbf{P}^{(r)}$ can be defined with elements $|\langle Y_{lm}^{\alpha} | \phi_{n\mathbf{k}} \rangle|^2$. \
These *real* spherical harmonics representing atomic orbitals are real functions $Y_{lm}:S^2 \rightarrow \mathbb{R}$, generally defined by linear combinations of the complex spherical harmonics $Y_{l}^{m}:S^2 \rightarrow \mathbb{C}$ (within the Condon-Shortley phase convention). For example, for $l=2$ the d-orbitals are represented by\
\
$d_{xy}\propto \frac{i}{\sqrt{2}}(Y_{2}^{-2}-Y_{2}^{2}),\,d_{yz}\propto\frac{i}{\sqrt{2}}(Y_{2}^{-1}-Y_{2}^{1}),\,d_{z^2}\propto Y_{2}^{0},\,d_{xz}\propto\frac{1}{\sqrt{2}}(Y_{2}^{-1}-Y_{2}^{1}),\,d_{x^2-y^2}\propto\frac{1}{\sqrt{2}}(Y_{2}^{-2}-Y_{2}^{2})$\
\
Therefore, the projection matrix of Kohn-Sham orbitals onto complex spherical harmonics can be obtained by a unitary transformation: $\mathbf{P}^{(c)} = \mathbf{U}^{\dagger} \mathbf{P}^{(r)} \mathbf{U}$.\
\
Now, basis rotations $\mathcal{R}$ of *complex* spherical harmonics can be done using the *Wigner D-matrix*: $\mathbf{P}'^{(c)} = \mathbf{D}^{(l)}(\mathcal{R}) \mathbf{P}^{(c)} [\mathbf{D}^{(l)}(\mathcal{R})]^{\dagger}$\
\
3D basis rotations can be expressed in terms of Euler angles $\mathcal{R}(\alpha, \beta, \gamma) = e^{-i\alpha J_z}e^{-i\beta J_y}e^{-i\gamma J_x}$, where $J_x, J_y, J_z$ are the angular momentum operators.
>[!NOTE]
>There are different conventions for basis rotations via Euler angles. In this script, the $Z(\alpha)-Y(\beta)-Z(\gamma)$ convention will be chosen. This means that the basis rotation from initial frame $xyz$ to final frame $XYZ$ can be done in the following order:
> 1. rotation about $z=Z$ axis by angle $\alpha$
> 2. rotation about new $Y'$ axis by angle $\beta$
> 3. rotation about new $Z''$ axis by angle $\gamma$ 
>
>For a given rotated frame $XYZ$, the Euler angles associated with the basis rotation can be obtained from:
> 
>$\alpha = \arcsin \left( \frac{Z_2}{\sqrt{1-Z_3^2}} \right)$ \
>$\beta = \arccos (Z_3)$ \
>$\gamma = \arcsin \left( \frac{Y_3}{\sqrt{1-Z_3^2}} \right)$ \
> \
> where $Y_3, Z_2, Z_3$ are components of the new axes in the initial $xyz$ frame.

In this notation, the Wigner D-matrix becomes a unitary square matrix that can be factorised into $D_{m'm}^{(l)}(\alpha, \beta, \gamma) = e^{-im'\alpha} d_{m'm}^{(l)} e^{-im\gamma}$
where $d_{m'm}^{(l)}$ are elements of the *Wigner's small d-matrix*. \
\
In summary, new orbital populations (i.e. diagonal elements of the real projection matrix) for a rotated basis are obtained by first transforming the projections onto *real* spherical harmonics $Y_{lm}$ to projections onto *complex* spherical harmonics $Y_{l}^{m}$, then rotating the basis of these using the Wigner D-matrix $\mathbf{D}^{(l)}(\mathcal{R})$, and finally transforming the new complex projection matrix back to a real projection matrix in order to obtain the new populations from the diagonal.

### Numerical
The new rotated population are calculated using the procedure outlined above on a uniform grid of `surface_resolution` sampling points for each Euler angle. Three methods are implemented for finding the optimal basis rotation.
1. **Orbital Population** \
For a given d-orbital, only the population of the rotated orbital of the same character (e.g. $|\langle Y_{l0}^{\alpha} | \phi_{n\mathbf{k}} \rangle|^2$ for $d_{z^2}$ orbitals) is considered and the optimal basis rotation is the one that maximises this new population.
2. **Population Residuals** \
For a given d-orbital, the sum of populations of the rotated orbitals except for the one with the same character (e.g. $\sum_{m \neq 0}|\langle Y_{lm}^{\alpha} | \phi_{n\mathbf{k}} \rangle|^2$ for $d_{z^2}$ orbitals) is considered and the optimal basis rotation is the one that minimises these residuals.
3. **Modified Population Residuals**\
For a given d-orbital, the difference between the residuals and the population of the orbital with the same character (e.g. $\sum_{m \neq 0}\left[|\langle Y_{lm}^{\alpha} | \phi_{n\mathbf{k}} \rangle|^2 \right]- |\langle Y_{l0}^{\alpha} | \phi_{n\mathbf{k}} \rangle|^2$ for $d_{z^2}$ orbitals) is considered and the optimal basis rotation is the one that minimises this quantity.

Note this algorith will output the global minimum/maximum of the above quantities, but often there are more than one orientation of orbitals are possible and one needs to consider a range of possible Euler angles. For this reason, the code plots the new orbital population (or one of the other quantities listed above) on a 3D surface (see example below). Here, the caveat is that one can only plot this wrt. 2 dimensions so the rotation by Euler angle $\gamma$ (i.e. rotation about final $Z$ axis) is fixed to the one obtained by minimising/maximising the chosed quantity. The 3D plot thus effectively shows the respective quantity for all possible orientations of the new $Z$ axis but for a fixed rotation about this axis.\
![Example of 3D plot generated by this code for the rotation of a $d_{z^2}$ orbital, where the plotted quantity is the orbital population (option 1)](/examples/Fe1_dz2.png)

## Usage

The only inputs needed for obtaining the optimal basis rotations are the initial orbital populations as a `numpy.array` of shape `(5,)` for the site of interest. Often these values are directtly printed to the output of the DFT code. 
>[!NOTE]
>In Vasp, the populations are written to the PROCAR file (one needs to set `LORBIT=11`) for each k-point and band seperately, so one needs to do an additional post-processing step to obtain the total valence population. The script `read_occuations.py` uses Pymatgen to parse the PROCAR file and calculate the populations summed over all k-points and valence bands.

One also needs to select an orbital of interest by setting `select_orbital` to the relevant integer value, and specify the number of sampling points by setting `surface_resolution`. 

For plotting there are several parameters that also need to be specified:
- `site_label`: this is the label of the atomic site of interest, and will be included in the figure title
- `plot_filename`: name of the PNG file for saving the image
- `viewing_orientation`: this tuple of 3 numbers (elevation, azimuth, roll) determines the orientation of the 3D plot as described in https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html.
- `cbar_location`: this sets the position of the colorbar relative to the figure `{'left', 'right', 'top', 'bottom'}`.
- `plot_quantity`: determined which of the quantities listed above to plot on the 3D surface, options are `'P'` for orbital population, `'R'` for residuals, and `'M'` for modified residuals.

### Parallelisation

Note that the computational costs scale with $N^3$ where $N$ is the number of sampling points, so it's better to start with smaller numbers first and test the scaling behaviour. If one sets `DEBUG=True` the code will write some timing information to the output.\
The main bottleneck if one uses a fine sampling grid is a triple nested for loop:
```
for nrow in range(num_points):
    for ncol in range(num_points):
        for nrot in range(num_points):
            # rotated populations are calculated here
            ...
```
but each iteration of the loop is independent and therefore easily parallelisable.
There are three versions of this code:
- `rotate_orbital_populations.py` is the standard serial version of the code. One can make use of Numpy's internal multithreading via OpenMP by setting the maximum number of threads per process (e.g. by setting `os.environ["OMP_NUM_THREADS"] = "<threads>"` inside the script), but the gains are often only minimal, since the matrices handled by Numpy are only up to $5\times5$.
- `rotate_orbital_populations_multiproc.py` uses python's multiprocessing module to distribute the outer loop (running over indices `nrow`) over available CPUs. This can lead to significant speed gains (~$10 \times$ faster for $N=500$ running on 32 cores) if one has enough cores available. Note that one needs to be careful to avoid oversubscription when setting `OMP_NUM_THREADS` to values larger than 1. One can balance the distribution over cores used by `multiprocessing` and the number of OpenMP threads for marginal speed gains, but it is generally safer to run this version on a single thread.
- `rotate_orbital_populations_numba_multiproc_.py` in addition to `multiprocessing` to distribute the outer loop over processors, the python package `numba` is used to compile the inner kernel into machine code using the `@njit` decorator, as well as some of the computation done by Numpy. This leads to significant spead gains (up to ~$100\times$ faster or more) when running on multiple cores.

Note that if one uses one of the parallel versions of the code, it is recommended to execute the script with CPU binding, e.g.:
```
srun --cpu-bind=cores python rotate_orbital_populations_numba_multiproc.py
```
in order to reduce overhead.