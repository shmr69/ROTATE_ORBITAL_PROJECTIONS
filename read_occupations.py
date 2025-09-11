from pymatgen.io.vasp import Procar, Poscar, Outcar
import numpy as np

'''
uses pymatgen's POTCAR parser to read projections onto atomic orbitals for collinear spin-polarised calculations.
Note that the built-in get_occupation() method sums over all bands (including conduction), so this code includes a weighted summation over valence bands.
'''

path = '/nobackup/shmr69/Sr2Fe2O5/epitaxy/Pbcm/perpendicular/orbital_orientation/'
procar_data = Procar(path + 'PROCAR')
poscar_data = Poscar.from_file(path + 'POSCAR')
outcar_data = Outcar(path + 'OUTCAR')
num_vb = outcar_data.nelect/2 # infer occupation numbers assuming material is insulator 
print(dir(procar_data))
print(f'number of k-points: {procar_data.nkpoints}')
print(f'number of bands: {procar_data.nbands} ({num_vb:.0f} occupied)')
print(f'number of ions: {procar_data.nions}')
print(f'orbitals: {procar_data.orbitals}')

sum_proj = np.zeros(shape=(2, procar_data.nions,len(procar_data.orbitals)))
occ = np.zeros(shape=(2, procar_data.nions,len(procar_data.orbitals)))

for ns, spin in enumerate(procar_data.data): # reading projections for occupied bands
    channel = procar_data.data[spin]
    for nk in range(procar_data.nkpoints):
        for nb in range(procar_data.nbands):
            if nb+1 <= num_vb:
                sum_proj[ns] += channel[nk][nb]*procar_data.weights[nk]
    for na in range(procar_data.nions): # reading orbital occupations, but summed over all bands
        for norb, orb in enumerate(procar_data.orbitals):
            occ[ns][na][norb] = procar_data.get_occupation(na, orb)[spin]

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print()
print('valence occupations:')
for s, spin in enumerate(['up', 'down']):
    print()
    print(f'Spin {spin}:')
    print(f'     {np.array((procar_data.orbitals))}')
    line_count = 0
    for n, nat in enumerate(poscar_data.natoms):
        for i in range(nat):
            if poscar_data.site_symbols[n] == 'Fe': print(poscar_data.site_symbols[n], f': {sum_proj[s][line_count]}')
            line_count += 1

print()
print('total occupations:')
for s, spin in enumerate(['up', 'down']):
    print()
    print(f'Spin {spin}:')
    print(f'     {np.array((procar_data.orbitals))}')
    line_count = 0
    for n, nat in enumerate(poscar_data.natoms):
        for i in range(nat):
            if poscar_data.site_symbols[n] == 'Fe': print(poscar_data.site_symbols[n], f': {occ[s][line_count]}')
            line_count += 1

