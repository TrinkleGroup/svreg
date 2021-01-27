import numpy as np

from ase import Atoms
from ase.build import bulk

################################################################################

r0  = 1.5
a0  = 2.5
vac = 10.0

################################################################################
# Dimers

dimer_aa = Atoms([1, 1], positions=[[0, 0, 0], [r0, 0, 0]]) 
dimer_ab = Atoms([1, 2], positions=[[0, 0, 0], [r0, 0, 0]]) 
dimer_bb = Atoms([2, 2], positions=[[0, 0, 0], [r0, 0, 0]]) 

dimer_aa.center(vacuum=vac)
dimer_ab.center(vacuum=vac)
dimer_bb.center(vacuum=vac)

dimers = {
    'aa': dimer_aa,
    'ab': dimer_ab,
    'bb': dimer_bb,
}
################################################################################
# Trimers

trimer_aaa = Atoms([1, 1, 1],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_bbb = Atoms([2, 2, 2],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_abb = Atoms([1, 2, 2],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_bab = Atoms([2, 1, 2],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_baa = Atoms([2, 1, 1],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_aba = Atoms([1, 2, 1],
    positions=[[0, 0, 0], [r0, 0, 0], [r0 / 2, np.sqrt(3) * r0 / 2, 0]]
)

trimer_aaa.center(vacuum=vac)
trimer_bbb.center(vacuum=vac)
trimer_abb.center(vacuum=vac)
trimer_bab.center(vacuum=vac)
trimer_baa.center(vacuum=vac)
trimer_aba.center(vacuum=vac)

trimers = {
    'aaa': trimer_aaa,
    'bbb': trimer_bbb,
    'abb': trimer_abb,
    'bab': trimer_bab,
    'baa': trimer_baa,
    'aba': trimer_aba,
}

################################################################################
# Bulk orthogonal in vacuum

type1 = bulk('H', crystalstructure='fcc', a=a0, orthorhombic=True)
type1 = type1.repeat((4, 4, 4))
type1.rattle()
type1.center(vacuum=vac)
type1.set_pbc(True)
type1.set_chemical_symbols(np.ones(len(type1), dtype=int))

bvo = {
    'bulk_vac_ortho_type1': type1,
}

################################################################################
_all_test_structs = {
    **dimers,
    **trimers,
    **bvo,
}