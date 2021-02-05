import numpy as np

from ase import Atoms
from ase.build import bulk

################################################################################

r0  = 2.0
a0  = 2.5
vac = 100.0

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
# Symmetric trimers

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
# Right-angle trimers (for non-zero forces)

trimer_aaa = Atoms([1, 1, 1],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_bbb = Atoms([2, 2, 2],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_abb = Atoms([1, 2, 2],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_bab = Atoms([2, 1, 2],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_baa = Atoms([2, 1, 1],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_aba = Atoms([1, 2, 1],
    positions=[[0, 0, 0], [1.5*r0, 0, 0], [0, 1.5*r0, 0]]
)

trimer_aaa.center(vacuum=vac)
trimer_bbb.center(vacuum=vac)
trimer_abb.center(vacuum=vac)
trimer_bab.center(vacuum=vac)
trimer_baa.center(vacuum=vac)
trimer_aba.center(vacuum=vac)

trimers_rangle = {
    'aaa_rangle': trimer_aaa,
    'bbb_rangle': trimer_bbb,
    'abb_rangle': trimer_abb,
    'bab_rangle': trimer_bab,
    'baa_rangle': trimer_baa,
    'aba_rangle': trimer_aba,
}


################################################################################
# Linear trimers (for non-zero forces in opposite directions)

trimer_aaa = Atoms([1, 1, 1],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_bbb = Atoms([2, 2, 2],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_abb = Atoms([1, 2, 2],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_bab = Atoms([2, 1, 2],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_baa = Atoms([2, 1, 1],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_aba = Atoms([1, 2, 1],
    positions=[[-1.2*r0, 0, 0], [0, 0, 0], [1.2*r0, 0, 0]]
)

trimer_aaa.center(vacuum=vac)
trimer_bbb.center(vacuum=vac)
trimer_abb.center(vacuum=vac)
trimer_bab.center(vacuum=vac)
trimer_baa.center(vacuum=vac)
trimer_aba.center(vacuum=vac)

trimers_linear = {
    'aaa_lin': trimer_aaa,
    'bbb_lin': trimer_bbb,
    'abb_lin': trimer_abb,
    'bab_lin': trimer_bab,
    'baa_lin': trimer_baa,
    'aba_lin': trimer_aba,
}

################################################################################
# Asymmetric trimers (9-13-14 scalene triangle)

y = 108/28
x = np.sqrt(81 - y*y)

trimer_aaa = Atoms([1, 1, 1],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_bbb = Atoms([2, 2, 2],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_abb = Atoms([1, 2, 2],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_bab = Atoms([2, 1, 2],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_baa = Atoms([2, 1, 1],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_aba = Atoms([1, 2, 1],
    positions=[[0, 0, 0], [14*r0, 0, 0], [(14-y)*r0, x*r0, 0]]
)

trimer_aaa.center(vacuum=vac)
trimer_bbb.center(vacuum=vac)
trimer_abb.center(vacuum=vac)
trimer_bab.center(vacuum=vac)
trimer_baa.center(vacuum=vac)
trimer_aba.center(vacuum=vac)

trimers_asym = {
    'aaa_asym': trimer_aaa,
    'bbb_asym': trimer_bbb,
    'abb_asym': trimer_abb,
    'bab_asym': trimer_bab,
    'baa_asym': trimer_baa,
    'aba_asym': trimer_aba,
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
    **trimers_rangle,
    **trimers_linear,
    **trimers_asym,
    **bvo,
}