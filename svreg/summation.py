import itertools
import numpy as np
from numba import jit
from scipy.sparse import diags
from scipy.interpolate import CubicSpline
from ase.neighborlist import NeighborList


class Summation:
    """
    A Summation is a representation of a summation over certain values in
    an atomic configuration (e.g. a summation over all triplets of atoms). This
    object is used for building a vector representation, as well as for
    directly performing the defined summation for a given atomic configuration.

    There are three possible outputs from the Summation class:
        1)  A vector representation of the summation; used constructing
            databases. The "energy" vector should have a shape of (N, K). The
            "forces" vector should have a shape of (3*N*N, K). Here N is the
            number of atoms in the given structure, and K is product of the
            number of fitting parameters for each component (e.g. in an "FFG"
            Summation, there are two F components and one G component).

        2) The total energy of an atomic configuration. This should have a shape
           of (1,).

        3) The atomic forces on the atoms in the system. This should have a
            shape of (3, N). TODO: in the future, when multi-component systems
            are supported, this will need to have a shape of (3, N, N) to
            account for different embedding functions on each branch of the
            tree.
    
    Attributes:
        name (str):
            The name of the structure vector type.

        components (list):
            A list of the names of the different components of the SV.

        inputTypes (dict):
            The set of allowed neighbor element types. Used for looping over
            only the specified neighbor types. key=component name, value=list of
            inputs for that component.

        numParams (dict):
            A dictionary where the key is the name of a component, and the value
            is the number of fitting parameters for that component (without
            taking into account any fixed knots).

        restrictions (dict):
            A dictionary where the key is the name of a component, and the value
            is a list of tuples of restricted knots and their values.

        paramRanges (dict):
            A dictionary where the key is the name of a component, and the value
            is a tuple of (low, high).

        bonds (dict):
            A dictionary where the key is the name of a bond type, and the value
            is a list of components that make up the bond.

        cutoffs (tuple):
            (low, high) cutoffs specifying the smallest and largest knot points.
            This is extremely important for distance-based splines, as the upper
            cutoff specifies the interatomic potential cutoff distance.

        numElements (int):
            The number of unique element types that the structure vector will be
            used for.
    """

    def __init__(
        self, name, allElements, neighborElements, components, inputTypes, numParams, restrictions,
        paramRanges, bonds, bondMapping, cutoffs, numElements, bc_type,
        ):

        self.name           = name

        if isinstance(inputTypes, dict):
            self.inputTypes = inputTypes
        elif isinstance(inputTypes, list):
            self.inputTypes = {
                c: t for c,t in zip(components, inputTypes)
            }
        elif isinstance(inputTypes, np.ndarray):
            cleanedTypes = inputTypes.tolist()
            self.inputTypes = {
                c: t.decode('utf-8') for c,t in zip(components, cleanedTypes)
            }
        else:
            raise RuntimeError(
                "inputTypes should be dict, list, or numpy.ndarray; was {}".format(
                    type(inputTypes)
                )
            )

        # Used for looping only over selected neighbor types
        self._inputTypesSet = set(itertools.chain.from_iterable(
            self.inputTypes.values()
        ))

        # By sorting these two lists, it ensures that the components can be
        # indexed in order to identify which elements single-input components
        # correspond to without having to use another data structure. Note that
        # currently 

        # TODO: shouldn't use sorting; might need to assume they're already
        # sorted here. A problem came up where the database constructor didn't
        # sort the elements alphabetically, which caused the input types to
        # with the ones here since they're sorted here

        # TODO: GA assumes sorted elements for computing errors

        # self.elements       = elements
        self.components     = sorted(components)

        self.allElements        = sorted(allElements)
        self.neighborElements   = sorted(neighborElements)

        self.numParams      = numParams
        self.restrictions   = restrictions
        self.paramRanges    = paramRanges
        self.bonds          = bonds
        self.bondMappingStr = bondMapping
        self.bondMapping    = eval(bondMapping)
        self.cutoffs        = cutoffs
        self.numElements    = numElements
        self.bc_type        = bc_type


    def loop(self, atoms, evalType, hostType=None):
        """
        Loops over the desired properties of the structure, and returns the
        specified evaluation type.

        Args:
            atoms (ase.Atoms):
                The atomic structure

            evalType (str):
                One of 'energy', 'forces', or 'vector'. 'vector' returns the
                actual vector representation of the structure vector, which is
                used for constructing databases.

            hostType (str):
                The host atom type. Used with multi-component trees when it may
                be necessary to loop over atoms only with a given type. If
                hostType is None then it loops over all atoms.

        Returns:
            Depends upon specified `evalType`. If `energy`, returns a (1, N)
            array corresponding to the per-atom energies of the system. If
            `forces`, returns an (1, N, N, 3) array of forces on each atom. If
            `vector`, returns a dictionary of the form
            {bondType: {energy/forces: vector}}.

            Note that the specific shapes of the energies and forces matrices is
            so that the derivatives can be computed properly for the embedding
            functions.
        """


        pass


class FFG(Summation):

    def  __init__(self, *args, **kwargs):
        Summation.__init__(self, *args, **kwargs)

        self.fSplines = {}
        self.gSplines = {}

        for el1 in kwargs['neighborElements']:
            self.fSplines[el1] = Spline(
                knots=np.linspace(
                    self.cutoffs[0], self.cutoffs[1],
                    self.numParams[self.components[0]]
                    + len(self.restrictions[self.components[0]]) - 2
                ),
                bc_type=('natural', 'fixed')
                if self.bc_type == 'natural'
                else ('fixed', 'fixed')
                # bc_type=('fixed', 'fixed')
            )

            for el2 in kwargs['neighborElements']:
                key = '_'.join(sorted(list(set([el1, el2]))))

                self.gSplines[key] = Spline(
                    knots=np.linspace(
                        -1, 1,
                        self.numParams[self.components[1]]
                        + len(self.restrictions[self.components[1]]) - 2
                    ),
                    bc_type=('natural', 'natural')
                    if self.bc_type == 'natural'
                    else ('fixed', 'fixed')
                    # bc_type=('fixed', 'fixed')
                )

        # Pointers to the currently active splines; updated constantly in loop()
        self.fjSpline = None
        self.fkSpline = None
        self.gSpline  = None

    
    def loop(self, atoms, evalType, hostType=None):

        totalEnergy = None
        energySV = None
        forcesSV = None
        partialsum = None
        gVal = None
        fkVal = None
        fjPrime = None
        fjVal = None
        jForces = None
        iForces = None
        forces = None

        atomTypesStrings = atoms.get_chemical_symbols()
        atomTypes = np.array(
            list(map(lambda s: self.allElements.index(s), atomTypesStrings))
        )

        N = len(atoms)
        if evalType == 'vector':
            # Prepare structure vectors
            energySV = {bondType: None for bondType in self.bonds}
            forcesSV = {bondType: None for bondType in self.bonds}

            for bondType in self.bonds:
                cartsize = np.prod([
                    self.numParams[c] + len(self.restrictions[c])
                    for c in self.bonds[bondType]
                ])

                energySV[bondType] = np.zeros((N, cartsize))
                forcesSV[bondType] = np.zeros((3*N*N, cartsize))

        elif evalType == 'energy':
            totalEnergy = np.zeros((1, N))
        elif evalType == 'forces':
            forces = np.zeros((1, N, N, 3))

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(
            np.ones(N)*(self.cutoffs[-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(atoms)

        cell = atoms.get_cell()

        for i, atom in enumerate(atoms):

            if hostType is not None:
                if atomTypesStrings[i] != hostType:
                   continue 

            ipos = atom.position

            neighbors, offsets = nl.get_neighbors(i)

            if evalType == 'forces':
                iForces = np.zeros((3,))

            jIdx = 0
            for j, offsetj in zip(neighbors, offsets):
                jIdx += 1

                # j = atomic ID, so it can be used for indexing
                jtype = atomTypes[j]
                jtypeStr = atomTypesStrings[j]

                if hostType is not None:
                    if atomTypesStrings[j] not in self._inputTypesSet:
                        continue

                self.fjSpline = self.fSplines[jtypeStr]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, cell)

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                if (evalType == 'energy') or (evalType == 'forces'):
                    fjVal = self.fjSpline(rij)
                    partialsum = 0.0
                if evalType == 'forces':
                    fjPrime = self.fjSpline(rij, 1)
                    jForces = np.zeros((3,))
               
                for k, offsetk in zip(neighbors[jIdx:], offsets[jIdx:]):
                    ktype = atomTypes[k]
                    ktypeStr = atomTypesStrings[k]

                    if hostType is not None:
                        neighTypes = set([
                            atomTypesStrings[j], atomTypesStrings[k]
                        ])

                        if neighTypes != self._inputTypesSet:
                            continue

                    self.fkSpline = self.fSplines[ktypeStr]

                    key = '_'.join(sorted(list(set([jtypeStr, ktypeStr]))))

                    self.gSpline  = self.gSplines[key] 

                    kpos = atoms[k].position + np.dot(offsetk, cell)

                    kvec = kpos - ipos
                    rik = np.sqrt(kvec[0]**2 + kvec[1]**2 + kvec[2]**2)
                    kvec /= rik

                    cosTheta = np.dot(jvec, kvec)#/rij/rik

                    d0 = jvec                       # on i due to j
                    d1 = -cosTheta * jvec / rij     # on j due to i?
                    d2 = kvec / rij                 # on i and j due to k?
                    d3 = kvec                       # on i due to k
                    d4 = -cosTheta * kvec / rik     # on k due to i?
                    d5 = jvec / rik                 # on i and k due to j?

                    dirs = np.vstack([d0, d1, d2, d3, d4, d5])

                    if evalType == 'vector':
                        bondType = self.bondMapping(jtype, ktype)

                        oldJ = j
                        oldRij = rij
                        oldFjSpline = self.fjSpline

                        if jtype != ktype:  # Then this is a cross-term
                            """
                            Since there is only one bond type for cross-terms
                            (e.g. AB, not AB and BA), we need to flip the
                            ordering of triplets that aren't in the correct
                            order. For example, we must flip a BA term so that
                            is in AB ordering.

                            In the case of forces, we must also flip the atom
                            tags and the direction vectors to make sure that the
                            results get added to the proper indices in the force
                            SV.
                            """

                            # Figure out expected order of bondType

                            # e.g. ['f_A', 'f_B', 'g_AB']
                            bondComponents = self.bonds[bondType]
                            bondInputs = None
                            for bc in bondComponents:
                                # The G spline will have the expected order
                                if 'g' in bc:
                                    # TODO: this currently assumes that the
                                    # component names are given as 'g_**'
                                    bondInputs = self.inputTypes[bc]

                            # If neighbors aren't in the correct order, swap
                            if [jtypeStr, ktypeStr] != bondInputs:
                                j = k
                                k = oldJ

                                rij = rik
                                rik = oldRij

                                dirs = np.vstack([d3, d4, d5, d0, d1, d2])

                                self.fjSpline = self.fkSpline
                                self.fkSpline = oldFjSpline 

                        # Update structure vectors (in-place)
                        self.add_to_energy_sv(
                            energySV[bondType], rij, rik, cosTheta, i
                            )

                        self.add_to_forces_sv(
                            forcesSV[bondType],
                            rij, rik, cosTheta, dirs, i, j, k
                        )

                        j = oldJ
                        rij = oldRij
                        self.fjSpline = oldFjSpline
                    elif (evalType == 'energy') or (evalType == 'forces'):
                        fkVal = self.fkSpline(rik)
                        gVal = self.gSpline(cosTheta)
                        partialsum += fkVal*gVal

                    if evalType == 'forces':

                        fkPrime = self.fkSpline(rik, 1)
                        gPrime  = self.gSpline(cosTheta, 1)

                        fij = -gVal*fkVal*fjPrime
                        fik = -gVal*fjVal*fkPrime

                        prefactor = fjVal*fkVal*gPrime

                        prefactor_ij = prefactor/rij
                        prefactor_ik = prefactor/rik

                        fij += prefactor_ij*cosTheta
                        fik += prefactor_ik*cosTheta

                        fj = jvec*fij - kvec*prefactor_ij
                        fk = kvec*fik - jvec*prefactor_ik

                        jForces += fj
                        iForces -= fk

                        forces[:, i, k, :] += fk
                # end triplet loop

                if evalType == 'energy':
                    totalEnergy[:, i] += fjVal*partialsum
                elif evalType == 'forces':
                    forces[:, i, i, :] -= jForces
                    forces[:, i, j, :] += jForces
            # end neighbor loop

            if evalType == 'forces':
                forces[:, i, i, :] += iForces
        # end atom loop

        if evalType == 'vector':
            return energySV, forcesSV
        elif evalType == 'energy':
            return totalEnergy
        elif evalType == 'forces':
            return forces

 
    def add_to_energy_sv(self, sv, rij, rik, cos, atomId):
        # Encode into energy SV
        fjCoeffs = np.sum(self.fjSpline.get_coeffs_wrapper(rij, 0), axis=0)
        fkCoeffs = np.sum(self.fkSpline.get_coeffs_wrapper(rik, 0), axis=0)
        gCoeffs  = np.sum(self.gSpline.get_coeffs_wrapper(cos, 0), axis=0)
        
        sv[atomId, :] += np.outer(np.outer(fjCoeffs, fkCoeffs), gCoeffs).ravel()

   
    def get_coeffs(self, rij, rik, cos, deriv=[0,0,0]):
        """
        A helper function for computing the coefficients for each spline,
        possibly each with different derivatives; useful for force calculations.
        """

        fjDeriv, fkDeriv, gDeriv = deriv

        fjCoeffs = self.fjSpline.get_coeffs_wrapper(np.atleast_1d(rij), fjDeriv)
        fkCoeffs = self.fkSpline.get_coeffs_wrapper(np.atleast_1d(rik), fkDeriv)
        gCoeffs  = self.gSpline.get_coeffs_wrapper(np.atleast_1d(cos),  gDeriv)

        return fjCoeffs.squeeze(), fkCoeffs.squeeze(), gCoeffs.squeeze()
       

    def add_to_forces_sv(self, sv, rij, rik, cos, dirs, i, j, k):

        fj_1, fk_1, g_1 = self.get_coeffs(rij, rik, cos, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_coeffs(rij, rik, cos, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_coeffs(rij, rik, cos, [0, 0, 1])

        fj, fk = self.jitted_force_mixer(
            fj_1, fk_1, g_1, fj_2, fk_2, g_2, fj_3, fk_3, g_3, dirs
        )

        """
        SV[i, i] is the sum of the forces on atom i due to its neighbors
        SV[i, j] is the forces on neighbor j due to atom i
        """

        N = int(np.sqrt(sv.shape[0]//3))
        for a in range(3):
            sv[3*N*i + 3*i + a, :] += fj[:, a]
            sv[3*N*i + 3*j + a, :] -= fj[:, a]

            sv[3*N*i + 3*i + a, :] += fk[:, a]
            sv[3*N*i + 3*k + a, :] -= fk[:, a]


    @staticmethod
    @jit(
        'Tuple((float64[:,:], float64[:,:]))(float64[:], float64[:], float64[:],float64[:], float64[:], float64[:],float64[:], float64[:], float64[:], float64[:,:])',
        nopython=True
    )
    def jitted_force_mixer(fj_1, fk_1, g_1, fj_2, fk_2, g_2, fj_3, fk_3, g_3, dirs):
        v1 = np.outer(np.outer(fj_1, fk_1), g_1).ravel()
        v2 = np.outer(np.outer(fj_2, fk_2), g_2).ravel()
        v3 = np.outer(np.outer(fj_3, fk_3), g_3).ravel()

        """
        When computing the derivatives of the 3-body term, the derivative must
        be taken with respect to rij _and_ rik. The forces parallel to the ij
        bond will 

        When computing forces, the following terms must be computed:
        dE/d_xi | xj and dE\d_xi | xk

        Using the chain and product rules, these derivatives result in 6 total
        terms, 3 for the ij bond and 3 for the ik bond.
        """

        # all 6 terms to be added
        t0 = dirs[0]*v1.reshape((-1, 1))
        t1 = dirs[1]*v3.reshape((-1, 1))
        t2 = dirs[2]*v3.reshape((-1, 1))

        t3 = dirs[3]*v2.reshape((-1, 1))
        t4 = dirs[4]*v3.reshape((-1, 1))
        t5 = dirs[5]*v3.reshape((-1, 1))

        # condensed versions
        fj = t0 + t1 + t2
        fk = t3 + t4 + t5

        return fj, fk


    def setParams(self, params):
        """
        Loads in a full parameter vector, parses it for each spline, and builds
        modified scipy CubicSpline evaluators for each spline.
        """

        splits = np.cumsum([
            self.numParams[c]+len(self.restrictions[c]) for c in self.components
        ])[:-1]
        splitParams = np.array_split(params, splits)

        # fCopy = list(self.fSplines[::-1])
        # gCopy = list([list(l[::-1]) for l in self.gSplines[::-1]])

        # fCopy = deepcopy(self.fSplines)
        # gCopy = deepcopy(self.gSplines)

        fIndexer  = 0
        gIndexers = [None, None]

        for y, cname in zip(splitParams, self.components):

            if isinstance(cname, bytes):
                cname = cname.decode('utf-8')
            if 'f' in cname:
                el1 = self.neighborElements[fIndexer]
                self.fSplines[el1].buildDirectEvaluator(y)

                gIndexers[fIndexer] = el1
                fIndexer += 1

                # fCopy[-1].buildDirectEvaluator(y)
                # fCopy.pop()
            else:
                el1 = gIndexers[0]
                el2 = gIndexers[1]
                if el2 is None:
                    el2 = el1

                key = '_'.join(sorted(list(set([el1, el2]))))

                self.gSplines[key].buildDirectEvaluator(y)

                # gCopy[-1][-1].buildDirectEvaluator(y)
                # gCopy[-1].pop()
                # if len(gCopy[-1]) == 0:
                #     gCopy.pop()


class Rho(Summation):
    """
    A two-body summation that loops over all neighbors for all atoms. Note that
    this summation allows double counting of bonds, meaning i->j and i<-j are
    both counted.
    """

    def  __init__(self, *args, **kwargs):
        Summation.__init__(self, *args, **kwargs)

        self.splines = {}

        # Note: a Rho Summation will only ever have one spline since it's based
        # on the neighbor type

        for el in kwargs['neighborElements']:
            self.splines[el] = Spline(
                knots=np.linspace(
                    self.cutoffs[0], self.cutoffs[1],
                    self.numParams[self.components[0]]
                    + len(self.restrictions[self.components[0]]) - 2
                ),
                bc_type=('natural', 'fixed')
                if self.bc_type == 'natural'
                else ('fixed', 'fixed')
                # bc_type=('fixed', 'fixed')
            )

        self.rho = None


    def loop(self, atoms, evalType, hostType=None):

        totalEnergy = None
        energySV = None
        forcesSV = None
        forces = None

        atomTypesStrings = atoms.get_chemical_symbols()
        atomTypes = np.array(
            list(map(lambda s: self.allElements.index(s), atomTypesStrings))
            # list(map(lambda s: types.index(s), atomTypesStrings))
        )

        N = len(atoms)
        if evalType == 'vector':

            energySV = {bondType: None for bondType in self.bonds}
            forcesSV = {bondType: None for bondType in self.bonds}

            # Prepare structure vectors
            for bondType in self.bonds:
                totalNumParams = sum([
                    self.numParams[c] + len(self.restrictions[c])
                    for c in self.bonds[bondType]
                ])

                energySV[bondType] = np.zeros((N, totalNumParams))
                # forcesSV[bondType] = np.zeros((3*N*N, totalNumParams))
                forcesSV[bondType] = np.zeros((3*N*N, totalNumParams))

        elif evalType == 'energy':
            totalEnergy = np.zeros((1, N))
        elif evalType == 'forces':
            forces = np.zeros((1, N, N, 3))

        # Note that double counting is always allowed, but it must be done
        # manually (rather than directly multiplying each bond by 2)

        nl = NeighborList(
            np.ones(N)*(self.cutoffs[-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(atoms)

        cell = atoms.get_cell()

        for i, atom in enumerate(atoms):
            itypeStr = atomTypesStrings[i]

            if hostType is not None:
                if itypeStr != hostType:
                    continue

            ipos = atom.position

            neighbors, offsets = nl.get_neighbors(i)

            for j, offsetj in zip(neighbors, offsets):
                # j = atomic ID, so it can be used for indexing
                jtype = atomTypes[j]
                jtypeStr = atomTypesStrings[j]

                if hostType is not None:
                    if atomTypesStrings[j] not in self._inputTypesSet:
                        continue

                self.rho = self.splines[jtypeStr]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, cell)

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0]**2 + jvec[1]**2 + jvec[2]**2)
                jvec /= rij

                if evalType == 'vector':

                    bondType = self.bondMapping(jtype)

                    self.add_to_energy_sv(energySV[bondType], rij, i)
                    # self.add_to_energy_sv(energySV[bondType], rij, j)

                    # TODO: it's inefficient to call add_to_forces twice since
                    # it builds the same coefficients; just add to both indices
                    # at once, like in FFG

                    # Forces acting on i
                    self.add_to_forces_sv(forcesSV[bondType], rij,  jvec, i, i)
                    self.add_to_forces_sv(forcesSV[bondType], rij,  -jvec, i, j)
                    
                    """
                    SV[bondType][i, i] is the forces on atom i due to all
                    neighbors that form the correct bondType

                    SV[bondType][i, j] is the forces on atom j due to the bond
                    of bondType with atom i
                    """


                    # # Forces acting on j
                    # self.add_to_forces_sv(forcesSV[bondType], rij, -jvec, j, i)
                    # self.add_to_forces_sv(forcesSV[bondType], rij, -jvec, j, j)

                elif evalType == 'energy':
                    # Note: i->j == i<-j iff i and j are the same element
                    # type. If they are different types, then they need to be
                    # evaluated with different Rho splines, which is why you
                    # can't simply multiply by 2 here

                    totalEnergy[:, i] += self.rho(rij)
                elif evalType == 'forces':
                    # rhoPrimeI = self.splines[itypeStr](rij, 1)
                    rhoPrimeJ = self.splines[jtypeStr](rij, 1)

                    # fcs = jvec*(rhoPrimeI + rhoPrimeJ)
                    fcs = jvec*rhoPrimeJ

                    forces[:, i, i, :] += fcs
                    forces[:, i, j, :] -= fcs

        if evalType == 'vector':
            return energySV, forcesSV
        elif evalType == 'energy':
            return totalEnergy
        elif evalType == 'forces':
            return forces


    def add_to_energy_sv(self, sv, rij, atomId):
        sv[atomId, :] += self.rho.get_coeffs_wrapper(rij).sum(axis=0)

    def add_to_forces_sv(self, sv, rij, direction, i, j):
        coeffs = self.rho.get_coeffs_wrapper(rij, 1).sum(axis=0).ravel()
        coeffs = np.einsum('i,j->ij', coeffs, direction)

        N = int(np.sqrt(sv.shape[0]//3))
        for a in range(3):
            # sv[N2*a + N*i + j, :] += coeffs[:, a]
            sv[3*N*i + 3*j + a, :] += coeffs[:, a]
            # sv[N*a + i, :] += coeffs[:, a]
            # sv[3*i + a, :] += coeffs[:, a]


    def setParams(self, params):

        splits = np.cumsum([
            self.numParams[c]+len(self.restrictions[c]) for c in self.components
        ])[:-1]
        splitParams = np.array_split(params, splits)

        # A Rho Summation will only ever have one spline
        for y, cname in zip(splitParams, self.components):
            self.splines[self.inputTypes[cname][0]].buildDirectEvaluator(y)


class Spline:
    """
    A helper class for embedding spline input values into vectors of spline
    coefficients and for performing the corresponding direct spline evaluations.
    """

    def __init__(self, knots, bc_type=None):

        if not np.all(knots[1:] > knots[:-1], axis=0):
            raise ValueError("knots must be strictly increasing")

        self.knots = np.array(knots, dtype=float)
        self.n_knots = len(knots)
        self.bc_type = bc_type
        self.index = 0

        """
        Extrapolation is done by building a spline between the end-point
        knot and a 'ghost' knot that is separated by a distance of
        extrap_dist.

        NOTE: the assumption that all extrapolation points are added at once
        is NOT needed, since get_coeffs() scales each point accordingly
        """

        self.extrap_dist = (knots[-1] - knots[0]) / 2.
        self.lhs_extrap_dist = self.extrap_dist
        self.rhs_extrap_dist = self.extrap_dist

        """
        A 'structure vector' is an array of coefficients defined by the
        Hermitian form of cubic splines that will evaluate a spline for a
        set of points when dotted with a vector of knot y-coordinates and
        boundary conditions. Some important definitions:

        M: the matrix corresponding to the system of equations for y'

        alpha: the set of coefficients corresponding to knot y-values

        beta: the set of coefficients corresponding to knot y'-values

        gamma: the result of M being row-scaled by beta

        structure vector: the summation of alpha + gamma
        """

        self.M = build_M(len(knots), knots[1] - knots[0], bc_type)

        self.cutoff = (self.knots[0], self.knots[-1])

        # TODO: assumes equally-spaced knots
        self.h = self.knots[1] - self.knots[0]

    
    def buildDirectEvaluator(self, y):
        """
        Construct a scipy.CubicSpline object for evaluating the spline directly,
        and for handling extrapolation.

        Args:
            y (np.arr):
                The full vector of fitting parameters (knot y-values, and 
                boundary conditions).
        """

        knotDerivs = self.M @ y.T
        self.d0 = knotDerivs[0]; self.dN = knotDerivs[-1]

        tmp = [self.d0, self.dN]

        bc = []
        for i, cond in enumerate(self.bc_type):
            if cond == 'natural':
                bc.append(cond)
            elif cond == 'fixed':
                bc.append((1, tmp[i]))

        self._scipy_cs = CubicSpline(self.knots, y[:-2], bc_type=bc)

    
    def in_range(self, x):
        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])


    def extrap(self, x):
        """Performs linear extrapolation past the endpoints of the spline"""

        val = 0

        if x < self.cutoff[0]:
            val = self(self.knots[0]) - self.d0*(self.knots[0]-x)
        elif x > self.cutoff[1]:
            val = self(self.knots[-1]) + self.dN*(x-self.knots[-1])

        return val


    def __call__(self, x, i=0):
        """Evaluates the spline at the given point, linearly extrapolating if
        outside of the spline cutoff. If 'i' is specified, evaluates the ith
        derivative instead. i=0 means evaluate the function, not a derivative.
        Note that `i` will only be 0 or 1.
        """

        # TODO: what if I don't want to do linear extrapolation? Could just use
        # scipy.CubicSpline evaluation for everything, which makes more sense...
        # The problems is that the SV encodings assume linear extrapolation.

        if i not in [0, 1]:
            raise RuntimeError('Only i = 0, 1 are supported')

        if x < self.cutoff[0]:
            return self.extrap(x) if i == 0 else self.d0
        elif x > self.cutoff[1]:
            return self.extrap(x) if i == 0 else self.dN
        else:
            return self._scipy_cs(x, i)


    @staticmethod
    @jit(
        'Tuple((float64[:,:], float64[:,:]))(float64[:], int64, float64, float64[:], int64)',
        nopython=True
    )
    def get_coeffs(x, deriv=0, extrap_dist=None, tmp_knots=None,
                 n_knots=None):
        """Calculates the spline coefficients for a set of points x

        Args:
            x (np.arr): list of points to be evaluated
            deriv (int): optionally compute the 1st derivative instead

        Returns:
            alpha: vector of coefficients to be added to alpha
            beta: vector of coefficients to be added to betas
            lhs_extrap: vector of coefficients to be added to lhs_extrap vector
            rhs_extrap: vector of coefficients to be added to rhs_extrap vector
        """
        x = np.atleast_1d(x)

        mn = np.min(x)
        mx = np.max(x)

        lhs_extrap_dist = max(float(extrap_dist), tmp_knots[0] - mn)
        rhs_extrap_dist = max(float(extrap_dist), mx - tmp_knots[-1])

        # add ghost knots
        lhs = tmp_knots[0] - lhs_extrap_dist
        rhs = tmp_knots[-1] + rhs_extrap_dist

        knots = np.zeros(len(tmp_knots)+2, dtype=np.float64)
        knots[0] = lhs
        knots[1:-1] = tmp_knots
        knots[-1] = rhs

        # indicates the splines that the points fall into
        spline_bins = np.digitize(x, knots, right=True) - 1

        for idx in range(spline_bins.shape[0]):
            if spline_bins[idx] < 0:
                spline_bins[idx] = 0
            elif spline_bins[idx] > len(knots) - 2:
                spline_bins[idx] = len(knots) - 2

        if (np.min(spline_bins) < 0) or (np.max(spline_bins) >  n_knots+2):
            raise ValueError(
                "Bad extrapolation; a point lies outside of the "
                "computed extrapolation range"
            )

        prefactor = knots[spline_bins + 1] - knots[spline_bins]

        t = (x - knots[spline_bins]) / prefactor
        t2 = t*t
        t3 = t2*t

        A = np.zeros(x.shape, dtype=np.float64)
        B = np.zeros(x.shape, dtype=np.float64)
        C = np.zeros(x.shape, dtype=np.float64)
        D = np.zeros(x.shape, dtype=np.float64)

        if deriv == 0:

            A = 2*t3 - 3*t2 + 1
            B = t3 - 2*t2 + t
            C = -2*t3 + 3*t2
            D = t3 - t2

        elif deriv == 1:

            A = 6*t2 - 6*t
            B = 3*t2 - 4*t + 1
            C = -6*t2 + 6*t
            D = 3*t2 - 2*t

        elif deriv == 2:

            A = 12*t - 6
            B = 6*t - 4
            C = -12*t + 6
            D = 6*t - 2
        else:
            raise ValueError("Only allowed derivative values are 0, 1, and 2")

        scaling = 1 / prefactor
        scaling = scaling**deriv

        B *= prefactor
        D *= prefactor

        A *= scaling
        B *= scaling
        C *= scaling
        D *= scaling

        alpha = np.zeros((len(x), n_knots))
        beta = np.zeros((len(x), n_knots))

        # values being extrapolated need to be indexed differently
        lhs_extrap_mask = spline_bins == 0
        rhs_extrap_mask = spline_bins == n_knots

        lhs_extrap_indices = np.arange(len(x))[lhs_extrap_mask]
        rhs_extrap_indices = np.arange(len(x))[rhs_extrap_mask]

        # if True in lhs_extrap_mask:
        if np.sum(lhs_extrap_mask) > 0:
            alpha[lhs_extrap_indices, 0] += A[lhs_extrap_mask]
            alpha[lhs_extrap_indices, 0] += C[lhs_extrap_mask]

            beta[lhs_extrap_indices, 0] += A[lhs_extrap_mask]*(-lhs_extrap_dist)
            beta[lhs_extrap_indices, 0] += B[lhs_extrap_mask]
            beta[lhs_extrap_indices, 0] += D[lhs_extrap_mask]

        # if True in rhs_extrap_mask:
        if np.sum(rhs_extrap_mask) > 0:
            alpha[rhs_extrap_indices, -1] += A[rhs_extrap_mask]
            alpha[rhs_extrap_indices, -1] += C[rhs_extrap_mask]

            beta[rhs_extrap_indices, -1] += B[rhs_extrap_mask]
            beta[rhs_extrap_indices, -1] += C[rhs_extrap_mask]*rhs_extrap_dist
            beta[rhs_extrap_indices, -1] += D[rhs_extrap_mask]

        # now add internal knots
        a = lhs_extrap_mask.shape

        internal_mask = np.logical_not(lhs_extrap_mask + rhs_extrap_mask)

        shifted_indices = spline_bins[internal_mask] - 1

        rng = np.arange(len(x))[internal_mask]

        for im, si in zip(rng, shifted_indices):
            alpha[im, si] += A[im]
            alpha[im, si+1] += C[im]

            beta[im, si] += B[im]
            beta[im, si+1] += D[im]

        return np.concatenate((alpha, np.zeros((len(x), 2))), axis=1), beta


    def get_coeffs_wrapper(self, x, deriv=0):

        alpha, beta = self.get_coeffs(
            np.atleast_1d(x), deriv=deriv, extrap_dist=self.extrap_dist,
            tmp_knots=self.knots, n_knots=self.n_knots
        )

        gamma = np.einsum('ij,ik->kij', self.M, beta.T)

        return alpha + np.sum(gamma, axis=1)


    @staticmethod
    @jit(
        'Tuple((float64[:,:], float64[:,:]))(float64[:], float64[:], float64[:],float64[:], float64[:], float64[:],float64[:], float64[:], float64[:], float64[:,:])',
        nopython=True
    )
    def ffg_abcd_mixer(fj_1, fk_1, g_1, fj_2, fk_2, g_2, fj_3, fk_3, g_3, dirs):
        v1 = np.outer(np.outer(fj_1, fk_1), g_1).ravel()
        v2 = np.outer(np.outer(fj_2, fk_2), g_2).ravel()
        v3 = np.outer(np.outer(fj_3, fk_3), g_3).ravel()

        # all 6 terms to be added
        t0 = dirs[0]*v1.reshape((-1, 1))
        t1 = dirs[1]*v3.reshape((-1, 1))
        t2 = dirs[2]*v3.reshape((-1, 1))

        t3 = dirs[3]*v2.reshape((-1, 1))
        t4 = dirs[4]*v3.reshape((-1, 1))
        t5 = dirs[5]*v3.reshape((-1, 1))

        # condensed versions
        fj = t0 + t1 + t2
        fk = t3 + t4 + t5

        return fj, fk


def build_M(num_x, dx, bc_type):
    """Builds the A and B matrices that are needed to find the function
    derivatives at all knot points. A and B come from the system of equations
    that comes from matching second derivatives at internal spline knots
    (using Hermitian cubic splines) and specifying boundary conditions

        Ap' = Bk

    where p' is the vector of derivatives for the interpolant at each knot
    point and k is the vector of parameters for the spline (y-coordinates of
    knots and second derivatives at endpoints).

    Let N be the number of knot points

    In addition to N equations from internal knots and 2 equations from boundary
    conditions, there are an additional 2 equations for requiring linear
    extrapolation outside of the spline range. Linear extrapolation is
    achieved by specifying a spline whose first derivatives match at each end
    and whose endpoints lie in a line with that derivative.

    With these specifications, A and B are both (N+2, N+2) matrices

    A's core is a tridiagonal matrix with [h''_10(1), h''_11(1)-h''_10(0),
    -h''_11(0)] on the diagonal which is dx*[2, 8, 2] based on their definitions

    B's core is a tridiagonal matrix with [-h''_00(1), h''_00(0)-h''_01(1),
    h''_01(0)] on the diagonal which is [-6, 0, 6] based on their definitions

    Note that the dx is a scaling factor defined as dx = x_k+1 - x_k, assuming
    uniform grid points, and is needed to correct for the change into the
    variable t, defined below.

    and functions h_ij are defined as:

        h_00 = (1+2t)(1-t)^2
        h_10 = t (1-t)^2
        h_01 = t^2 (3-2t)
        h_11 = t^2 (t-1)

        with t = (x-x_k)/dx

    which means that the h''_ij functions are:

        h''_00 = 12t - 6
        h''_10 = 6t - 4
        h''_01 = -12t + 6
        h''_11 = 6t - 2

    Args:
        num_x (int): the total number of knots

        dx (float): knot spacing (assuming uniform spacing)

        bc_type (tuple): tuple of 'natural' or 'fixed'

    Returns:
        M (np.arr):
            A^(-1)B
    """

    n = num_x - 2

    if n <= 0:
        raise ValueError("the number of knots must be greater than 2")

    # note that values for h''_ij(0) and h''_ij(1) are substituted in
    # TODO: add checks for non-grid x-coordinates

    bc_lhs, bc_rhs = bc_type
    bc_lhs = bc_lhs.lower()
    bc_rhs = bc_rhs.lower()

    A = np.zeros((n + 2, n + 2))
    B = np.zeros((n + 2, n + 4))

    # match 2nd deriv for internal knots
    fillA = diags(np.array([2, 8, 2]), [0, 1, 2], (n, n + 2))
    fillB = diags([-6, 0, 6], [0, 1, 2], (n, n + 2))
    A[1:n+1, :n+2] = fillA.toarray()
    B[1:n+1, :n+2] = fillB.toarray()

    # equation accounting for lhs bc
    if bc_lhs == 'natural':
        A[0,0] = -4; A[0,1] = -2
        B[0,0] = 6; B[0,1] = -6; B[0,-2] = 1
    elif bc_lhs == 'fixed':
        A[0,0] = 1/dx;
        B[0,-2] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    # equation accounting for rhs bc
    if bc_rhs == 'natural':
        A[-1,-2] = 2; A[-1,-1] = 4
        B[-1,-4] = -6; B[-1,-3] = 6; B[-1,-1] = 1
    elif bc_rhs == 'fixed':
        A[-1,-1] = 1/dx
        B[-1,-1] = 1
    else:
        raise ValueError("Invalid boundary condition. Must be 'natural' or 'fixed'")

    A *= dx

    # M = A^(-1)B
    return np.dot(np.linalg.inv(A), B)


_implemented_sums = {
    'ffg': FFG,
    'rho': Rho,
}