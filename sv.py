"""
TODO: need stuff for actually *building* a database

Constructors for each different type of SV
    - Should take in an atomic configuration and output an SV
    - Use a  StructureVector base class
    - Have them indexed by a global map

SVs should have an "equation" form that can be used for evaluating them
directly.

"""
from scipy.interpolate import CubicSpline
from ase.neighborlist import NeighborList

class StructureVector:
    """
    Populates a database with the given structure:

    <sv_name>
        .attrs['components'] (list):
            A list of component names (e.g. ['rho_A', 'rho_B'])

        .attrs['numParams'] (int):
            The number of fitting parameters for each component of
            this type of SV.

        .attrs['paramRanges'] (list):
            Optional list of (low, high) ranges of allowed values
            for each component. If left unspecified, uses the
            default range of (0, 1) for all components.

        <bond_type>
            .attrs['components'] (list):
                A list of components that are used for the given
                bond type. If multiple, assumes that the SV is
                constructed by computing outer products of the
                parameters for each of the components of the bond.

            <eval_type> ('energy' or 'forces')

    
    Args:
        name (str):
            The name of the structure vector type.

        bc_type (tuple):
            A tuple of (bc1, bc2), where the boundary conditions can be either
            `fixed` or `natural`. If `fixed`, then the boundary condition is
            assumed to have a fixed 1st derivative at the end-point. If
            `natural`, then the boundary condition is assumed to have a zero
            second derivative at the end-point.

        components (list):
            A list of the names of the different components of the SV.

        numParams (dict):
            A dictionary where the key is the name of a component, and the value
            is the number of fitting parameters for that component (without
            taking into account any fixed knots).

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
        self, name, components, numParams, paramRanges, bonds, cutoffs,
        numElements
        ):

        self.name = name
        self.components = components
        self.numParams = numParams
        self.paramRanges = paramRanges
        self.bonds = bonds
        self.cutoffs = cutoffs
        self.numElements = numElements


    def loop(self, structure):
        """
        Loops over the structure according to a specific summation, returning
        either energies/forces or a vector encoding the structure.
        """
        pass


class FFG(StructureVector):

    def  __init__(self, kwargs):
        StructureVector.__init__(**kwargs)

        # An FFG spline has one `f` spline for each element, and one `g` spline
        # for each unique bond type
        self.components = []
        self.fSplines = []
        self.gSplines = []
        for i in range(numElements):
            # TODO: if numParams or knot positions are ever not the same for all
            # splines, then this section is going to need to change

            self.components.append('f_{}'.format(i))
            self.fSplines.append(
                # Append a F spline
                Spline(
                    knots=np.linspace(
                        self.cutoffs[0], self.cutoffs[1], self.numParams
                    ),
                    bc_type=('natural', 'fixed'),
                )
            )

            tmpG = []

            for j in range(j, numElements):
                self.components.append('g_{}{}'.format(i, j))
                tmpG.append(
                    # Append a G spline
                    Spline(
                        knots=np.linspace(-1, 1, self.numParams),
                        bc_type=('natural', 'natural'),
                    )
                )

            self.gSplines.append(tmpG)

        
        # Pointers to the currently active splines; updated constantly in loop()
        self.fjSpline = None
        self.fkSpline = None
        self.gSpline  = None

    
    def loop(self, atoms, evalType):
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

        Returns:
            Depends upon specified `evalType`. If `energy`, returns a single
            float corresponding to the total energy of the system. If `forces`,
            returns an (N, 3) array of forces on each atom. If `vector`, returns
            a dictionary of the form {bondType: {energy/forces: vector}}.
        """

        types = sorted(list(set(atoms.get_chemical_symbols)))

        atomTypes = np.array(
            list(map(lambda s: types.index(s), atoms.get_chemical_symbols()))
        )

        N = len(atoms)

        if evalType == 'vector':
            # Prepare structure vectors
            energySV = np.zeros((N, self.numParams+2))
            forcesSV = np.zeros((3*N*N, int((self.numParams+2)**3)))
        elif evalType == 'energy':
            totalEnergy = 0.0

        # Allows double counting bonds; needed for embedding energy calculations
        nl = NeighborList(
            np.ones(N) * (self.cutoffs[-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(atoms)

        cell = atoms.get_cell()

        for i, atom in enumerate(atoms):
            itype = atomTypes[i]
            ipos = atom.position

            neighbors, offsets = nl.get_neighbors(i)

            jIdx = 0
            for j, offestj in zip(neighbors, offsets):
                # j = atomic ID, so it can be used for indexing
                jtype = atomTypes[j]

                self.fjSpline = self.fSplines[jtype]

                # offset accounts for periodic images
                jpos = atoms[j].position + np.dot(offsetj, cell)

                jvec = jpos - ipos
                rij = np.sqrt(jvec[0] ** 2 + jvec[1] ** 2 + jvec[2] ** 2)
                jvec /= rij

                # prepare for angular calculations
                a = jpos - ipos
                na = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

                # Construct a list of vectors for each component, then combine
                # for each bond

                # Since it assumed that each spline has the same number of knots
                # and that each radial spline has the same cutoffs, we don't
                # need to worry about indexing the components properly

                if evalType == 'energy':
                    fjVal = self.fjSpline(rij)
                    partialsum = 0.0
                
                jIdx += 1
                for k, offsetk in zip(neighbors[jIdx:], offsets[jIdx:]):
                    ktype = atomTypes[k]

                    self.fkSpline = self.fSplines[ktype]
                    self.gSpline  = self.gSplines[jtype][ktype] 

                    kpos = atoms[k].position + np.dot(offsetk, cell)

                    kvec = kpos - ipos
                    rik = np.sqrt(
                        kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2)
                    kvec /= rik

                    b = kpos - ipos
                    nb = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

                    cosTheta = np.dot(a, b) / na / nb

                    d0 = jvec
                    d1 = -cosTheta * jvec / rij
                    d2 = kvec / rij
                    d3 = kvec
                    d4 = -cosTheta * kvec / rik
                    d5 = jvec / rik

                    dirs = np.vstack([d0, d1, d2, d3, d4, d5])

                    if evalType == 'vector':
                        # Update structure vectors (in-place)
                        self.add_to_energy_sv(energySV, rij, rik, cosTheta, i)
                        self.forces_sv(
                            forcesSV, rij, rik, cosTheta, dirs, i, j, k
                        )
                    elif evalType == 'energy':
                        fkVal = self.fkSpline(rik)
                        gVal = self.gSpline(cosTheta)
                        partialsum += fkVal*gVal

                if evalType == 'energy':
                    totalEnergy += fjVal*partialsum


        if evalType == 'vector':
            return energySv, forcesSV

 
    def add_to_energy_sv(self, sv, rij, rik, cos, atomId):
        # Encode into energy SV
        fjCoeffs = np.sum(self.fjSpline.get_coeffs_wrapper(rij, 0), axis=0)
        fkCoeffs = np.sum(self.fkSpline.get_coeffs_wrapper(rik, 0), axis=0)
        gCoeffs  = np.sum(self.gSpline.get_coeffs_wrapper(cos, 0), axis=0))
        
        sv += np.outer(
            np.outer(fjCoeffs.ravel(), fkCoeffs.ravel()),
            gCoeffs.ravel()
        )

   
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
       

    def forces_sv(self, sv, rij, rik, cosTheta, i, j, k):

        fj_1, fk_1, g_1 = self.get_coeffs(rij, rik, cos, [1, 0, 0])
        fj_2, fk_2, g_2 = self.get_coeffs(rij, rik, cos, [0, 1, 0])
        fj_3, fk_3, g_3 = self.get_coeffs(rij, rik, cos, [0, 0, 1])

        fj, fk = self.jitted_force_mixer(
            fj_1, fk_1, g_1, fj_2, fk_2, g_2, fj_3, fk_3, g_3, dirs
        )

        # TODO: add documentation about why the SV has this 3*N*N shape

        # forcesSV = np.zeros((3*N*N, int((self.numParams+2)**3)))
        N = int(np.sqrt(sv.shape[0]//3))
        N2 = N*N
        for a in range(3):
            sv[N2*a + N*i + i, :] += fj[:, a]
            sv[N2*a + N*j + i, :] -= fj[:, a]

            sv[N2*a + N*i + i, :] += fk[:, a]
            sv[N2*a + N*k + i, :] -= fk[:, a]


    @staticmethod
    @jit(
        'Tuple((float64[:,:], float64[:,:]))(float64[:], float64[:], float64[:],float64[:], float64[:], float64[:],float64[:], float64[:], float64[:], float64[:,:])',
        nopython=True
    )
    def jitted_force_mixer(fj_1, fk_1, g_1, fj_2, fk_2, g_2, fj_3, fk_3, g_3, dirs):
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


    def setParams(self, params):
        """
        Loads in a full parameter vector, parses it for each spline, and builds
        modified scipy CubicSpline evaluators for each spline.
        """

        splits = np.cumsum(self.numParams[c] for c in self.components)[:-1]
        splitParams = np.array_split(params, splits)

        fIdx = 0
        gIdx = 0
        for y, cname in zip(splitParams, self.components):
            if 'f' in cname:
                self.fSplines[fIdx].buildDirectEvaluator(y)
                fIdx += 1
            else:
                self.gSplines[gIdx].buildDirectEvaluator(y)
                gIdx += 1


class Spline:
    """
    A helper class for embedding spline input values into vectors of spline
    coefficients.
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
        self.cutoff = (self.knots[0], self.knots[-1])

        # TODO: assumes equally-spaced knots
        self.h = self.knots[1] - self.knots[0]

    
    def in_range(self, x):
        return (x >= self.cutoff[0]) and (x <= self.cutoff[1])


    def extrap(self, x):
        """Performs linear extrapolation past the endpoints of the spline"""

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