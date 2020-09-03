import h5py


class SVDatabase(h5py.File):
    """
    A basic wrapper to an HDF5 database that documents the expected structure
    and attributes.

    Expected format:

        database
            <structure_name>
                .attrs['natoms'] (int):
                    The number of atoms in the system.
                    k
                .attrs['energy'] (float):
                    The true energy of the structure.
                
                .attrs['forces'] (np.arr):
                    The true forces of the structure.

                <sv_name>
                    .attrs['components'] (list):
                        A list of component names (e.g. ['rho_A', 'rho_B'])

                    .attrs['numParams'] (int):
                        The number of fitting parameters for each component of
                        this type of SV.

                    .attrs['restrictions'] (list):
                        A list of tuples for each component, setting fixed
                        values for any desired knots.

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
    """

    def __init__(self, path, openType, *args, **kwargs):
        super().__init__(path, openType, *args, **kwargs)


    @classmethod
    def from_hdf5(cls, pathToFile):
        """
        Constructs a Database object from an already existing HDF5 file that
        has the expected format.

        Args:
            pathToFile (str):
                The full path to an existing HDF5 file.
        """

        return cls(pathToFile, 'r')


    @classmethod
    def from_folder(cls, pathToFolder):
        """
        Reads atoms in a folder, constructs the structure vectors, and stores
        them in the expected format.
        """

        raise NotImplementedError


    def loadTrueValues(self):
        """Return {structName: {'energy': eng, 'forces': fcs}}"""

        trueValues = {}
        for structName in self:
            eng = self[structName].attrs['energy']
            fcs = self[structName].attrs['forces']

            trueValues[structName] = {'energy': eng, 'forces': fcs}

        return trueValues