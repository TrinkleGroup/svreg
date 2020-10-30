import os
import h5py
import glob

from ase.io.xyz import read_xyz


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

                    <eval_type> ('energy' or 'forces')
    """

    def __init__(self, path, openType, *args, **kwargs):
        super().__init__(path, openType, *args, **kwargs)
        self.structureVectors = []


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


    def buildDatabase(self, pathToFolder):
        """
        Constructs the database using a folder of atomic structures.

        Args:
            pathToFolder (str):
                The name of the folder containing the atomic structures. It is
                assumed that the atomic structures are stored in XYZ (or 
                extended XYZ) format as specified in the ASE documentation.

                ASE extended XYZ format:
                https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#xyz
        """

        if len(self.structureVectors) < 1:
            raise RuntimeError('Must define structure vectors first.')

        for fileName in glob.glob(os.path.join(pathToFolder, '*.xyz')):
            atoms = read_xyz(fileName)
            shortName = os.path.split(fileName)[-1]
            shortName = os.path.splitext(shortName)[0]

            self.addStructure(shortName, atoms)

    
    def addStructure(self, name, atoms):

        newGroup = self.create_group(name)
        newGroup.attrs['natoms'] = len(atoms)

        for sv in self.structureVectors:
            # Assumes `sv` is a fully-prepared StructureVector object

            vectors = sv.loop(atoms, 'vector')

            for bondType in vectors:
                energyData = newGroup[sv.name][bondType]['energy']
                energyData.resize((1,) + vectors[bondType]['energy'].shape)
                energyData[:] = vectors[bondType]['energy']

                forcesData = newGroup[sv.name][bondType]['forces']
                forcesData.resize((1,) + vectors[bondType]['forces'].shape)
                forcesData[:] = vectors[bondType]['forces']

