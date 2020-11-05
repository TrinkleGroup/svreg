import numpy as np
import dask.array as da


class SVDatabase(dict):
    """
    Loads all of the structure vectors into Dask arrays, storing them with
    the structure vector name as the key.

    The expected structure of the HDF5 file is as follows:

        database
            .attrs['elements'] (list):
                The element types in the database.

            .attrs['structNames'] (list):
                The list of structures represented in the database. Used for
                parsing results.

            .attrs['natoms'] (np.arr):
                The number of atoms in each structure, matching the order of
                .attrs['structNames']. Used for parsing results.

            <evalType> ('energy' or 'forces')
                .attrs['natom_splits'] (np.arr):
                    An array that can be used to split axis 0 by structure.

                .attrs['nelem_splits'] (np.arr):
                    A 2D array, where each row can be used to split a 
                    per-structure group of entries by host element type.

                <bondType> (Dataset):
                    The stacked structure vectors of a given type for all
                    structures. Should be parsed by first splitting with
                    .attrs['natom_splits'] then with .attrs['nelem_splits'].

            <'trueValues'>
                <structName>
                    <evalType> ('energy' or 'forces')
    """

    def __init__(self, h5pyFile):
        # Prepare class variables
        self['energy'] = {}
        self['forces'] = {}

        self.attrs['elements'] = h5pyFile.attrs['elements']
        self.attrs['elements'] = np.char.decode(
            self.attrs['elements'].astype(np.bytes_)
        )

        self.attrs = {
            'elements': None,
            'structNames': None,
            'natoms': None,
            'energy': {
                el: {'natom_splits': None, 'nelem_splits': None}
                for el in self.attrs['elements']
            },
            'forces': {
                el: {'natom_splits': None, 'nelem_splits': None}
                for el in self.attrs['elements']
            },
        }

        self.attrs['natoms']        = h5pyFile.attrs['natoms']

        self.attrs['structNames']   = h5pyFile.attrs['structNames']
        self.attrs['structNames'] = np.char.decode(
            self.attrs['structNames'].astype(np.bytes_)
        )

        self.trueValues = {}

        # Load data from file
        for evalType in h5pyFile:
            if evalType == 'trueValues': continue

            for bondType in h5pyFile[evalType]:
                self.attrs[bondType] = {}

                for k, v in h5pyFile[evalType][bondType].attrs.items():
                    self.attrs[bondType][k] = v

                group = h5pyFile[evalType][bondType]

                for elem in self.attrs['elements']:

                    natom_splits = group[elem].attrs['natom_splits']
                    nelem_splits = group[elem].attrs['nelem_splits']

                    self.attrs[evalType][elem]['natom_splits'] = np.array(
                        # natom_splits[:-1], dtype=int
                        natom_splits, dtype=int
                    )

                    self.attrs[evalType][elem]['nelem_splits'] = np.array(
                        # nelem_splits[:, :-1], dtype=int
                        nelem_splits, dtype=int
                    )


                    self[evalType][bondType][elem] = da.from_array(
                        group[elem][()],
                        chunks=(1000, 1000)
                    )

        # Load true values
        for structName in h5pyFile['trueValues']:
            self.trueValues[structName] = {}

            for evalType in ['energy', 'forces']:
                self.trueValues[structName][evalType] =\
                    h5pyFile['trueValues'][structName][evalType]

