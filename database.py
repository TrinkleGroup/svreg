import numpy as np
import dask.array as da
from dask.distributed import wait


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

                <bondType> (Dataset):
                    The stacked structure vectors of a given type for all
                    structures. Should be parsed by splitting with
                    .attrs['natom_splits'].

            <'trueValues'>
                <structName>
                    <evalType> ('energy' or 'forces')
    """

    def __init__(self, client, h5pyFile):
        # Prepare class variables
        self['energy'] = {}
        self['forces'] = {}

        elements = np.char.decode(
            h5pyFile.attrs['elements'].astype(np.bytes_)
        )

        self.attrs = {
            'elements': elements,
            'structNames': None,
            'natoms': None,
            'energy': {
                el: {'natom_splits': None}
                for el in elements
            },
            'forces': {
                el: {'natom_splits': None}
                for el in elements
            },
        }

        self.attrs['natoms']        = h5pyFile.attrs['natoms']

        self.attrs['structNames']   = h5pyFile.attrs['structNames']
        self.attrs['structNames'] = np.char.decode(
            self.attrs['structNames'].astype(np.bytes_)
        )

        self.trueValues = {}

        thingsToPersist = []

        import time
        start = time.time()

        # Load data from file
        for evalType in h5pyFile:
            if evalType == 'trueValues': continue

            for bondType in h5pyFile[evalType]:
                self[evalType][bondType] = {}
                self.attrs[bondType] = {}

                for k, v in h5pyFile[evalType][bondType].attrs.items():
                    self.attrs[bondType][k] = v

                group = h5pyFile[evalType][bondType]

                for elem in self.attrs['elements']:

                    natom_splits = group[elem].attrs['natom_splits']

                    self.attrs[evalType][elem]['natom_splits'] = np.array(
                        # natom_splits[:-1], dtype=int
                        natom_splits, dtype=int
                    )

                    print("Database loading:", evalType, bondType, elem,
                            group[elem].shape, group[elem].dtype, flush=True)

                    # This makes data get loaded onto Client task first, then
                    # sent out; faster than parallel reads I think
                    data = group[elem][()]

                    # if evalType == 'energy':
                    #     self[evalType][bondType][elem] = group[elem][()]
                    # elif evalType == 'forces':
                    self[evalType][bondType][elem] = da.from_array(
                        data,
                        # group[elem],
                        # chunks=group[elem].shape,
                        chunks=(10000, group[elem].shape[1]),
                        # name='sv-{}-{}-{}'.format(evalType, bondType, elem),
                    ).persist()
                    thingsToPersist.append(self[evalType][bondType][elem])
        print("Full load time: {} (s)".format(time.time() - start))

        wait(thingsToPersist)

        # Load true values
        for structName in h5pyFile['trueValues']:
            self.trueValues[structName] = {}

            for evalType in ['energy', 'forces']:
                self.trueValues[structName][evalType] =\
                    h5pyFile['trueValues'][structName][evalType][()]

