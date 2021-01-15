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

    def __init__(self, h5pyFile):
        # Prepare class variables
        self['energy'] = {}
        self['forces'] = {}

        structNames = list(h5pyFile.keys())
        svNames =  list(h5pyFile[structNames[0]].keys())
        elements = list(h5pyFile[structNames[0]][svNames[0]].keys())

        self.trueValues = {}
        for struct in structNames:
            self.trueValues[struct] = {
                'energy': h5pyFile[struct].attrs['energy'],
            }

            for el in elements:
                fname = 'forces_' + el
                self.trueValues[struct][fname] = h5pyFile[struct].attrs[fname]

        self.attrs = {
            'structNames': structNames,
            'svNames': svNames,
            'elements': elements,
            'natoms': {s: h5pyFile[s].attrs['natoms'] for s in structNames},
        }

        # self.attrs['structNames'] = np.char.decode(
        #     self.attrs['structNames'].astype(np.bytes_)
        # )

        import time
        start = time.time()
        elements = list(h5pyFile[structNames[0]][svNames[0]].keys())

        futures = []
        for struct in structNames:
            self[struct] = {}
            for sv in svNames:
                self[struct][sv] = {}
                self.attrs[sv] = {}

                # components, restrictions, etc.
                for k, v in h5pyFile[struct][sv].attrs.items():
                    self.attrs[sv][k] = v

                for elem in elements:

                    self[struct][sv][elem] = {}

                    group = h5pyFile[struct][sv][elem]
                    forceData = group['forces']

                    self[struct][sv][elem]['energy'] = group['energy'][()]

                    # self[struct][sv][elem]['energy'] = da.from_array(
                    #     group['energy'][()],
                    #     chunks=group['energy'][()].shape
                    # ).persist()

                    self[struct][sv][elem]['forces'] = da.from_array(
                        forceData,
                        # chunks=(5000, forceData.shape[1]),
                        chunks=forceData.shape
                    ).persist()

                    futures.append(self[struct][sv][elem]['energy'])
                    futures.append(self[struct][sv][elem]['forces'])

        wait(futures)
        print("Full load time: {} (s)".format(time.time() - start))
