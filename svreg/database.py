import h5py
import numpy as np


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
        elements = sorted(list(h5pyFile[structNames[0]][svNames[0]].keys()))

        self.trueValues = {}
        for struct in structNames:
            self.trueValues[struct] = {
                'energy': h5pyFile[struct].attrs['energy'],
            }

            self.trueValues[struct]['forces'] = h5pyFile[struct].attrs['forces']


        self.attrs = {
            'structNames': structNames,
            'svNames': svNames,
            'elements': elements,
            'natoms': {s: h5pyFile[s].attrs['natoms'] for s in structNames},
        }


    def load(self, h5pyFile):

        structNames = self.attrs['structNames']
        svNames = self.attrs['svNames']
        elements = self.attrs['elements']

        print("Loading {} structures...".format(len(structNames)), flush=True)

        for struct in structNames:
            self[struct] = {}
            for sv in svNames:
                self[struct][sv] = {}
                self.attrs[sv] = {}

                # components, restrictions, etc.
                for k, v in h5pyFile[struct][sv].attrs.items():
                    self.attrs[sv][k] = v

                for elem in elements:
                    # TODO: this is a placeholder so that the regressor knows
                    # structures there are

                    self[struct][sv][elem] = None


def worker_load(h5pyFileName, localNames, svNames, elements):

    from dask.distributed import get_worker

    worker = get_worker()

    worker._structures = {}
    worker._true_forces = {}

    with h5py.File(h5pyFileName, 'r') as h5pyFile:
        for struct in localNames:
            worker._structures[struct] = {}
            for sv in svNames:
                worker._structures[struct][sv] = {}

                for elem in elements:

                    worker._structures[struct][sv][elem] = {}

                    group = h5pyFile[struct][sv][elem]

                    energyData = np.array(group['energy'][()], dtype=np.float32)
                    forcesData = np.array(group['forces'][()], dtype=np.float32)

                    worker._structures[struct][sv][elem]['energy'] = energyData
                    worker._structures[struct][sv][elem]['forces'] = forcesData
                    
            tvF = h5pyFile[struct].attrs['forces']
            worker._true_forces[struct] = np.array(tvF, dtype=np.float32)
