import h5py
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

                <bondType> (Dataset):
                    The stacked structure vectors of a given type for all
                    structures. Should be parsed by splitting with
                    .attrs['natom_splits'].

            <'trueValues'>
                <structName>
                    <evalType> ('energy' or 'forces')
    """

    def __init__(self, h5pyFile, refStruct, namesFile=None):
        # Prepare class variables
        # self['energy'] = {}
        # self['forces'] = {}

        if namesFile is None:
            structNames = sorted(list(h5pyFile.keys()))
        else:
            with open(namesFile, 'r') as nf:
                structNames = [l.strip() for l in nf.readlines()]

            if refStruct not in structNames:
                structNames.append(refStruct)

        structNames = sorted(structNames)
                
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


    def load(self, h5pyFile, useDask=True):

        structNames = self.attrs['structNames']
        svNames = self.attrs['svNames']
        elements = self.attrs['elements']

        print("Loading {} structures...".format(len(structNames)), flush=True)

        self.splits = {}

        for sv in svNames:
            self.attrs[sv] = {}

            # components, restrictions, etc.
            for k, v in h5pyFile[structNames[0]][sv].attrs.items():
                self.attrs[sv][k] = v

            self[sv] = {}
            self.splits[sv] = {}

            for elem in elements:

                continue

                bigSVE = [
                    h5pyFile[struct][sv][elem]['energy'] for struct in structNames
                ]

                bigSVF = [
                    h5pyFile[struct][sv][elem]['forces'] for struct in structNames
                ]

                splits = np.cumsum([sve.shape[0] for sve in bigSVE])

                splits = np.concatenate([[0], splits])
                self.splits[sv][elem] = splits

                self[sv][elem] = {}

                if useDask:
                    self[sv][elem]['energy'] = [np.array(sve) for sve in bigSVE]
                    # self[sv][elem]['forces'] = da.from_array(np.concatenate(bigSVF, axis=0))
                    self[sv][elem]['forces'] = [da.from_array(np.array(svf)) for svf in bigSVF]
                else:
                    # self[sv][elem]['energy'] = np.concatenate(bigSVE, axis=0)
                    # self[sv][elem]['forces'] = np.concatenate(bigSVF, axis=0)

                    self[sv][elem]['energy'] = bigSVE
                    self[sv][elem]['forces'] = bigSVF


def worker_load(h5pyFileName, localNames, svNames, elements, allSums):

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

                    if (allSums) and (len(forcesData.shape) == 4):
                        forcesData = forcesData.sum(axis=0)

                    worker._structures[struct][sv][elem]['energy'] = energyData
                    worker._structures[struct][sv][elem]['forces'] = forcesData

            tvF = h5pyFile[struct].attrs['forces']
            worker._true_forces[struct] = np.array(tvF, dtype=np.float32)