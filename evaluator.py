"""
An SVEvaluator object is designed to manage the distributed evaluation of
multiple structure vectors for multiple different atomic structures.

The currently-implemented MPI data flow can be described as follows:
    - A master process passes a population of parameters to an Evaluator.
    - The Evaluator transmits the population to each Manager.
    - Each Manager splits the population across its farm of compute nodes.
    - Compute nodes split the sub-population across their physical cores.
    - Physical cores evaluate their sub-sub-population on the structure
        vectors for the structures of their corresponding Manager.
    - Results are passed back from core->node->Manager->Evaluator->master.

This logic may be changed in the future if there's a strong reason to need
a different type of parallel structure for some reason.
"""

import numpy as np
from mpi4py import MPI

# store databases at the module level to enable memory sharing
database = {}
natoms = {}


class Manager:
    """
    Manages farms of physical compute nodes to compute results on a subset of
    the full database.
    """

    def __init__(self, id, rank, comm, structures):
        self.id = id
        self.rank = rank
        self.comm = comm
        self.structures = structures

        self.isHead = (self.rank == 0)
        self.size = self.comm.Get_size()

    
    def evaluate(self, population, evalType):
        """
        Evaluates the population by splitting it across the farm of processors.

        Args:
            population (dict):
                {svName: list of populations, each maybe of different shapes}

            evalType (str):
                'energy' or 'forces'

        Returns:
            values (dict):
                {structName: {svName: list of results}}. Since each SVNode
                may not necessarily have the same number of each evaluations for
                each svName, the results need to be kept as lists.
        """

        population = self.comm.bcast(population, root=0)

        values = {}
        for structName in database:
            values[structName] = {}
            for svName in database[structName]:
                n = int(natoms[structName])
                listOfPops = population[svName]
                values[structName][svName] = []
                for pop in listOfPops:
                    # The current model sums over bond types
                    intermediates = []
                    for bondType in database[structName][svName]:
                        sv = database[structName][svName][bondType][evalType]
                        val = (sv @ pop.T).T

                        if evalType == 'energy':
                            val = val.sum(axis=1)/n

                        if evalType == 'forces':
                            val = val.reshape(pop.shape[0], 3, n, n)
                            val = val.sum(axis=-1).swapaxes(1, 2)

                        intermediates.append(val)
                    
                    values[structName][svName].append(sum(intermediates))

        return values


    def loadDatabase(self, h5pyFile):
        """
        Takes in `database`, an HDF5 file and loads all structure vectors for
        each structure in `structNames`.

        Args:
            h5pyFile (Database):
                The database that stores the structure vectors. Formatted as
                specified in the Database class.
        """

        for struct in self.structures:
            structGroup = h5pyFile[struct]
            for svName in structGroup:
                # Ideally, map N-D bond-types into a 1D index
                for bondType in structGroup[svName]:
                    path = [struct, svName, bondType]

                    self.loadToSharedMemory(path, h5pyFile)


    def loadToSharedMemory(self, path, h5pyFile):
        """
        Loads the data sets from the given path into shared memory.

        Args:
            path (list):
                A list of keys for the database.

            h5pyFile (Database):
                The database

        Return:
            None. Populates Manager-level database with structure vectors. Note
            that the shared-memory database follows the same structure as the
            provided HDF5 database.
        """

        mpiDoubleSize = MPI.DOUBLE.Get_size()

        if self.isHead:
            # Have the head node figure out how much memory to allocate
            fullKey = '/'.join(path)

            eng = h5pyFile[fullKey]['energy'][()]
            fcs = h5pyFile[fullKey]['forces'][()]

            engShape = eng.shape
            fcsShape = fcs.shape

            engBytes = np.prod(engShape)*mpiDoubleSize
            fcsBytes = np.prod(fcsShape)*mpiDoubleSize
        else:
            # Everyone else doesn't need to allocate anything
            engShape = None
            fcsShape = None
            engBytes = 0
            fcsBytes = 0

        engShape = self.comm.bcast(engShape, root=0)
        fcsShape = self.comm.bcast(fcsShape, root=0)

        # Have everyone open windows to the shared memory
        engShmemWin = MPI.Win.Allocate_shared(
            engBytes, mpiDoubleSize, comm=self.comm
        )

        fcsShmemWin = MPI.Win.Allocate_shared(
            fcsBytes, mpiDoubleSize, comm=self.comm
        )

        # Get buffers in the shared memory to fill; convert to Numpy buffers
        engBuf, _ = engShmemWin.Shared_query(0)
        fcsBuf, _ = fcsShmemWin.Shared_query(0)

        engShmemArr = np.ndarray(buffer=engBuf, dtype='d', shape=engShape)
        fcsShmemArr = np.ndarray(buffer=fcsBuf, dtype='d', shape=fcsShape)

        # Add natoms to shmem too
        natomsBytes = mpiDoubleSize  # should be enough
        natomsShmemWin = MPI.Win.Allocate_shared(
            natomsBytes, mpiDoubleSize, comm=self.comm
        )
        natomsBuf, _ = natomsShmemWin.Shared_query(0)
        natomsShmemArr = np.ndarray(buffer=natomsBuf, dtype='d', shape=(1,))

        # Iteratively build Manager-level database, then insert buffer
        pointer = database
        for i in range(len(path)):
            # pointer[path[i]] = {}
            if path[i] in pointer:
                pointer = pointer[path[i]]
            else:
                pointer[path[i]] = {}
                pointer = pointer[path[i]]

        pointer['energy'] = engShmemArr
        pointer['forces'] = fcsShmemArr
        natoms[path[0]] = natomsShmemArr

        # Finally, populate buffer
        if self.isHead:
            pointer['energy'][...] = eng
            pointer['forces'][...] = fcs
            natoms[path[0]][...] = h5pyFile[path[0]].attrs['natoms']


class SVEvaluator:
    """
    Attributes:

        comm (MPI.Communicator):
            The MPI communicator for processes that will handle SV evaluation.
            This will usually be MPI.WORLD_COMM.

        structNames (list):
            The list of keys corresponding to the names of the structures
            to use during evaluation.

        settings (Settings):
            A dictionary of settings for the entire simulation. Takes in the
            full dictionary, but will only search for a few expected keys.

            Current expected key-value pairs:
                PROCS_PER_MANAGER (int):
                    The number of MPI tasks to allocate to each Manager.
                
                PROCS_PER_PHYS_NODE (int):
                    The number of cores on a physical compute node. The number
                    of physical compute nodes assigned to each Manager is equal
                    to (PROCS_PER_MANAGER / PROCS_PER_PHYS_NODE).
    """

    def __init__(self, comm, structNames, settings):
        self.comm = comm
        self.rank = self.comm.Get_rank()

        # Recall that this is being run on every MPI rank, so we need to check
        # if we're the master process or not
        self.isMaster = (self.rank == 0)

        self.structNames = structNames
        self.settings = settings

        # It's important to know if we're in charge of a Manager object
        self.isManager = ((self.rank % settings['PROCS_PER_MANAGER']) == 0)

        # Build Manager object
        managerId = self.rank // settings['PROCS_PER_MANAGER']
        localRank = self.rank  % settings['PROCS_PER_MANAGER']

        # Build a communicator between master process and Manager heads
        self.managerComm = self.buildManagerComm()

        managerStructs = np.array_split(
            self.structNames,
            self.comm.Get_size() // settings['PROCS_PER_MANAGER']
        )[managerId]

        # Build a communicator between Manager head and its farm of processors
        localComm = MPI.Comm.Split(comm, managerId, self.rank)

        self.manager = Manager(managerId, localRank, localComm, managerStructs)


    def buildManagerComm(self):
        """Build a communicator between each Manager"""
        managerRanks = np.arange(
            0, self.comm.Get_size(), self.settings['PROCS_PER_MANAGER']
        )

        worldGroup = self.comm.Get_group()
        managerGroup = worldGroup.Incl(managerRanks)

        return self.comm.Create(managerGroup)

   
    def distributeDatabase(self, h5pyFile):
        """
        Distribute the contents of an HDF5-style database across compute nodes.

        Args:
            h5pyFile (h5py.File):
                The database of structure vectors.
        """

        self.manager.loadDatabase(h5pyFile)


    def evaluate(self, population, evalType):
        """
        Evaluate the structure vectors by transmitting the population to each
        Manager, then having each Manager split the population to its farm of
        processes.

        Args:
            population (dict):
                {svName: list of populations, each maybe of different shapes}

            evalType (str):
                'energy' or 'forces'

        Return:
            resuls (dict):
                {structName: {svName: <result array>}}
        """

        if self.isManager:
            population = self.managerComm.bcast(population, root=0)

        return self.manager.evaluate(population, evalType)