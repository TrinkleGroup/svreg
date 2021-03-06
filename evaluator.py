"""
An SVEvaluator object is designed to manage the distributed evaluation of
multiple structure vectors for multiple different atomic structures.

A Manager object manages farms of physical compute nodes to compute results on a
subset of the full database.

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

    def __init__(self, id, farmRank, farmComm, structures, ppn):
        self.id = id
        self.farmRank = farmRank
        self.farmComm = farmComm
        self.structures = structures
        self.ppn = ppn

        self.isHead = (self.farmRank == 0)
        self.numWorkers = self.farmComm.Get_size()

        self.nodeId   = self.farmRank // self.ppn
        self.nodeRank = self.farmRank %  self.ppn

        self.isNodeHead = (self.nodeRank == 0)

        self.numNodes = self.numWorkers // self.ppn

        # Build a communicator between Manager and node heads
        nodeHeadRanks = np.arange(0, self.farmComm.Get_size(), self.ppn)

        farmGroup = self.farmComm.Get_group()
        nodeHeadGroup = farmGroup.Incl(nodeHeadRanks)

        self.nodeHeadComm = self.farmComm.Create(nodeHeadGroup)

        # Build a communicator between node heads and their physical processors
        self.physComm = MPI.Comm.Split(farmComm, self.nodeId, self.nodeRank)
        self.physSize = self.physComm.Get_size()

   
    # @profile
    def evaluate(self, population, evalType):
        """
        Evaluates the population by splitting it across the farm of processors.
        The Manager head first scatter the population across its node heads,
        who then scatter their splits to their physical processors.

        Args:
            population (dict):
                {svName: {bondType: list of populations for trees}}

            evalType (str):
                'energy' or 'forces'

        Returns:
            managerValues (dict):
                {structName: {svName: list of results}}. Since each SVNode
                may not necessarily have the same number of each evaluations for
                each svName, the results need to be kept as lists.
        """

        if self.isHead:
            # Group the populations, but track how to un-group them for later
            splits = {}
            batchedPopulations = {}
            for svName in population.keys():
                batchedPopulations[svName] = {}
                for bondType, listOfPops in population[svName].items():
                    splitIndices = np.cumsum(
                        [pop.shape[0] for pop in listOfPops]
                    )[:-1]

                    # Record how to un-group the populations for each tree
                    # Note: don't need to save for all bondTypes because they
                    # should all have the same dimensions
                    splits[svName] = splitIndices
                    
                    if len(listOfPops) > 0:
                        # Split the full population across the workers
                        batchedPopulations[svName][bondType] = np.array_split(
                            np.vstack(listOfPops), self.numWorkers
                        )
                    else:
                        batchedPopulations[svName][bondType] = \
                             [None]*self.numWorkers


            # Prepare the split populations for MPI scatter()
            localPopulations = [
                {
                    svName: {
                        bondType: batchedPopulations[svName][bondType][i]
                        for bondType in population[svName]
                    }
                    for svName in population
                }
                for i in range(self.numWorkers)
            ]
            
        else:
            localPopulations = None

        # Scatter the population to each node head
        localPop = self.farmComm.scatter(localPopulations, root=0)

        # TODO: each node head should open a shared memory window for a results
        # buffer, that way we can avoid in-node MPI calls, and the manager can
        # just gather the buffers from each node head. I'm going to ignore this
        # now, just because I'm not hitting communication limits yet.

        localValues = {}
        for structName in database:  # Loop over all locally-loaded structures
            n = int(natoms[structName])
            localValues[structName] = {}

            for svName in database[structName]:
                intermediates = []  # for summing over bond types
                for bondType in database[structName][svName]:
                    if localPop[svName][bondType] is None:
                        # Possible if a tree doesn't use a given SV;
                        # localPop[svName] will have [bondType] entries, so it
                        # won't be None
                        continue

                    sv = database[structName][svName][bondType][evalType]
                    val = (sv @ localPop[svName][bondType].T).T

                    if evalType == 'energy':
                        # Will convert to per-atom energies in __main__.py
                        val = val.sum(axis=1)#/n

                    elif evalType == 'forces':
                        # TODO: nodemanager had to apply U' because the
                        # embedding function could be different for each atom
                        # type. Since the embedding functions are now just
                        # simple algebraic functions, and are constant across
                        # all bond types, this can be summed safely here. In
                        # fact, the ffg splines don't need their 4th dimension.
                        # Make sure to address this in summation.py when you
                        # the functions for constructing splines.

                        # TODO: the above is no longer true. In order to account
                        # for different trees for each bond type, the SVs must
                        # keep their extra dimension to allow for different U'
                        # scaling for each atom type

                        val = val.reshape(
                            localPop[svName][bondType].shape[0], 3, n, n
                        )
                        val = val.sum(axis=-1).swapaxes(1, 2)

                    intermediates.append(val)

                if len(intermediates) > 0:
                    localValues[structName][svName] = sum(intermediates)
                else:
                    localValues[structName][svName] = None

        workerValues = self.farmComm.gather(localValues, root=0)

        # Gather results on head
        if self.isHead:
            # workerValues = [{structName: {svName: sub-population}}]
            managerValues = {structName: {} for structName in database}
            for structName in database:
                for svName in database[structName]:
                    if workerValues[0][structName][svName] is None:
                        continue

                    # Stack over worker results, then split by tree pop sizes
                    managerValues[structName][svName] = np.split(
                        np.concatenate([
                            v[structName][svName] for v in workerValues
                        ]),
                        splits[svName]
                    )

        else:
            managerValues = None

        return managerValues


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

        # mpiDoubleSize = MPI.FLOAT.Get_size()
        mpiDoubleSize = MPI.DOUBLE.Get_size()

        if self.isNodeHead:
            # Have all node heads figure out how much memory to allocate
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

        # ALL processes in the Manager need to know the shape
        engShape = self.farmComm.bcast(engShape, root=0)
        fcsShape = self.farmComm.bcast(fcsShape, root=0)

        # Open shared memory windows on each physical node
        engShmemWin = MPI.Win.Allocate_shared(
            engBytes, mpiDoubleSize, comm=self.physComm
        )

        fcsShmemWin = MPI.Win.Allocate_shared(
            fcsBytes, mpiDoubleSize, comm=self.physComm
        )

        # Get buffers in the shared memory to fill; convert to Numpy buffers
        engBuf, _ = engShmemWin.Shared_query(0)
        fcsBuf, _ = fcsShmemWin.Shared_query(0)

        engShmemArr = np.ndarray(buffer=engBuf, dtype='f4', shape=engShape)
        fcsShmemArr = np.ndarray(buffer=fcsBuf, dtype='f4', shape=fcsShape)

        # Add natoms to shmem too
        natomsBytes = mpiDoubleSize  # should be enough
        natomsShmemWin = MPI.Win.Allocate_shared(
            natomsBytes, mpiDoubleSize, comm=self.physComm
        )
        natomsBuf, _ = natomsShmemWin.Shared_query(0)
        natomsShmemArr = np.ndarray(buffer=natomsBuf, dtype='f4', shape=(1,))

        # Iteratively build Manager-level database, then insert buffer
        pointer = database
        for i in range(len(path)):
            if path[i] in pointer:
                pointer = pointer[path[i]]
            else:
                pointer[path[i]] = {}
                pointer = pointer[path[i]]

        pointer['energy'] = engShmemArr
        pointer['forces'] = fcsShmemArr
        natoms[path[0]] = natomsShmemArr

        # Finally, scatter data to node heads and have them populate buffers
        if self.isNodeHead:
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

        # Build Manager object
        managerId = self.rank // settings['PROCS_PER_MANAGER']
        localRank = self.rank  % settings['PROCS_PER_MANAGER']

        # It's important to know if we're in charge of a Manager object
        self.isManager = (localRank == 0)

        # Build a communicator between master process and Manager heads
        self.managerComm = self.buildManagerComm()

        managerStructs = np.array_split(
            self.structNames,
            self.comm.Get_size() // settings['PROCS_PER_MANAGER']
        )[managerId]

        # Build a communicator between Manager head and its farm of processors
        farmComm = MPI.Comm.Split(comm, managerId, localRank)

        self.manager = Manager(
            managerId, localRank, farmComm, managerStructs,
            settings['PROCS_PER_PHYS_NODE']
        )


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
            values (dict):
                {structName: {svName: <result array>}}
        """

        if self.isManager:
            population = self.managerComm.bcast(population, root=0)

        managerValues = self.manager.evaluate(population, evalType)

        if self.isManager:
            # Now gather all manager values back to the master process
            allValues = self.managerComm.gather(managerValues, root=0)

        if self.isMaster:
            # allValues = [{structName: list of values for subset of structs}]
            values = {}
            for mgrVal in allValues:
                for structName, vals in mgrVal.items():
                    values[structName] = vals
        else:
            values = None

        return values
        
