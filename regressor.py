import random
import numpy as np
from copy import deepcopy

import cma
from optimizers import GAWrapper, SofomoreWrapper

from nodes import SVNode
from tree import SVTree
from tree import MultiComponentTree as MCTree

import dask.array

# TODO: may be able to merge this into ga.py, it really only initializes
# trees/optimizers and evaluates the trees.

class SVRegressor:
    """
    A class for running a genetic algorithm to perform symbolic
    regression using structure vectors.

    Attributes:

        settings (Settings):
            The settings used during regression. Use
            Settings.printValidSettings() to see valid settings options and
            suggested values.

        svNodePool (list):
            A list of SVNode objects representing the different types of SVs
            to be used during the regression.

        optimizer (object):
            The constructor for an optimizer object that will be used to
            optimize tree parameters. It is assumed that `optimizer` is a
            population-based optimizer and that it implements an ask() and
            tell() interface. An optimizer instance will be generated by calling
            optimizer(tree.populate(N=1)[0], **optimizerArgs) for tree in
            self.trees.

        optimizerArgs (dict):
            A dictionary of named arguments to be passed to the tree optimizers.

        trees (list):
            A list of SVTree objects being optimized.
    """

    def __init__(
            self,
            settings,
            database,
        ):

        # Note: it is assumed that Settings has already performed all validation
        # checks on the provided settings.

        self.settings = settings
        self.svNodePool = buildSVNodePool(database)
        numStructs = len(database.attrs['structNames'])

        if settings['optimizer'] == 'CMA':
            self.optimizer = cma.CMAEvolutionStrategy
            self.optimizerArgs = [
                1.0,  # defaulted sigma0 value
                {'verb_disp': 0, 'popsize':settings['optimizerPopSize']}
            ]
        elif settings['optimizer'] == 'GA':
            self.optimizer = GAWrapper
            self.optimizerArgs = [
                {
                    'verb_disp': 0,
                    'popsize': settings['optimizerPopSize'],
                    'pointMutateProb': settings['pointMutateProb']
                }
            ]
        elif settings['optimizer'] == 'Sofomore':
            self.optimizer = SofomoreWrapper
            self.optimizerArgs = {
                'numStructs': numStructs,
                'paretoDimensionality': 2,
                'CMApopSize': settings['optimizerPopSize'],
                'SofomorePopSize': settings['numberOfTrees'],  # Placeholder
                # 'threads_per_node': settings['PROCS_PER_PHYS_NODE'],
                'threads_per_node': None,
            }
        else:
            raise NotImplementedError(
                'Must be one of `GA`, `CMA`, or `Sofomore`.'
            )

        self.trees = []


    def initializeTrees(self, elements):
        """Populates the GA with randomly-generated equation trees."""

        numElements = len(elements)

        if numElements < 1:
            raise RuntimeError("numElements must be >= 1 in initializeTrees()")

        if numElements == 1:
            self.trees = [
                SVTree.random(
                    svNodePool=self.svNodePool,
                    maxDepth=random.randint(1, self.settings['maxTreeDepth']),
                ) for _ in range(self.settings['numberOfTrees'])
            ]
        else:
            uniqueTreeNames = []
            treesToAdd = []
            while len(uniqueTreeNames) < self.settings['numberOfTrees']:
                randTree = MCTree.random(
                    svNodePool=self.svNodePool,
                    maxDepth=random.randint(0, self.settings['maxTreeDepth']),
                    elements=elements
                )

                if str(randTree) not in uniqueTreeNames:
                    uniqueTreeNames.append(str(randTree))
                    treesToAdd.append(randTree)
            self.trees = treesToAdd


    def evaluateTrees(self, svResults, N):
        """
        Updates the SVNode objects in the trees with the given values, then
        evaluate the trees

        Args:
            svResults (dict):
                svResults[evalType][structName][svName][elem]

            N (int):
                The number of parameter sets for each node. Used for splitting
                the population of results.

        Return:
            energies, forces(dict):
                {structName: [tree.eval() for tree in self.trees]}
        """

        svEng = svResults['energy']
        svFcs = svResults['forces']

        energies = {struct:[] for struct in svEng.keys()}
        forces   = {struct:[] for struct in svFcs.keys()}
        for structName in energies:
            for svName in svEng[structName]:
                for elem in svEng[structName][svName]:
                    # The list of stacked values for each tree for a given SV type
                    stackedEng = svEng[structName][svName][elem]
                    stackedFcs = svFcs[structName][svName][elem]

                    # Possible if svName wasn't use in any trees of elem
                    if stackedEng is None: continue

                    # Un-stack values for each element, and append to list
                    unstackedValues = []

                    numNodes = stackedEng.shape[0]//N

                    nodeEng = stackedEng.reshape((numNodes, N))
                    nodeFcs = stackedFcs.reshape(
                        (numNodes, N, stackedFcs.shape[1], stackedFcs.shape[2])
                    )

                    for val1, val2 in zip(nodeEng, nodeFcs):
                        unstackedValues.append((val1, val2))

                    # Loop backwards over the list of values
                    for tree in self.trees[::-1]:
                        for svNode in tree.chemistryTrees[elem].svNodes:
                            # Only update the SVNode objects of the current type
                            if svNode.description == svName:
                                # TODO: I could put dask.compute() here, but it
                                # would involve a lot more dask calls. The advantage
                                # is that it would be more asynchronous and might be
                                # able to load balance better.
                                svNode.values = unstackedValues.pop()

                    # Error check to see if there are leftovers
                    leftovers = len(unstackedValues)
                    if leftovers > 0:
                        raise RuntimeError('Found leftover results.')

            # If here, all of the nodes have been updated with their values
            for tree in self.trees:
                eng, fcs = tree.eval()
                energies[structName].append(eng)
                forces[structName].append(fcs)

        return energies, forces


    def initializeOptimizers(self):
        self.optimizers = [
            self.optimizer(
                tree.populate(N=1)[0],
                *self.optimizerArgs
                # self.optimizerArgs
            )
            for tree in self.trees
        ]
    

    def tournament(self):
        """
        Finds the lowest cost individual from the current set of trees

        Return:
            A deep copy (to avoid multiple trees pointing to the same nodes) of
            the best individual.
        """

        # contenders = random.sample(range(len(self.trees)), len(self.trees))
        contenders = [random.choice(range(len(self.trees)))]

        costs = [self.trees[idx].cost for idx in contenders]

        return deepcopy(self.trees[np.argmin(costs)])

    
    def evolvePopulation(self):
        """
        Performs an in-place evolution of self.trees using tournament selection,
        crossover, and mutation. Assumes that self.trees is the current set of
        parent trees.
        """

        newTrees = []
        for _ in range(self.settings['numberOfTrees'] - len(self.trees)):
            parent = deepcopy(random.choice(self.trees))

            # For handling only allowing crossover OR point mutation
            pmProb = self.settings['crossoverProb']\
                + self.settings['pointMutateProb']

            rand = random.random()
            if rand < self.settings['crossoverProb']:
                # Randomly perform crossover operation
                donor = self.tournament()
                parent.crossover(donor)
            elif rand < pmProb:
                parent.pointMutate(
                    self.svNodePool, self.settings['pointMutateProb']
                )

            parent.updateSVNodes()
            newTrees.append(parent)
        
        return newTrees


    def mate(self):
        raise NotImplementedError

    
    def mutate(self):
        raise NotImplementedError


    def printTop10Header(self, regStep):
        print(regStep, flush=True)

        for treeNum, t in enumerate(self.trees):
            print(treeNum, t)

        print('\t\t\t', ''.join(['{:<10}'.format(i) for i in range(10)]))


    def generatePopulationDict(self, N):
        """
        Generates N parameter sets for each tree, then groups them by SV name
        for easy batch evaluation.

        Returns:
            populationDict: {svName: {el: stacked population}}
            rawPopulation: the parameter arrays generated by each optimizer
        """

        rawPopulations = [np.array(opt.ask(N)) for opt in self.optimizers]

        # Now parse the populations and group them by SV type
        # populationDict[svName][elem]
        populationDict = {}
        for pop, tree in zip(rawPopulations, self.trees):
            # popDict= {el: {svName: population}}
            popDict = tree.parseArr2Dict(pop)

            for elem in popDict:
                for svName in popDict[elem]:
                    if svName not in populationDict:
                        populationDict[svName] = {}
                    if elem not in populationDict[svName]:
                        populationDict[svName][elem] = []

                    populationDict[svName][elem].append(popDict[elem][svName])

        # Stack each group
        for svName in populationDict:
            for elem, popList in populationDict[svName].items():
                # TODO: convert this to Dask array?
                dat = np.vstack(popList)
                populationDict[svName][elem] = dask.array.from_array(
                    dat,
                    chunks=dat.shape,
                    # chunks=(100, popList[0].shape[1]),
                )

        return populationDict, rawPopulations

    
    def updateOptimizers(self, rawPopulations, costs, penalties):
        for treeIdx in range(len(self.optimizers)):
            fullCost = costs[treeIdx] + penalties[treeIdx]

            opt = self.optimizers[treeIdx]
            opt.tell(rawPopulations[treeIdx], fullCost)


def buildSVNodePool(database):
    """Prepare svNodePool for use in tree construction"""

    svNodePool = []

    # `group` is a pointer to an entry for a structure in the database
    for svName in database['energy']:

        restrictions = None
        if 'restrictions' in database.attrs[svName]:
            restrictions = []
            resList = database.attrs[svName]['restrictions'].tolist()[::-1]
            for num in database.attrs[svName]['numRestrictions']:
                tmp = []
                for _ in range(num):
                    tmp.append(tuple(resList.pop()))
                restrictions.append(tmp)

        bondComps = sorted(set(database.attrs[svName]['components']))
        numParams = []
        restr = []

        cList = database.attrs[svName]['components'].tolist()
        for c in bondComps:
            idx = cList.index(c)
            numParams.append(database.attrs[svName]['numParams'][idx])
            restr.append(restrictions[idx])

        if 'paramRanges' in database.attrs[svName]:
            pRanges = []
            for c in bondComps:
                idx = cList.index(c)
                pRanges.append(database.attrs[svName]['paramRanges'][idx])
        else:
            pRanges = None

        bondComps = [c.decode('utf-8') for c in bondComps]


        svNodePool.append(
            SVNode(
                description=svName,
                components=bondComps,
                constructor=[
                    c.decode('utf-8')
                    for c in database.attrs[svName]['components']
                ],
                numParams=numParams,
                restrictions=restr,
                paramRanges=pRanges,
            )
        )

    return svNodePool

