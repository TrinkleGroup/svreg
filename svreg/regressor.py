import os
import json
import random
import pickle
import numpy as np
from copy import deepcopy

import cma
from svreg.optimizers import GAWrapper#, SofomoreWrapper
from svreg.nodes import SVNode
from svreg.tree import SVTree
from svreg.tree import MultiComponentTree as MCTree

import dask

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

        self.structNames    = database.attrs['structNames']
        self.svNames        = database.attrs['svNames']
        self.elements       = database.attrs['elements']

        if settings['optimizer'] == 'CMA':
            self.optimizer = cma.CMAEvolutionStrategy
            self.optimizerArgs = [
                1.0,  # defaulted sigma0 value
                {
                    'verb_disp': 0,
                    'popsize': settings['optimizerPopSize'],
                    'maxiter': settings['maxNumOptimizerSteps'],
                    'tolx': 1e-8,  # changes in x-values
                    'tolfunhist': 1e-8,
                    'tolfun': 1e-8,
                    'tolfunrel': 1e-8,
                }
            ]
        elif settings['optimizer'] == 'GA':
            self.optimizer = GAWrapper
            self.optimizerArgs = [
                {
                    'verb_disp': 0,
                    'popsize': settings['optimizerPopSize'],
                    'maxiter': settings['maxNumOptimizerSteps'],
                    'tolx': 1e-8,  # changes in x-values
                    'tolfunhist': 1e-8,
                    'tolfun': 1e-8,
                    'tolfunrel': 1e-8,
                }
            ]
        # elif settings['optimizer'] == 'Sofomore':
        #     self.optimizer = SofomoreWrapper
        #     self.optimizerArgs = {
        #         'numStructs': numStructs,
        #         'paretoDimensionality': 2,
        #         'CMApopSize': settings['optimizerPopSize'],
        #         'SofomorePopSize': settings['numberOfTrees'],  # Placeholder
        #         'threads_per_node': settings['PROCS_PER_PHYS_NODE'],
        #         'threads_per_node': None,
        #     }
        else:
            raise NotImplementedError(
                'Must be one of `GA`, `CMA`, or `Sofomore`.'
            )

        self.trees = []
        self.populationDict = None


    def initializeTrees(self, elements):
        """Populates the GA with randomly-generated equation trees."""

        numElements = len(elements)

        if numElements < 1:
            raise RuntimeError("numElements must be >= 1 in initializeTrees()")

        uniqueTreeNames = []

        for ii, tree in enumerate(self.trees):
            tn = str(tree)
            if tn not in uniqueTreeNames:
                uniqueTreeNames.append(tn)
            else:
                del self.trees[ii]
                print("Removed duplicate tree: {}".format(tn))

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

        self.trees += treesToAdd


    def evaluateTrees(self, svEng, svFcs, P, trueValues, useDask=True):
        """
        Updates the SVNode objects in the trees with the given values, then
        evaluate the trees

        Args:
            svEng (dict):
                svEng[structName][svName][elem] = computed values for given node

            svFcs (dict):
                svFcs[structName][svName][elem] = computed values for given node

            P (int):
                The number of parameter sets for each node. Used for splitting
                the population of results.

        Return:
            energies, forces (dict):
                {structName: [tree.eval() for tree in self.trees]}
        """

        structNames = list(svEng.keys())
        svNames     = list(svEng[structNames[0]].keys())
        
        # NOTE: elements must be sorted here to match assumed ordering of
        # tree.svNodes
        elements = sorted(list(svEng[structNames[0]][svNames[0]].keys()))

        # indexers[sv][el][i] = list of indices for svEng[*][sv][el] for tree i
        indexers = {}

        struct0 = structNames[0]
        # Build dictionary of indexers for each SV
        for svName in svEng[struct0]:
            indexers[svName] = {}
            for elem in elements:
                indexers[svName][elem]  = []

                counter = 0

                for tree in self.trees:
                    treeIndices = []
                    for svNode in tree.chemistryTrees[elem].svNodes:
                        # Only update the SVNode objects of the current type
                        if svNode.description == svName:
                            treeIndices.append(counter)
                            counter += 1

                    # Reverse list here since we'll be popping from it later
                    indexers[svName][elem].append(treeIndices[::-1])

        taskArgs = []
        for struct in structNames:
            indexCopy = deepcopy(indexers)

            for ii, tree in enumerate(self.trees):
                treeArgs = []
                for elem in elements:
                    for svNode in tree.chemistryTrees[elem].svNodes:
                        svName = svNode.description

                        engDot = svEng[struct][svName][elem]
                        fcsDot = svFcs[struct][svName][elem]

                        treeArgs.append(
                            (engDot, fcsDot, indexCopy[svName][elem][ii].pop())
                        )

                taskArgs.append(treeArgs)

        taskArgs = taskArgs[::-1]

        perTreeResults = []
        for structName in structNames:
            for t in self.trees:
                args = taskArgs.pop()

                if useDask:
                    perTreeResults.append(
                        dask.delayed(parseAndEval, pure=True, nout=2)(
                            pickle.dumps(t), args, P,
                            trueValues[structName]['forces'],
                            allSums=self.settings['allSums']
                        )
                    )
                else:
                    perTreeResults.append(
                        parseAndEval(
                            t, args, P,
                            trueValues[structName]['forces'],
                            allSums=self.settings['allSums']
                        )
                    )

        return perTreeResults 


    def initializeOptimizers(self):
        import hashlib

        h5Hash = lambda t: hashlib.md5(str(t).encode()).hexdigest()

        path = os.path.join(self.settings['outputPath'], 'outcmaes', '{}/')

        # self.optimizers = [
        #     self.optimizer(
        #         tree.populate(N=1)[0],
        #         *.update(
        #             self.optimizerArgs[-1]
        #         )
        #     )
        #     for tree in self.trees
        # ]

        argsCopy = deepcopy(self.optimizerArgs)

        self.optimizers = []
        for tree in self.trees:
            d = {'verb_filenameprefix': path.format(h5Hash(tree))}
            d.update(self.optimizerArgs[-1])

            argsCopy[-1] = d

            self.optimizers.append(
                self.optimizer(
                    tree.populate(N=1)[0],
                    *argsCopy
                )
            )


    def tournament(self, topN):
        """
        Randomly return a random individual from the topN individuals in the
        population.

        Return:
            A deep copy (to avoid multiple trees pointing to the same nodes) of
            the best individual.
        """

        indices = np.arange(len(self.trees))
        costs = [t.cost for t in self.trees]

        argsort = np.argsort(costs)
        costs = costs[argsort]
        indices = indices[argsort]

        return deepcopy(self.trees[random.choice(indices[:topN])])


    def newIndividual(self):
        """
        Generates a new individual using the current population. Allow for
        random point mutation.
        """

        newTree = self.tournament(self.settings['tournamentSize'])
        donor   = self.tournament(self.settings['tournamentSize'])

        newTree.crossover(donor)

        if random.random() < self.settings['pointMutateProb']:
            newTree.pointMutate(
                self.svNodePool, self.settings['pointMutateProb']
            )

        newTree.updateSVNodes()

        return newTree


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

        # Used for parsing later
        self.numNodes = {}

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


        # Count the number of each node to help with parsing results later
        for tree in self.trees:
            for elem in tree.chemistryTrees:
                for svNode in tree.chemistryTrees[elem].svNodes:
                    svName = svNode.description

                    if svName not in self.numNodes:
                        self.numNodes[svName] = {}

                    if elem not in self.numNodes[svName]:
                        self.numNodes[svName][elem] = 0

                    self.numNodes[svName][elem] += 1

        self.chunks = {}
        # Stack each group
        for svName in populationDict:
            if svName not in self.chunks:
                self.chunks[svName] = {}

            for elem, popList in populationDict[svName].items():
                dat = np.concatenate(popList, axis=0).T
                dat = dat.astype('float32')

                populationDict[svName][elem] = dat.astype('float32')

        return populationDict, rawPopulations

    
    def updateOptimizers(self, rawPopulations, costs, penalties):
        for treeIdx in range(len(self.optimizers)):
            fullCost = costs[treeIdx] + penalties[treeIdx]

            opt = self.optimizers[treeIdx]
            opt.tell(rawPopulations[treeIdx], fullCost)
            opt.logger.add()


    def checkStale(self):
        """
        Returns a list of the indices of any trees that have finished
        optimizing.
        """

        stale = []
        messages = []

        for i, opt in enumerate(self.optimizers):
            if opt.stop():
                stale.append(i)
                messages.append(opt.stop())

        return stale, messages


def buildSVNodePool(database):
    """Prepare svNodePool for use in tree construction"""

    svNodePool = []

    for svName in database[database.attrs['structNames'][0]]:

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
                inputTypes=json.loads(
                    database.attrs[svName]['inputTypes'].decode('utf-8').replace("'", '"')
                )
            )
        )

    return svNodePool

    
def parseAndEval(
    *args,
    **kwargs,
    ):

    tree            = kwargs['tree']
    listOfIndices   = kwargs['listOfIndices']
    P               = kwargs['P']
    tvF             = kwargs['tvF']
    numChunks       = kwargs['numChunks']
    allSums         = kwargs['allSums']

    import pickle
    tree = pickle.loads(tree)

    nodeCounter     = 0
    chunkCounter    = 0
    for elem in tree.elements:
        for svNode in tree.chemistryTrees[elem].svNodes:
            svName = svNode.description

            chunkTup = []
            for _ in range(numChunks[svName][elem]):
                chunkTup.append(args[chunkCounter])
                chunkCounter += 1

            idx = listOfIndices[nodeCounter]
            nodeCounter += 1

            eng = np.concatenate([c[0] for c in chunkTup], axis=-1)
            fcs = np.concatenate([c[1] for c in chunkTup], axis=-1)

            Ne = eng.shape[0]
            Nn = eng.shape[1] // P

            eng = eng.reshape((Ne, Nn, P))
            eng = np.moveaxis(eng, 1, 0)
            eng = np.moveaxis(eng, -1, 1)

            # fcs shape: (Ne, Na, 3, P*Nn)

            if allSums:
                Na = fcs.shape[0]
                fcs = fcs.reshape(Na, 3, Nn, P)
            else:
                Na = fcs.shape[1]
                fcs = fcs.reshape(Ne, Na, 3, Nn, P)

            fcs = np.moveaxis(fcs, -2, 0)
            fcs = np.moveaxis(fcs, -1, 1)

            # fcs shape: (Nn, P, Ne, Na, 3)

            svNode.values = (eng[idx], fcs[idx])

    engResult, fcsResult = tree.eval(useDask=False, allSums=allSums)

    fcsErrors = np.average(abs(sum(fcsResult) - tvF['forces']), axis=(1,2))

    return sum(engResult), fcsErrors
