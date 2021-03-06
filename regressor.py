import random
import numpy as np
from copy import deepcopy

from tree import SVTree

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
            svNodePool,
            optimizer=None,
            optimizerArgs=None,
        ):

        # Note: it is assumed that Settings has already performed all validation
        # checks on the provided settings.

        self.settings = settings
        self.svNodePool = svNodePool

        self.optimizer = optimizer

        self.optimizerArgs = optimizerArgs
        if self.optimizerArgs is None:
            self.optimizerArgs = {}

        self.trees = []


    def initializeTrees(self, singleNodeTreeProb=0.1):
        """Populates the GA with randomly-generated equation trees."""

        self.trees = [
            SVTree.random(
                svNodePool=self.svNodePool,
                maxDepth=random.randint(1, self.settings['maxTreeDepth']),
            ) for _ in range(self.settings['numberOfTrees'])
        ]

    def evaluateTrees(self, svEng, svFcs, N):
        """
        Updates the SVNode objects in the trees with the given values, then
        evaluate the trees

        Args:
            svEng, svFcs (dict):
                {structName: {svName: list of values for each tree}}

            N (int):
                The number of parameter sets for each node. Used for splitting
                the population of results.

        Return:
            energies, forces(dict):
                {structName: [tree.eval() for tree in self.trees]}
        """

        energies = {struct:[] for struct in svEng.keys()}
        forces   = {struct:[] for struct in svFcs.keys()}
        for structName in energies:
            for svName in svEng[structName]:
                # The list of stacked values for each tree for a given SV type
                listOfEng = svEng[structName][svName]
                listOfFcs = svFcs[structName][svName]

                # Un-stack any stacked values
                unstackedValues = []
                for val1, val2 in zip(listOfEng, listOfFcs):
                    unstackedValues += list(zip(
                        np.split(val1, val1.shape[0]//N, axis=0),
                        np.split(val2, val2.shape[0]//N, axis=0)
                    ))

                # Loop over the list of values
                # for tree, treeVals in zip(self.trees, listOfValues):
                for tree in self.trees[::-1]:
                    # Each node has the same population size, so just split
                    for svNode in tree.svNodes[::-1]:
                        # Only update the SVNode objects of the current type
                        if svNode.description == svName:
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

    
    def evolvePopulation(self, svNodePool):
        """
        Performs an in-place evolution of self.trees using tournament selection,
        crossover, and mutation. Assumes that self.trees is the current set of
        parent trees.
        """

        newTrees = []
        for _ in range(self.settings['numberOfTrees'] - len(self.trees)):
            # parent = self.tournament()
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
                parent.pointMutate(svNodePool, self.settings['pointMutateProb'])

            parent.updateSVNodes()
            newTrees.append(parent)
        
        return newTrees


    def mate(self):
        raise NotImplementedError

    
    def mutate(self):
        raise NotImplementedError
