import random
import numpy as np
from copy import deepcopy

from svreg.tree import MultiComponentTree as MCTree

class Population(list):
    """
    A class for tracking the current population of trees. Trees in the
    Population have finished optimization and can be used for performing GA
    operations. Note that this is different from the Regressor.trees list, which
    tracks the trees that are currently being optimized.
    """

    def __init__(self, settings, svNodePool, elements):
        list.__init__(self,)
        self.settings = settings
        self.svNodePool = svNodePool
        self.elements = elements

    
    def attemptInsert(self, newTree):
        """
        Compares newTree to a random candidate in the current population, and
        replaces the candidate if newTree has a better fitness.

        Returns:
            True if newTree was added, else False
        """

        if len(self) < self.settings['numberOfTrees']:
            self.append(newTree)
            return True
        else:
            indices = np.arange(len(self))
            idx = random.choice(indices)
            candidateRemoval = self[idx]

            if newTree.cost < candidateRemoval.cost:
                self[idx] = newTree

                return True
        
        return False


    def tournament(self, topN):
        """
        Randomly return a random individual from the topN individuals in the
        population.

        Return:
            A deep copy (to avoid multiple trees pointing to the same nodes) of
            the best individual.
        """

        indices = np.arange(len(self))
        costs = np.array([t.cost for t in self])

        argsort = np.argsort(costs)
        costs = costs[argsort]
        indices = indices[argsort]

        return deepcopy(self[random.choice(indices[:topN])])


    def newIndividual(self):
        """
        Generates a new individual using the current population. Allow for
        random point mutation. Returns a new random tree if population is not
        full-sized yet.
        """

        if len(self) < self.settings['numberOfTrees']:
            return MCTree.random(
                svNodePool=self.svNodePool,
                maxDepth=random.randint(0, self.settings['maxTreeDepth']),
                elements=self.elements,
                allSums=self.settings['allSums']
            ), None, None

        newTree = self.tournament(self.settings['tournamentSize'])
        parentCopy = deepcopy(newTree)
        donor   = self.tournament(self.settings['tournamentSize'])

        newTree.crossover(donor)

        if random.random() < self.settings['pointMutateProb']:
            newTree.pointMutate(
                self.svNodePool, self.settings['pointMutateProb'],
                allSums=self.settings['allSums']
            )

        newTree.updateSVNodes()

        return newTree, parentCopy, donor

