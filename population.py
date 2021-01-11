import random
import numpy as np
from copy import deepcopy


class Population(list):
    """
    A class for tracking the current population of trees. Trees in the
    Population have finished optimization and can be used for performing GA
    operations. Note that this is different from the Regressor.trees list, which
    tracks the trees that are currently being optimized.
    """

    def __init__(self, settings, svNodePool):
        list.__init__(self,)
        self.settings = settings
        self.svNodePool = svNodePool

    
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

