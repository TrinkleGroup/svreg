import os
import pickle
import shutil
import random
import numpy as np
from scipy.special import erf

from svreg.tree import SVTree


class Entry:
    """A helper class for the Archive"""

    def __init__(self, tree, savePath):
        self.savePath   = savePath

        self._tree      = str(tree)
        self.tree       = tree

        self.cost       = np.inf
        self.converged  = False
        self.bestParams = None
        self.bestErrors = None


    # Use getters/setters to read/write objects from pickle files
    @property
    def tree(self):
        path = os.path.join(self.savePath, self._tree, 'tree.pkl')
        with open(path, 'rb') as saveFile:
            t = pickle.load(saveFile)
        
        return t

    
    @tree.setter
    def tree(self, tree):
        self._tree = str(tree)

        os.mkdir(os.path.join(self.savePath, self._tree))

        path = os.path.join(self.savePath, self._tree, 'tree.pkl')
        with open(path, 'wb') as saveFile:
            pickle.dump(tree, saveFile)

    
    @property
    def optimizer(self):
        path = os.path.join(self.savePath, self._tree, 'opt.pkl')
        with open(path, 'rb') as saveFile:
            o = pickle.load(saveFile)
        
        return o

    
    @optimizer.setter
    def optimizer(self, opt):
        path = os.path.join(self.savePath, self._tree, 'opt.pkl')
        with open(path, 'wb') as saveFile:
            pickle.dump(opt, saveFile)


    def __getstate__(self):
        self_dict = self.__dict__.copy()

        del self_dict['_tree']

        # TODO: I think this is saving the Optimizers still

        return self_dict

# TODO: this could probably be merged with regressor somehow

class Archive(dict):
    """
    A class for handling the archiving of tree objects and their optimizers to
    help avoid sampling duplicate trees during symbolic regression, and to help
    manage resuming tree optimization from the previous state.
    """

    def __init__(self, savePath):
        dict.__init__(self,)
        self.savePath = savePath

        if os.path.isdir(self.savePath):
            shutil.rmtree(self.savePath)

        os.mkdir(self.savePath)


    def update(self, trees, costs, errors, params, optimizers):
        for tree, c, e, p, o in zip(trees, costs, errors, params, optimizers):
            # Check if tree in archive, otherwise create a new entry for it
            key = str(tree)
            # entry = self.get(key, Entry(tree, self.savePath))
            if key in self:
                entry = self[key]
            else:
                entry = Entry(tree, self.savePath)

            # Update optimizer
            entry.optimizer = o
            entry.converged = o.stop()

            # Update best cost and parameter set of entry
            bestIdx = np.argmin(c)
            if c[bestIdx] < entry.cost:
                entry.cost = c[bestIdx]
                entry.bestParams = p[bestIdx]
                entry.bestErrors = e[bestIdx]

            self[key] = entry


    @property
    def convergences(self):
        return [entry.converged for entry in self]

    
    @property
    def fitnesses(self):
        return  [entry.cost for entry in self]

    
    def sample(self, N):
        """Returns a sample of N unique trees and their optimizers"""

        keys = list(self.keys())
        costs = np.array([self[k].cost for k in keys])
        costs = 1-erf(np.log(costs))
        costs[np.where(np.isnan(costs))] = 0
        costs /= np.sum(costs)

        sampleNames = np.random.choice(
            keys, size=N, replace=False, p=costs
        )

        trees = [self[k].tree for k in sampleNames]
        opts  = [self[k].optimizer for k in sampleNames]

        return trees, opts

    def log(self):
        """
        Saves the current state of the archive. Note that since the trees and
        optimizers have already been saved, this method only logs the errors,
        costs, and optimal parameters.
        """

        with open(os.path.join(self.savePath, 'archive.pkl'), 'wb') as outfile:
            pickle.dump(self, outfile)
