import os
import pickle
import shutil
import hashlib
import numpy as np
from scipy.special import erf

def md5Hash(tree):
    return hashlib.md5(str(tree).encode()).hexdigest()


class Entry:
    """A helper class for the Archive"""

    def __init__(self, tree, savePath):
        self.savePath   = savePath

        self._tree      = md5Hash(tree)
        self.tree       = tree

        self.cost       = np.inf
        # self.converged  = False
        self.bestParams = None
        # self.bestErrors = None


    # Use getters/setters to read/write objects from pickle files
    @property
    def tree(self):
        path = os.path.join(self.savePath, self._tree, 'tree.pkl')
        with open(path, 'rb') as saveFile:
            t = pickle.load(saveFile)
        
        return t

    
    @tree.setter
    def tree(self, tree):
        self._tree = md5Hash(tree)

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


class Archive(dict):
    """
    A class for handling the archiving of tree objects and their optimizers to
    log results and help avoid sampling duplicate trees during symbolic
    regression.
    """

    def __init__(self, savePath):
        dict.__init__(self,)
        self.savePath = savePath

        if os.path.isdir(self.savePath):
            shutil.rmtree(self.savePath)

        os.mkdir(self.savePath)

        self.hashLog = {}


    def update(self, tree, cost, errors, params, optimizer):
    # def update(self, tree, cost, params, optimizer):
        # Check if tree in archive, otherwise create a new entry for it
        key = md5Hash(tree)
        # entry = self.get(key, Entry(tree, self.savePath))
        if key in self:
            raise RuntimeError("How did you re-run an existing tree? {}".format(key))
            entry = self[key]
        else:
            entry = Entry(tree, self.savePath)

        # Update optimizer
        entry.optimizer = optimizer
        # entry.converged = optimizer.stop()

        entry.cost = cost
        entry.bestParams = params
        entry.bestErrors = errors

        # # Update best cost and parameter set of entry
        # bestIdx = np.argmin(cost)
        # if cost[bestIdx] < entry.cost:
        #     entry.cost = cost[bestIdx]
        #     entry.bestParams = params[bestIdx]
            # entry.bestErrors = errors[bestIdx]

        self[key] = entry

        self.hashLog[key] = str(tree)

    # @property
    # def convergences(self):
    #     return [entry.converged for entry in self]

    
    @property
    def fitnesses(self):
        return  [entry.cost for entry in self]

    
    def sample(self, N):
        """Returns a sample of N unique trees and their optimizers"""

        keys = list(self.keys())
        costs = np.array([self[k].cost for k in keys])
        costs = 1-erf(np.log(costs))
        costs[np.where(np.isnan(costs))] = 0

        # Handle the case where all costs are really high
        if sum(costs) == 0:
            costs = np.random.random(size=costs.shape)

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


    def printStatus(self, regressorNames):
        print()
        printNames = list(self.keys())
        printCosts = [self[n].cost for n in printNames]

        for idx in np.argsort(printCosts):
            indicator = ''
            if printNames[idx] in regressorNames:
                indicator = '->'

            print(
                '\t{}{:.2f}'.format(indicator, printCosts[idx]),
                printNames[idx],
            )
        print(flush=True)


    def pruneAndLoad(self, sampledTrees, newTrees, opts, regressor):
        currentTreeNames = [md5Hash(t) for t in sampledTrees]

        uniqueTrees = sampledTrees
        uniqueOptimizers = opts
        for tree in newTrees:
            treeName = md5Hash(tree)
            if treeName not in currentTreeNames:
                # Not a duplicate
                uniqueTrees.append(tree)

                if treeName in self:
                    # Load archived optimizer
                    uniqueOptimizers.append(self[treeName].optimizer)
                else:
                    # Create new optimizer
                    uniqueOptimizers.append(
                        regressor.optimizer(
                            tree.populate(N=1)[0],
                            *regressor.optimizerArgs
                        )
                    )

        return uniqueTrees, uniqueOptimizers