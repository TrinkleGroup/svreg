import dask
import dask.array
import numpy as np

from numba import jit

@dask.delayed
@jit(nopython=True)
def jitEval(sv, pop):
    return sv @ pop

def futureEval(sv, pop):
    return sv.dot(pop)

@dask.delayed
def delayedEval(sv, pop):
    return sv.dot(pop)

class SVEvaluator:

    def __init__(self, database, settings):
        self.database   = database
        self.settings   = settings

  
    # @profile
    def evaluate(self, populationDict, evalType, useDask=True):
        """
        Evaluates all of the populations on all of their corresponding structure
        vectors in the database.

        Args:
            populationDict (dict):
                {svName: population}

        Returns:
            svResults (dict):
                svResults[structName][svName][elem]
        """

        structNames = list(self.database.attrs['structNames'])
        allSVnames  = list(self.database.attrs['svNames'])
        elements    = list(self.database.attrs['elements'])

        results = []
        for struct in structNames:
            for svName in allSVnames:
                if svName not in populationDict: continue

                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sv = self.database[struct][svName][elem][evalType]
                    pop = populationDict[svName][elem]

                    if useDask:
                        if evalType == 'energy':
                            results.append(sv.dot(pop))
                        else:
                            results.append(delayedEval(sv, pop))
                    else:
                        results.append(np.array(sv).dot(pop))

        # Now sum by chunks before computing to avoid extra communication
        summedResults = {
            structName: {
                svName: {
                    elem: None for elem in elements
                }
                for svName in allSVnames
            }
            for structName in structNames
        }

        @dask.delayed
        def delayedReshape(val, n):
            nhost = val.shape[0]//3//n
            return val.T.reshape(val.shape[1], 3, nhost, n)
        
        @dask.delayed
        def delayedSum(val):
            return val.sum(axis=-1).swapaxes(1, 2)

        for struct in structNames[::-1]:
            for svName in allSVnames[::-1]:
                if svName not in populationDict: continue

                for elem in elements[::-1]:
                    if elem not in populationDict[svName]: continue

                    res = results.pop()

                    # Parse the per-structure results
                    val = None
                    if evalType == 'energy':
                        # TODO: is this a point of possible SV compression?

                        # val = res.sum(axis=0)
                        val = res

                    elif evalType == 'forces':
                        val = res

                        # n = self.database.attrs['natoms'][struct]

                        # if useDask:
                        #     val = delayedReshape(val, n)
                        #     val = delayedSum(val)
                        # else:
                        #     nhost = val.shape[0]//3//n

                        #     val = val.T.reshape(res.shape[1], 3, nhost, n)
                        #     val = val.sum(axis=-1).swapaxes(1, 2)

                    summedResults[struct][svName][elem] = val
                
        return summedResults
