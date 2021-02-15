import dask
import dask.array
from dask.distributed import get_client

import numpy as np

from numba import jit

@dask.delayed
@jit(nopython=True)
def jitDot(sv, pop):
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
        
        tasks = []
        for struct in structNames:
            for svName in allSVnames:
                if svName not in populationDict: continue

                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sv = self.database[struct][svName][elem][evalType]
                    pop = populationDict[svName][elem]

                    tasks.append((sv, pop))

        def dot(tup):
            return tup[0].dot(tup[1])

        if useDask:
            results = [dask.delayed(dot)(t) for t in tasks]

            client = get_client()
            results = client.compute(results)
        else:
            results = [dot((np.array(t[0]), t[1])) for t in tasks]

        summedResults = {
            structName: {
                svName: {
                    elem: None for elem in elements
                }
                for svName in allSVnames
            }
            for structName in structNames
        }

        for struct in structNames[::-1]:
            for svName in allSVnames[::-1]:
                if svName not in populationDict: continue

                for elem in elements[::-1]:
                    if elem not in populationDict[svName]: continue

                    summedResults[struct][svName][elem] = results.pop()
                
        return summedResults
