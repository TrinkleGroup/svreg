import dask
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

    def __init__(self, client, database, settings):
        self.client     = client
        self.database   = database
        self.settings   = settings

  
    # @profile
    def evaluate(self, populationDict):
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

        engResults = []
        fcsResults = []
        results = []
        for struct in structNames:
            for svName in allSVnames:
                for elem in elements:
                    for evalType in ['energy', 'forces']:
                        if elem not in populationDict[svName]: continue

                        sv = self.database[struct][svName][elem][evalType]
                        pop = populationDict[svName][elem]

                        # TODO: can I use JIT somehow? Like make a wrapper to .dot()?
                        # if evalType == 'energy':
                        #     # results.append(sv.dot(pop))
                        #     engResults.append(delayedEval(sv, pop))
                        # else:
                        #     # results.append(delayedEval(sv, pop))
                        #     fcsResults.append(delayedEval(sv, pop))

                        results.append(delayedEval(sv, pop))

                    # results.append(delayedEval(sv, pop))
                    # results.append(self.client.submit(futureEval, sv, pop))

        # results = self.client.compute(results)
        # engResults = self.client.compute(engResults)
        # fcsResults = self.client.compute(fcsResults)

        # if evalType == 'forces':
        #     from dask.distributed import wait
        #     wait(results)

        # results = self.client.gather(results)

        # Now sum by chunks before computing to avoid extra communication
        summedResults = {
            evalType: {
                structName: {
                    svName: {
                        elem: None for elem in elements
                    }
                    for svName in allSVnames
                }
                for structName in structNames
            }
            for evalType in ['energy', 'forces']
        }

        @dask.delayed
        def delayedReshape(val, n):
            nhost = val.shape[0]//3//n
            return val.T.reshape(val.shape[1], 3, nhost, n)
        
        @dask.delayed
        def delayedSum(val):
            return val.sum(axis=-1).swapaxes(1, 2)

        @dask.delayed
        def delayedEngSum(val):
            return val.sum(axis=0)

        futures = []
        # TODO: consider dask computing before reshapes (comm while work?)
        for struct in structNames[::-1]:
            for svName in allSVnames[::-1]:
                for elem in elements[::-1]:
                    for evalType in ['forces', 'energy']:
                        if elem not in populationDict[svName]: continue

                        res = results.pop()

                        # Parse the per-structure results
                        val = None
                        if evalType == 'energy':
                            # val = res.sum(axis=0)
                            val = delayedEngSum(res)
                        elif evalType == 'forces':
                            val = res#.result()

                            n = self.database.attrs['natoms'][struct]
                            # nhost = val.shape[0]//3//n

                            # val = val.T.reshape(val.shape[1], 3, nhost, n)
                            # val = val.sum(axis=-1).swapaxes(1, 2)
                            val = delayedReshape(val, n)
                            val = delayedSum(val)

                        summedResults[evalType][struct][svName][elem] = val
                        futures.append(val)
                    
        self.client.compute(futures)
        # summedResults = self.client.gather(summedResults)
        # summedResults = self.client.gather(self.client.compute(summedResults))
        # summedResults = self.client.compute(summedResults)
        return summedResults
