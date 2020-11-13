import dask
import numpy as np


@dask.delayed
def delayedEval(sv, pop):
    return sv.dot(pop)

class SVEvaluator:

    def __init__(self, client, database, settings):
        self.client     = client
        self.database   = database
        self.settings   = settings

  
    # @profile
    def evaluate(self, populationDict, evalType):
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
                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sv = self.database[struct][svName][elem][evalType]
                    pop = populationDict[svName][elem]

                    # TODO: can I use JIT somehow? Like make a wrapper to
                    # .dot()?
                    if evalType == 'energy':
                        results.append(sv.dot(pop))
                    else:
                        results.append(delayedEval(sv, pop))

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

        # TODO: consider dask computing before reshapes (comm while work?)
        for struct in structNames[::-1]:
            for svName in allSVnames[::-1]:
                for elem in elements[::-1]:
                    if elem not in populationDict[svName]: continue

                    res = results.pop()

                    # Parse the per-structure results
                    val = None
                    if evalType == 'energy':
                        val = res.sum(axis=0)
                    elif evalType == 'forces':
                        val = res

                        n = self.database.attrs['natoms'][struct]
                        nhost = val.shape[0]//3//n

                        val = val.T.reshape(res.shape[1], 3, nhost, n)
                        val = val.sum(axis=-1).swapaxes(1, 2)

                    summedResults[struct][svName][elem] = val
                
        return summedResults
