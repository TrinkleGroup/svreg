import dask
import numpy as np


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
                svResults[evalType][structName][svName][elem]
        """

        structNames = list(self.database.attrs['structNames'])
        allSVnames  = list(self.database.attrs['svNames'])
        elements    = list(self.database.attrs['elements'])
        evalTypes   = ['energy', 'forces']

        results = []
        for struct in structNames:
            for svName in allSVnames:
                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    for evalType in evalTypes:

                        sv = self.database[struct][svName][elem][evalType]
                        pop = populationDict[svName][elem]

                        # TODO: can I use JIT somehow? Like make a wrapper to
                        # .dot()?
                        # TODO: is transpose slowing things down? T when generated
                        results.append(sv.dot(pop.T))

        # Now sum by chunks before computing to avoid extra communication
        # summedResults = [[eng[el].sum() for el in elements] for eng in engs]

        # TODO: profile communication costs for this data structure
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
            for evalType in evalTypes
        }

        # TODO: consider dask computing before reshapes (comm while work?)
        for struct in structNames[::-1]:
            for svName in allSVnames[::-1]:
                for elem in elements[::-1]:
                    if elem not in populationDict[svName]: continue

                    for evalType in evalTypes[::-1]:

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

                        summedResults[evalType][struct][svName][elem] = val
                
        summedResults = self.client.gather(self.client.compute(summedResults))
        
        return summedResults
