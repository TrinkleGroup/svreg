import dask
import numpy as np


class SVEvaluator:

    def __init__(self, client, database, settings):
        self.client     = client
        self.database   = database
        self.settings   = settings

    
    def distributeDatabase(self):
        """Send data to distributed RAM"""
        for evalType in self.database:
            for bondType in self.database[evalType]:
                for elem in self.database[evalType][bondType]:
                    self.client.persist(
                        self.database[evalType][bondType][elem]
                    )


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

        allSVnames  = list(self.database['energy'].keys())
        structNames = list(self.database.attrs['structNames'])
        elements    = list(self.database.attrs['elements'])
        evalTypes   = ['energy', 'forces']

        results = []
        for evalType in evalTypes:
            for svName in allSVnames:
                for elem in elements:
                    sv = self.database[evalType][svName][elem]
                    pop = populationDict[svName][elem]

                    # TODO: can I use JIT somehow? Like make a wrapper to .dot()?
                    results.append(sv.dot(pop.T))

        n = len(allSVnames)
        engResults, fcsResults = results[:n], results[n:]
        
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

        for evalType in evalTypes[::-1]:
            for svName in allSVnames[::-1]:
                for elem in elements[::-1]:
                    res = results.pop()

                    # Parse the per-structure results
                    attrs = self.database.attrs[evalType][elem]
                    start = {el: 0 for el in elements}
                    for i, structName in enumerate(structNames):
                        stop = attrs['natom_splits'][i]

                        if evalType == 'energy':
                            val = res[start[elem]:stop, :].sum(axis=0)
                        elif evalType == 'forces':
                            val = res[start[elem]:stop, :]

                            nhost = val.shape[0]//3//n

                            val = val.T.reshape(res.shape[1], 3, nhost, n)
                            val = val.sum(axis=-1).swapaxes(1, 2)

                        summedResults[evalType][structName][svName][elem] = val

                        start[elem] = stop
               
        # .compute() evaluates everything (blocking) and returns results
        # TODO: look into usage. may need traverse=True
        # TODO: it more be more efficient to use results lists then to put them
        # into a dicitonary later
        # TODO: I could probably push this compute into regressor.py
        dask.compute(summedResults)

        return summedResults