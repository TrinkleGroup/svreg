import dask
import numpy as np


class SVEvaluator:

    def __init__(self, client, database, settings):
        self.client     = client
        self.database   = database
        self.settings   = settings

        # self.distributeDatabase()

    
    def distributeDatabase(self):
        """Send data to distributed RAM"""
        for evalType in self.database:
            for bondType in self.database[evalType]:
                for elem in self.database[evalType][bondType]:
                    if evalType == 'forces':
                        self.client.persist(
                            dask.array.from_array(self.database[evalType][bondType][elem])
                        )


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

        allSVnames  = list(self.database['energy'].keys())
        structNames = list(self.database.attrs['structNames'])
        elements    = list(self.database.attrs['elements'])
        evalTypes   = ['energy', 'forces']

        # results_eng = []
        # results_fcs = []
        results = []
        for evalType in evalTypes:
            for svName in allSVnames:
                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sv = self.database[evalType][svName][elem]
                    pop = populationDict[svName][elem]

                    # TODO: can I use JIT somehow? Like make a wrapper to
                    # .dot()?
                    # TODO: is transpose slowing things down? T when generated
                    results.append(sv.dot(pop.T))
                    # res = sv.dot(pop.T)
                    # if evalType == 'energy':
                    #     results_eng.append(res)
                    # elif evalType == 'energy':
                    #     results_fcs.append(res)
                    # else:
                    #     raise RuntimeWarning(
                    #         'Invalid evalType: {}'.format(evalType)
                    #     )

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
        for evalType in evalTypes[::-1]:
            for svName in allSVnames[::-1]:
                for elem in elements[::-1]:
                    if elem not in populationDict[svName]: continue

                    res = results.pop()

                    # Parse the per-structure results
                    splits = self.database.attrs[evalType][elem]['natom_splits']
                    start = {el: 0 for el in elements}
                    for i, structName in enumerate(structNames):
                        stop = splits[i]

                        val = None
                        if evalType == 'energy':
                            val = res[start[elem]:stop, :].sum(axis=0)
                        elif evalType == 'forces':
                            val = res[start[elem]:stop, :]

                            n = self.database.attrs['natoms'][i]
                            nhost = val.shape[0]//3//n

                            val = val.T.reshape(res.shape[1], 3, nhost, n)
                            val = val.sum(axis=-1).swapaxes(1, 2)

                        summedResults[evalType][structName][svName][elem] = val

                        start[elem] = stop
               
        summedResults = self.client.gather(self.client.compute(summedResults))
        
        return summedResults
