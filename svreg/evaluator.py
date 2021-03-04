import dask
import dask.array
from dask.distributed import get_client

import numpy as np


class SVEvaluator:

    def __init__(self, database, settings):
        self.database   = database
        self.settings   = settings

  
    def evaluate(self, populationDict, evalType, numChunks, useDask=True):
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

        ffgTasks = []
        rhoTasks = []
        for struct in structNames:
            for svName in allSVnames:
                if svName not in populationDict: continue

                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sv = self.database[struct][svName][elem][evalType]
                    pop = populationDict[svName][elem]

                    if useDask:
                        if 'ffg' in svName:
                            for chunk in pop:
                                ffgTasks.append((sv, chunk))
                        else:
                            rhoTasks.append((sv, pop))
                    else:
                        tasks.append((sv, np.concatenate(pop, axis=-1)))

        def dot(tup):
            return tup[0].dot(tup[1])

        import time

        @dask.delayed
        def ffgDot(tup):
            start = time.time()
            res = tup[0].dot(tup[1])
            # print("ffgDot {}: {} (s)".format(tup[1].shape, time.time() - start), flush=True)
            return res

        @dask.delayed
        def rhoDot(tup):
            pop = np.concatenate(tup[1], axis=-1)
            return tup[0].dot(pop)

        if useDask:
            ffgResults = [ffgDot(t) for t in ffgTasks]
            rhoResults = [rhoDot(t) for t in rhoTasks]

            client = get_client()

            ffgResults = client.compute(ffgResults, priority=100)

            results = []
            for struct in structNames[::-1]:
                for svName in allSVnames[::-1]:
                    if svName not in populationDict: continue

                    for elem in elements[::-1]:
                        if elem not in populationDict[svName]: continue

                        if 'ffg' in svName:
                            tmp = []
                            for _ in range(numChunks[svName][elem]):
                                tmp.append(ffgResults.pop())
                            # results.append(stack(tmp[::-1]))
                            results.append(tmp[::-1])
                        else:
                            results.append(rhoResults.pop())

            results = results[::-1]
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
