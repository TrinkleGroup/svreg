import pickle
import dask
import dask.array as da
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

        # @dask.delayed
        # def stack(lst):
        #     return np.concatenate(lst, axis=-1)

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


    def build_dot_graph(self, trees, database, fullPopDict, P, numTasks=None, allSums=False):

        allResults = []

        # Evaluate each SV for all structures simultaneously
        results = {}

        for svName in fullPopDict:
            results[svName] = {}

            for elem in fullPopDict[svName]:

                # pop = cp.asarray(popDict[svName][elem])
                # pop = np.array(fullPopDict[svName][elem])
                pop = fullPopDict[svName][elem]

                results[svName][elem] = {}

                # sve = cp.asarray(entry[svName][elem]['energy'])
                # svf = cp.asarray(entry[svName][elem]['forces'])

                bigSVE = database[svName][elem]['energy']
                bigSVF = database[svName][elem]['forces']

                splits = database.splits[svName][elem]
                
                eng = bigSVE.dot(pop)
                fcs = bigSVF.dot(pop)

                # eng = cp.asnumpy(eng)
                # fcs = cp.asnumpy(fcs)

                Ne = eng.shape[0]
                Nn = eng.shape[1] // P

                eng = eng.reshape((Ne, Nn, P))
                eng = np.moveaxis(eng, 1, 0)
                eng = np.moveaxis(eng, -1, 1)

                if allSums:
                    Na = fcs.shape[0]
                    fcs = fcs.reshape((Na, 3, Nn, P))
                else:
                    Na = fcs.shape[1]
                    fcs = fcs.reshape((Ne, Na, 3, Nn, P))

                fcs = da.moveaxis(fcs, -2, 0)
                fcs = da.moveaxis(fcs, -1, 1)

                perEntryEng = []
                perEntryFcs = []

                for idx in range(len(splits)-1):
                    start   = splits[idx]
                    stop    = splits[idx + 1]

                    perEntryEng.append(eng[:, :, start:stop])
                    perEntryFcs.append(fcs[:, :, start:stop])

                # perEntryEng = np.array_split(eng, splits[:-1], axis=-1)
                # perEntryFcs = np.array_split(fcs, splits[:-1], axis=-2)

                results[svName][elem]['energy'] = perEntryEng
                results[svName][elem]['forces'] = perEntryFcs

                # del bigSVE
                # del bigSVF
                # cp._default_memory_pool.free_all_blocks()
                # cp._default_pinned_memory_pool.free_all_blocks()

            # results[svName][elem]['energy'] = [(Nn, P, Na) for each entry]
            # results[svName][elem]['forces'] = [(Nn, P, Na, 3) for each entry]

        from dask.distributed import get_client
        client = get_client()

        # results = client.gather(client.compute(results))

        # Now parse the results
        for entryNum, struct in enumerate(database.attrs['structNames']):
            counters = {
                svName: {
                    el: 0 for el in fullPopDict[svName]
                } for svName in fullPopDict
            }

            treeResults = []
            for tree in trees:
                # tree = pickle.loads(tree)

                for elem in tree.chemistryTrees:
                    for svNode in tree.chemistryTrees[elem].svNodes:
                        svName = svNode.description

                        idx = counters[svName][elem]

                        eng = results[svName][elem]['energy'][entryNum][idx]
                        fcs = results[svName][elem]['forces'][entryNum][idx]

                        svNode.values = (eng, fcs)

                        counters[svName][elem] += 1

                engResult, fcsResult = tree.eval(useDask=True, allSums=allSums)

                trueForces = database.trueValues[struct]['forces']
                fcsErrors = da.average(
                    abs(sum(fcsResult) - trueForces), axis=(1,2)
                )

                treeResults.append([sum(engResult), fcsErrors])

            allResults.append(treeResults)

        allResults = client.gather(client.compute(allResults))

        # allResults = [[results for each tree] for each entry]
        allResults = np.array(allResults, dtype=np.float32)

        return allResults

        graph   = {}
        keys    = []

        pickledTrees = [pickle.dumps(tree) for tree in trees]

        perTreeResults = []
        for chunkNum in range(numTasks):

            key = 'worker_eval-{}'.format(chunkNum)

            graph[key] = (
              treeStructEval,
              pickledTrees,
              fullPopDict,
              P,
              allSums
            )

            keys.append(key)

        return graph, keys
