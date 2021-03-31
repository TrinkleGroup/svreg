import pickle
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


    def build_dot_graph(self, trees, fullPopDict, trueValues, P, numTasks=None, allSums=False):

        structNames = list(self.database.attrs['structNames'])


        def treeStructEval(pickledTrees, popDict, P, allSums):

            from dask.distributed import get_worker
            worker = get_worker()

            names = list(worker._structures.keys())
            entries = [worker._structures[k] for k in names]
            tvFs = worker._true_forces.values()

            allResults = []
            for entry, tvF in zip(entries, tvFs):

                treeResults = []

                # Evaluate all SVs for the structure
                results = {}

                numStreams = 0
                for svName in popDict:
                    for elem in popDict[svName]:
                        numStreams += 0

                for svName in popDict:
                    results[svName] = {}

                    for elem in popDict[svName]:

                        # pop = cp.asarray(popDict[svName][elem])
                        pop = np.array(popDict[svName][elem])

                        results[svName][elem] = {}

                        # sve = cp.asarray(entry[svName][elem]['energy'])
                        # svf = cp.asarray(entry[svName][elem]['forces'])
                        sve = np.array(entry[svName][elem]['energy'])
                        svf = np.array(entry[svName][elem]['forces'])

                        eng = sve.dot(pop)
                        fcs = svf.dot(pop)

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

                        fcs = np.moveaxis(fcs, -2, 0)
                        fcs = np.moveaxis(fcs, -1, 1)

                        results[svName][elem]['energy'] = eng
                        results[svName][elem]['forces'] = fcs

                        del sve
                        del svf
                        # cp._default_memory_pool.free_all_blocks()
                        # cp._default_pinned_memory_pool.free_all_blocks()

                # Now parse the results
                counters = {
                    svName: {
                        el: 0 for el in popDict[svName]
                    } for svName in popDict
                }

                treeResults = []
                for tree in pickledTrees:
                    tree = pickle.loads(tree)

                    for elem in tree.chemistryTrees:
                        for svNode in tree.chemistryTrees[elem].svNodes:
                            svName = svNode.description

                            idx = counters[svName][elem]

                            eng = results[svName][elem]['energy'][idx]
                            fcs = results[svName][elem]['forces'][idx]

                            svNode.values = (eng, fcs)

                            counters[svName][elem] += 1

                    engResult, fcsResult = tree.eval(useDask=False, allSums=allSums)

                    fcsErrors = np.average(abs(sum(fcsResult) - tvF), axis=(1,2))

                    treeResults.append([sum(engResult), fcsErrors])

                allResults.append(treeResults)

            allResults = np.array(allResults, dtype=np.float32)

            return allResults, names

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
