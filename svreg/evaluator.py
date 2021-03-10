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


    def build_dot_graph(self, trees, fullPopDict, trueValues, P):

        structNames = list(self.database.attrs['structNames'])
        allSVnames  = list(self.database.attrs['svNames'])
        elements    = list(self.database.attrs['elements'])

        import pickle
        import cupy as cp

        def treeStructEval(entry, pickledTrees, popDict, tvF, P):
            treeResults = []

            # Evaluate all SVs for the structure
            results = {}


            # # Pack SVs and population for each SV type
            # els = list(popDict[list(popDict.keys())[0]])

            # packedSV    = {
            #     'ffg': {el: {'energy': [], 'forces': []} for el in els},
            #     'rho': {el: {'energy': [], 'forces': []} for el in els},
            # }

            # packedPop   = {
            #     'ffg': {el: [] for el in els},
            #     'rho': {el: [] for el in els},
            # }

            # for svName in popDict:
            #     for elem in popDict[svName]:
            #         svType  = 'ffg' if 'ffg' in svName else 'rho'

            #         packedSV[svType][elem]['energy'].append(entry[svName][elem]['energy'])
            #         packedSV[svType][elem]['forces'].append(entry[svName][elem]['forces'])

            #         packedPop[svType][elem].append(popDict[svName][elem])

            # # Now evaluate in batches
            # resuls = {}
            # for svType in packedSV:
            #     for elem in packedSV[svType]:

            #         pop = packedPop[svType][elem]
            #         print('pop:', [el.shape for el in pop], flush=True)
            #         pop = np.stack(pop)

            #         pop = cp.asarray(pop)

            #         for evalType in packedSV[svType][elem]:
            #             sv = packedSV[svType][elem][evalType]
            #             sv = np.stack(sv)
            #             sv = cp.asarray(sv)

            #             res = cp.tensordot(sv, pop, axes=-1)
            #             print('res:', sv.shape, pop.shape, res.shape, flush=True)

            # stream = cp.cuda.stream.Stream(non_blocking=True)
            
            numStreams = 0
            for svName in popDict:
                for elem in popDict[svName]:
                    numStreams += 0

            for svName in popDict:
                results[svName] = {}

                for elem in popDict[svName]:

                    pop = cp.asarray(popDict[svName][elem])

                    results[svName][elem] = {}

                    sve = cp.asarray(entry[svName][elem]['energy'])
                    svf = cp.asarray(entry[svName][elem]['forces'])

                    eng = sve.dot(pop)
                    fcs = svf.dot(pop)

                    eng = cp.asnumpy(eng)
                    fcs = cp.asnumpy(fcs)

                    Ne = eng.shape[0]
                    Nn = eng.shape[1] // P
                    Na = fcs.shape[1]

                    eng = eng.reshape((Ne, Nn, P))
                    eng = np.moveaxis(eng, 1, 0)
                    eng = np.moveaxis(eng, -1, 1)

                    fcs = fcs.reshape((Ne, Na, 3, Nn, P))
                    fcs = np.moveaxis(fcs, -2, 0)
                    fcs = np.moveaxis(fcs, -1, 1)

                    results[svName][elem]['energy'] = eng
                    results[svName][elem]['forces'] = fcs

                    del sve
                    del svf
                    cp._default_memory_pool.free_all_blocks()
                    cp._default_pinned_memory_pool.free_all_blocks()

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

                engResult, fcsResult = tree.eval(useDask=False)

                fcsErrors = np.average(abs(sum(fcsResult) - tvF), axis=(1,2))

                treeResults.append((sum(engResult), fcsErrors))

            return treeResults

        graph   = {}
        keys    = []

        pickledTrees = [pickle.dumps(tree) for tree in trees]
        for structNum, struct in enumerate(structNames):
            key = 'treeStructEval-struct_{}'.format(
                structNum
            )

            graph[key] = (
                treeStructEval,
                self.database[struct], pickledTrees, fullPopDict,
                trueValues[struct]['forces'],
                P
            )

            keys.append(key)

        return graph, keys

        # @dask.delayed
        def chunkDot(sve, svf, chunk):
            return sve.dot(chunk), svf.dot(chunk)

        graph = {}
        ordered_keys = []

        for structNum, struct in enumerate(structNames):
            for svName in allSVnames:
                if svName not in populationDict: continue

                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sve     = self.database[struct][svName][elem]['energy']
                    svf     = self.database[struct][svName][elem]['forces']
                    chunks  = populationDict[svName][elem]

                    for ci, chunk in enumerate(chunks):

                        key = 'chunkDot-struct_{}-{}-{}-{}'.format(
                            structNum, svName, elem, ci
                        )
                        ordered_keys.append(key)
                        
                        graph[key] = (chunkDot, sve, svf, chunk)


        client = get_client()
        results = client.get(graph, ordered_keys, sync=False)

        counter = 0
        for structNum, struct in enumerate(structNames):
            for svName in allSVnames:
                if svName not in populationDict: continue

                for elem in elements:
                    if elem not in populationDict[svName]: continue

                    sve     = self.database[struct][svName][elem]['energy']
                    svf     = self.database[struct][svName][elem]['forces']
                    chunks  = populationDict[svName][elem]

                    for ci, chunk in enumerate(chunks):

                        key = 'chunkDot-struct_{}-{}-{}-{}'.format(
                            structNum, svName, elem, ci
                        )
                        
                        graph[key] = results[counter]
                        counter += 1


        # graph[key] = task to eval eng/fcs for 1 struct for 1 sv
        return graph
