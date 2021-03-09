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


    def build_dot_graph(self, trees, rawPopulations, trueValues, P):

        structNames = list(self.database.attrs['structNames'])
        allSVnames  = list(self.database.attrs['svNames'])
        elements    = list(self.database.attrs['elements'])

        """
        TODO: make each task compute the energies/forces for a single tree.

        Notes:
            - This doesn't use batching logic; should just scatter raw
            populations batched populations.

            - You might be able to hold all intermediate results in GPU at the
            same time (leave everything as cupy array), that way tree evals are
            GPU-accelerated too

            - This might enable larger CMA population sizes since the GPU would
            be able to evaluate them trivially quickly.

            - Is GPU communication going to make each of these tasks still take
            awhile?


        for struct in structNames:
            for tree in trees:
                - pickle tree and send to worker (or maybe scatter before?)
                - send self.database[struct]
                - parse raw population into SV dictionary (or just send as
                dictionary).
                - evaluate each of the SVs of the tree
        """

        import pickle

        def treeStructEval(entry, tree, rawPop, tvF, P):
            tree = pickle.loads(tree)
            
            # popDict= {el: {svName: population}}
            popDict = tree.parseArr2Dict(rawPop)

            results = {}
            for elem in popDict:

                results[elem] = {}

                for svName in popDict[elem]:
                    results[elem][svName] = {}

                    sve = entry[svName][elem]['energy']
                    svf = entry[svName][elem]['forces']

                    eng = sve.dot(popDict[elem][svName].T)
                    fcs = svf.dot(popDict[elem][svName].T)

                    Ne = eng.shape[0]
                    Nn = eng.shape[1] // P
                    Na = fcs.shape[1]

                    eng = eng.reshape((Ne, Nn, P))
                    eng = np.moveaxis(eng, 1, 0)
                    eng = np.moveaxis(eng, -1, 1)

                    fcs = fcs.reshape((Ne, Na, 3, Nn, P))
                    fcs = np.moveaxis(fcs, -2, 0)
                    fcs = np.moveaxis(fcs, -1, 1)

                    results[elem][svName]['energy'] = eng
                    results[elem][svName]['forces'] = fcs

            counters = {
                el: {
                    svName: 0 for svName in popDict[el]
                } for el in popDict
            }

            for elem in tree.chemistryTrees:
                for svNode in tree.chemistryTrees[elem].svNodes:
                    svName = svNode.description

                    idx = counters[elem][svName]

                    eng = results[elem][svName]['energy'][idx]
                    fcs = results[elem][svName]['forces'][idx]

                    svNode.values = (eng, fcs)

                    counters[elem][svName] += 1

            engResult, fcsResult = tree.eval(useDask=False)

            fcsErrors = np.average(abs(sum(fcsResult) - tvF), axis=(1,2))

            return sum(engResult), fcsErrors

        graph   = {}
        keys    = []
        for structNum, struct in enumerate(structNames):
            for treeNum, (tree, rawPop) in enumerate(zip(trees, rawPopulations)):
                key = 'treeStructEval-struct_{}-tree_{}'.format(
                    structNum, treeNum
                )

                keys.append(key)

                graph[key] = (
                    treeStructEval,
                    self.database[struct], pickle.dumps(tree), rawPop,
                    trueValues[struct]['forces'],
                    P
                )

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