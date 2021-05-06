import h5py
import pickle
from re import I
import dask
import dask.array as da
from dask.distributed import get_client

import numpy as np


class SVEvaluator:

    def __init__(self, database, settings):
        self.database   = database
        self.settings   = settings

    # def evaluate(self, trees, database, fullPopDict, P, allSums=False, useDask=True, useGPU=False):
    def evaluate(self, trees, fullPopDict, P, database, numTasks, useDask=True, useGPU=False):
        if useGPU:
            import cupy as cp

        def singleStructEval(pickledTrees, P, struct, trueForces):
            with h5py.File(self.settings['databasePath'], 'r') as h5pyFile:

                entry = {}
                for sv in fullPopDict:
                    entry[sv] = {}

                    for elem in fullPopDict[sv]:
                        entry[sv][elem] = {}

                        group = h5pyFile[struct][sv][elem]

                        energyData = np.array(group['energy'][()], dtype=np.float32)
                        forcesData = np.array(group['forces'][()], dtype=np.float32)

                        if self.settings['allSums']:
                            forcesData = forcesData.sum(axis=1)

                        entry[sv][elem]['energy'] = energyData
                        entry[sv][elem]['forces'] = forcesData


            allResults = []

            results = {}

            for svName in fullPopDict:
                results[svName] = {}

                for elem in fullPopDict[svName]:
                    pop = np.array(fullPopDict[svName][elem])
                    sve = entry[svName][elem]['energy']
                    svf = entry[svName][elem]['forces']

                    if useGPU:
                        pop = cp.asarray(pop)
                        sve = cp.asarray(sve)
                        svf = cp.asarray(svf)

                    eng = sve.dot(pop)
                    fcs = svf.dot(pop)

                    if useGPU:
                        eng = cp.asnumpy(eng)
                        fcs = cp.asnumpy(fcs)

                    Ne = eng.shape[0]
                    Nn = eng.shape[1] // P

                    eng = eng.reshape((Ne, Nn, P))
                    eng = np.moveaxis(eng, 1, 0)
                    eng = np.moveaxis(eng, -1, 1)

                    Na = fcs.shape[1]

                    fcs = fcs.reshape((Ne, Na, 3, Nn, P))

                    fcs = np.moveaxis(fcs, -2, 0)
                    fcs = np.moveaxis(fcs, -1, 1)

                    results[svName][elem] = {}

                    results[svName][elem]['energy'] = eng
                    results[svName][elem]['forces'] = fcs

            counters = {
                svName: {
                    el: 0 for el in fullPopDict[svName]
                } for svName in fullPopDict
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

                engResult, fcsResult = tree.eval(useDask=False, allSums=self.settings['allSums'])

                fcsErrors = np.average(
                    abs(sum(fcsResult) - trueForces), axis=(1,2)
                )

                treeResults.append([sum(engResult), fcsErrors])

            allResults.append(treeResults)

            return allResults


        def batchStructsEval(pickledTrees, P):

            from dask.distributed import get_worker
            worker = get_worker()

            names = list(worker._structures.keys())

            allResults = []

            if not self.settings['allSums']:
                # Evaluate structures one at a time

                # TODO: if you can't stack, is there any point of batching? hurts load balancing
                # TODO: let rank 0 on each node use GPU, but have others run on CPU
                pass

                for struct in names:
                    allResults += singleStructEval(
                        pickledTrees, P,
                        worker._structures[struct],
                        worker._true_forces[struct]
                    )
            else:
                # Structures can be stacked and evaluated all at once

                entries = [worker._structures[k] for k in names]

                # Evaluate each SV for all structures simultaneously
                results = {}

                for svName in fullPopDict:
                    results[svName] = {}

                    for elem in fullPopDict[svName]:

                        # pop = cp.asarray(fullPopDict[svName][elem])
                        pop = np.array(fullPopDict[svName][elem])
                        if useGPU:
                            pop = cp.asarray(pop)

                        results[svName][elem] = {}

                        # bigSVE = np.array(database[svName][elem]['energy'])
                        # bigSVF = np.array(database[svName][elem]['forces'])

                        bigSVE = [e[svName][elem]['energy'] for e in entries]
                        bigSVF = [e[svName][elem]['forces'] for e in entries]

                        splits = np.cumsum([sve.shape[0] for sve in bigSVE])
                        splits = np.concatenate([[0], splits])

                        # useDask=True, allSums=True

                        bigSVE = np.concatenate(bigSVE)
                        bigSVF = np.concatenate(bigSVF)

                        if useGPU:
                            bigSVE = cp.asarray(bigSVE)
                            bigSVF = cp.asarray(bigSVF)

                        eng = bigSVE.dot(pop)
                        fcs = bigSVF.dot(pop)

                        if useGPU:
                            eng = cp.asnumpy(eng)
                            fcs = cp.asnumpy(fcs)

                        Ne = eng.shape[0]
                        Nn = eng.shape[1] // P

                        eng = eng.reshape((Ne, Nn, P))
                        eng = np.moveaxis(eng, 1, 0)
                        eng = np.moveaxis(eng, -1, 1)

                        Na = fcs.shape[0]
                        fcs = fcs.reshape((Na, 3, Nn, P))

                        fcs = da.moveaxis(fcs, -2, 0)
                        fcs = da.moveaxis(fcs, -1, 1)

                        perEntryEng = []
                        perEntryFcs = []

                        for idx in range(len(splits)-1):
                            start   = splits[idx]
                            stop    = splits[idx + 1]

                            perEntryEng.append(eng[:, :, start:stop])
                            perEntryFcs.append(fcs[:, :, start:stop])

                        results[svName][elem]['energy'] = perEntryEng
                        results[svName][elem]['forces'] = perEntryFcs

                        del bigSVE
                        del bigSVF
                        if useGPU:
                            cp._default_memory_pool.free_all_blocks()
                            cp._default_pinned_memory_pool.free_all_blocks()

                # Now parse the results
                # for entryNum, struct in enumerate(database.attrs['structNames']):
                for entryNum, struct in enumerate(names):

                    counters = {
                        svName: {
                            el: 0 for el in fullPopDict[svName]
                        } for svName in fullPopDict
                    }

                    treeResults = []
                    for tree in pickledTrees:
                        tree = pickle.loads(tree)

                        for elem in tree.chemistryTrees:
                            for svNode in tree.chemistryTrees[elem].svNodes:
                                svName = svNode.description

                                idx = counters[svName][elem]

                                eng = results[svName][elem]['energy'][entryNum][idx]
                                fcs = results[svName][elem]['forces'][entryNum][idx]

                                svNode.values = (eng, fcs)

                                counters[svName][elem] += 1

                        trueForces = worker._true_forces[struct]

                        engResult, fcsResult = tree.eval(useDask=False, allSums=self.settings['allSums'])

                        fcsErrors = np.average(
                            abs(sum(fcsResult) - trueForces), axis=(1,2)
                        )

                        treeResults.append([sum(engResult), fcsErrors])

                    allResults.append(treeResults)

            allResults = np.array(allResults, dtype=np.float32)

            return allResults, names

        graph   = {}
        keys    = []

        pickledTrees = [pickle.dumps(tree) for tree in trees]

        if self.settings['allSums']:
            for chunkNum in range(numTasks):

                key = 'worker_eval-{}'.format(chunkNum)

                graph[key] = (
                    batchStructsEval,
                    pickledTrees,
                    P,
                )

                keys.append(key)

            return graph, keys
        else:
            for structNum, struct in enumerate(database.attrs['structNames']):
                key = 'struct-eval-{}'.format(structNum)

                graph[key] = (
                    singleStructEval,
                    pickledTrees,
                    P,
                    # database[struct],
                    struct,
                    database.trueValues[struct]['forces']
                )

            def dummy(list_of_results):
                return itertools.chain.from_iterable(list_of_results), database.attrs['structNames']

            import itertools
            graph['gather-evals'] = (
                # itertools.chain.from_iterable,
                dummy,
                ['struct-eval-{}'.format(si) for si in range(len(database.attrs['structNames']))]
            )

            return graph, ['gather-evals']
