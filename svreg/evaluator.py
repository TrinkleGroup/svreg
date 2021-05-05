import pickle
import dask
import dask.array as da
from dask.distributed import get_client

import numpy as np
# import cupy as cp


class SVEvaluator:

    def __init__(self, database, settings):
        self.database   = database
        self.settings   = settings

    def evaluate(self, trees, database, fullPopDict, P, numTasks=None, allSums=False, useDask=True):

        allResults = []

        # Evaluate each SV for all structures simultaneously
        results = {}

        for svName in fullPopDict:
            results[svName] = {}

            for elem in fullPopDict[svName]:

                # pop = cp.asarray(fullPopDict[svName][elem])
                pop = np.array(fullPopDict[svName][elem])

                results[svName][elem] = {}

                # bigSVE = cp.asarray(database[svName][elem]['energy'])
                # bigSVF = cp.asarray(database[svName][elem]['forces'])
                bigSVE = np.array(database[svName][elem]['energy'])
                bigSVF = np.array(database[svName][elem]['forces'])

                if useDask:

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
                else:
                    perEntryEng = []
                    perEntryFcs = []

                    for ii, (sve, svf) in enumerate(zip(bigSVE, bigSVF)):
                        eng = np.dot(sve, pop)
                        fcs = np.dot(svf, pop)

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

                        perEntryEng.append(eng)
                        perEntryFcs.append(fcs)

                results[svName][elem]['energy'] = perEntryEng
                results[svName][elem]['forces'] = perEntryFcs

                del bigSVE
                del bigSVF
                # cp._default_memory_pool.free_all_blocks()
                # cp._default_pinned_memory_pool.free_all_blocks()

        # Now parse the results
        for entryNum, struct in enumerate(database.attrs['structNames']):
            counters = {
                svName: {
                    el: 0 for el in fullPopDict[svName]
                } for svName in fullPopDict
            }

            treeResults = []
            for tree in trees:

                for elem in tree.chemistryTrees:
                    for svNode in tree.chemistryTrees[elem].svNodes:
                        svName = svNode.description

                        idx = counters[svName][elem]

                        eng = results[svName][elem]['energy'][entryNum][idx]
                        fcs = results[svName][elem]['forces'][entryNum][idx]

                        svNode.values = (eng, fcs)

                        counters[svName][elem] += 1


                trueForces = database.trueValues[struct]['forces']
                if useDask:
                    engResult, fcsResult = tree.eval(useDask=True, allSums=allSums)

                    fcsErrors = da.average(
                        abs(sum(fcsResult) - trueForces), axis=(1,2)
                    )
                else:
                    engResult, fcsResult = tree.eval(useDask=False, allSums=allSums)

                    fcsErrors = np.average(
                        abs(sum(fcsResult) - trueForces), axis=(1,2)
                    )

                treeResults.append([sum(engResult), fcsErrors])

            allResults.append(treeResults)

        if useDask:
            from dask.distributed import get_client
            client = get_client()

            allResults = client.gather(client.compute(allResults))

        allResults = np.array(allResults, dtype=np.float32)

        return allResults
