# Imports
from tree import SVTree
import os
import time
import h5py
import shutil
import argparse
from mpi4py import MPI

import random
import numpy as np

from dask_mpi import initialize
initialize(
    nthreads=2,
    memory_limit='4 GB',
    # interface='ipogif0',
    # local_directory='/u/sciteam/vita/scratch/svreg/hyojung/hj_dask_prof_int',
)

import dask
import dask.array
from dask_jobqueue import PBSCluster
from dask.distributed import Client, wait

from archive import Archive
from settings import Settings
from database import SVDatabase
from regressor import SVRegressor
from evaluator import SVEvaluator
from population import Population

################################################################################
# Parse all command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--settings', type=str,
    help='The path to the settings file.'
)
parser.add_argument(
    '-ppn', '--procs_per_node', type=int,
    help='The number of cores on each physical compute node'
)
parser.add_argument(
    '-ppm', '--procs_per_manager', type=int,
    help='Used for tuning the parallelization. Given S structures, M'\
        'processors per manager, and C processors per compute node, the number'\
            'of structures loaded onto each compute node is equal to S/(M/C)'\
                '(approximately, accounting for imperfect division).'
)
parser.add_argument(
    '-n', '--names', type=str,
    help='The path to a file containing the names of the structures to use.',
)

args = parser.parse_args()
################################################################################
# Main functions

start = time.time()

# @profile
def main(client, settings):
    global start

    # Setup
    with h5py.File(settings['databasePath'], 'r') as h5pyFile:
        database = SVDatabase(h5pyFile)

        evaluator = SVEvaluator(client, database, settings)

    regressor = SVRegressor(settings, database)
    archive = Archive(os.path.join(settings['outputPath'], 'archive'))

    costFxn = buildCostFunction(settings, len(database.attrs['natoms']))

    # Begin symbolic regression
    regressor.initializeTrees(elements=database.attrs['elements'])
    regressor.initializeOptimizers()

    N = settings['optimizerPopSize']

    rawPopulations  = None
    errors          = None
    costs           = None

    population = Population(settings, regressor.svNodePool)
    archive = Archive(os.path.join(settings['outputPath'], 'archive'))

    numCompletedTrees = 0
    maxNumTrees = settings['numRegressorSteps']*settings['numberOfTrees']

    start = time.time()
    fxnEvals = 1
    while numCompletedTrees < maxNumTrees:

        # Continue optimization of currently active trees
        populationDict, rawPopulations = regressor.generatePopulationDict(N)

        svEng = evaluator.evaluate(populationDict, 'energy')

        # for svName in populationDict:
        #     for el, pop in populationDict[svName].items():
        #         populationDict[svName][el] = client.scatter(pop, broadcast=True)

        svFcs = evaluator.evaluate(populationDict, 'forces')

        energies, forces = regressor.evaluateTrees(svEng, svFcs, N)

        # Save the (per-struct) errors and the single-value costs
        errors = computeErrors(
            client, settings['refStruct'], energies, forces, database
        )

        costs = costFxn(errors)

        # Add ridge regression penalty
        penalties = np.array([
            np.linalg.norm(pop, axis=1)*settings['ridgePenalty']
            for pop in rawPopulations
        ])

        # Update optimizers
        updatedOpts = regressor.updateOptimizers(
            rawPopulations, costs, penalties
        )

        regressor.optimizers = client.gather(client.compute(updatedOpts))

        printTreeCosts(
            fxnEvals,
            [opt.result.fbest for opt in regressor.optimizers],
            penalties,
            start
        )

        fxnEvals += 1

        # Remove any converged trees, update population, and print new results
        staleIndices = regressor.checkStale()

        populationChanged = False

        # A tree has finished optimizing
        for staleIdx in staleIndices:
            candidate   = regressor.trees[staleIdx]
            opt         = regressor.optimizers[staleIdx]

            candidate.cost = opt.result.fbest

            numCompletedTrees += 1

            # Log completed tree
            archive.update(
                candidate, candidate.cost, opt.result.xbest, opt
            )

            archive.log()

            # Randomly insert into current population
            inserted = population.attemptInsert(candidate)

            if inserted:
                populationChanged = True

            # Replace completed tree with new tree

            # Make sure new tree isn't already in archive or active population
            currentRegNames = [str(t) for t in regressor.trees]

            newTree = population.newIndividual()

            generatedNew = False
            while not generatedNew:

                treeName = str(newTree)

                inArchive   = treeName in archive
                inReg       = treeName in currentRegNames

                if (not inArchive) and (not inReg):
                    generatedNew = True
                else:
                    newTree = population.newIndividual()

            # Insert new tree into list of trees being optimized
            newOpt = regressor.optimizer(
                newTree.populate(N=1)[0],
                *regressor.optimizerArgs
            )

            regressor.trees[staleIdx]        = newTree
            regressor.optimizers[staleIdx]   = newOpt

        if staleIndices:
            if populationChanged:
                # Print current population if it was updated
                print()
                print()
                print("Current population:")

                popCosts = [t.cost for t in population]
                argsort = np.argsort(popCosts)

                for idx in argsort:
                    print(population[idx].cost, population[idx])

                print()
            else:
                print()
                print()
                print("No new fitted trees were added to the population.")
                print()

        if staleIndices:
            print()
            print("Currently optimizing:")

            for pidx, t in enumerate(regressor.trees):
                print(pidx, t)
            print()
            print()

    print('Done')


def polish(client, settings):

    # Setup
    with h5py.File(settings['databasePath'], 'r') as h5pyFile:
        database = SVDatabase(h5pyFile)

        evaluator = SVEvaluator(client, database, settings)

        regressor = SVRegressor(settings, database)

    from nodes import FunctionNode
    from tree import MultiComponentTree as MCTree
    tree = MCTree(['Mo', 'Ti'])

    from copy import deepcopy

    treeMo = SVTree()
    treeMo.nodes = [
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[0]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[1]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[2]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[3]),
        deepcopy(regressor.svNodePool[4]),
    ]

    treeTi = SVTree()
    treeTi.nodes = [
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[0]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[1]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[2]),
        FunctionNode('add'),
        deepcopy(regressor.svNodePool[3]),
        deepcopy(regressor.svNodePool[4]),
    ]

    tree.chemistryTrees['Mo'] = treeMo
    tree.chemistryTrees['Ti'] = treeTi
    
    print(tree)

    tree.updateSVNodes()

    regressor.trees = [tree]
    regressor.initializeOptimizers()

    costFxn = buildCostFunction(settings, len(database.attrs['natoms']))

    N = settings['optimizerPopSize']

    optStart = time.time()
    for optStep in range(settings['numOptimizerSteps']):

        populationDict, rawPopulations = regressor.generatePopulationDict(N)

        # svEng = evaluator.evaluate(populationDict, 'energy')
        # print("Evaluated energies... {}".format(time.time() - start), flush=True)

        # futures = []
        # for svName in populationDict:
        #     for el, pop in populationDict[svName].items():
        #         populationDict[svName][el] = client.scatter(pop)
        #         futures.append(populationDict[svName][el])

        # from dask.distributed import wait
        # wait(futures)

        svEng = evaluator.evaluate(populationDict, 'energy')
        svFcs = evaluator.evaluate(populationDict, 'forces')

        # svResults = evaluator.evaluate(populationDict)
        # svEng = svResults['energy']; svFcs = svResults['forces']

        energies, forces = regressor.evaluateTrees(svEng, svFcs, N)

        results = {'energies': energies, 'forces': forces}

        # Save the (per-struct) errors and the single-value costs
        errors = computeErrors(
            client, settings['refStruct'], energies, forces, database
            # client, settings['refStruct'], results, database
        )

        costs = costFxn(errors)

        # Add ridge regression penalty
        penalties = np.array([
            np.linalg.norm(pop, axis=1)*settings['ridgePenalty']
            for pop in rawPopulations
        ])

        printTreeCosts(optStep, costs, penalties, optStart)

        # Update optimizers
        regressor.updateOptimizers(rawPopulations, costs, penalties)

################################################################################
# Helper functions

def printTreeCosts(optStep, costs, penalties, startTime):
    n = len(costs)

    firstN = costs[:n]
    firstNPen = penalties[:n]

    string = '\t\t'

    for c, p in zip(firstN, firstNPen):
        # argmin = np.argmin(c)
        # string += '{:.4f} ({:.4f})\t'.format(c[argmin], p[argmin])
        # string += '{:.6f}\t'.format(c[argmin])
        string += '{:.6f}\t'.format(c)

    print(
        optStep,
        '({:.4f} s)'.format(time.time() - startTime),
        string,
        flush=True
    )


def buildCostFunction(settings, numStructs):
    """
    A function factory for building different types of cost functions. Assumes
    that the cost function will take in a list of (P, S) arrays, where P is the
    population size and S is the number of structures. The cost function will
    return a single value for each entry in the list.

    It is assumed that energy errors are absolute values, and tree errors are
    MAE values (to reduce each structure to a single value regarless of the
    number of atoms).
    """

    scaler = np.ones(2*numStructs)
    scaler[::2]  *= settings['energyWeight']
    scaler[1::2] *= settings['forcesWeight']
        
    @dask.delayed
    def delayedMAE(err):
        return np.average(np.multiply(err, scaler), axis=1)

    # @profile
    def mae(errors):

        costs = []
        for treeErr in errors:
            # costs.append(np.average(np.multiply(treeErr, scaler), axis=1))
            costs.append(delayedMAE(treeErr))
            # costs.append(
            #     np.average(np.multiply(np.stack(treeErr).T, scaler), axis=1)
            # )

        # return np.array(costs)
        return costs

    def rmse(errors):
        
        costs = []
        for err in errors:
            tmp = np.array(dask.compute(err))
            tmp[:, ::2]  *= settings['energyWeight']
            tmp[:, 1::2] *= settings['forcesWeight']
            costs.append(np.sqrt(np.average(tmp**2, axis=1)))

        return np.array(costs)

    if settings['costFxn'] == 'MAE':
        return mae
    elif settings['costFxn'] == 'RMSE':
        return rmse
    else:
        raise RuntimeError("costFxn must be 'MAE' or 'RMSE'.")


# @profile
def computeErrors(client, refStruct, energies, forces, database):
# def computeErrors(client, refStruct, results, database):
    """
    Takes in dictionaries of energies and forces and returns the energy/force
    errors for each structure for each tree. Sorts structure names first.

    Args:
        refStruct (str):
            The name of the reference structure for computing energy
            differences.

        energies (dict):
            {structName: [(P,) for tree in trees]} where P is population size.

        energies (dict):
            {structName: [(P, 3, N) for tree in trees]} where P is population
            size and N is number of atoms.

    Returns:
        costs (list):
            A list of the total costs for the populations of each tree. Eeach
            entry will have a shape of (P, 2*S) where P is the population size
            and S is the number of structures being evaluated.
    """

    global start

    trueValues = database.trueValues
    natoms = database.attrs['natoms']
    elements = database.attrs['elements']

    keys = list(energies.keys())
    numTrees   = len(energies[keys[0]])
    # numPots    = energies[keys[0]][0][0].shape[0] # [struct][tree][elem]
    # numStructs = len(keys)

    keys = list(energies.keys())

    @dask.delayed
    def delayedAvg(err):
        # return np.average(err, axis=(1,2))
        return dask.array.average(err, axis=(1,2))

    @dask.delayed
    def delayedStack(err):
        return dask.array.stack(err)

    @dask.delayed
    def delayedStackT(err):
        return dask.array.stack(err).T

    errors = []
    for treeNum in range(numTrees):
        treeErrors = []
        for structName in sorted(keys):

            structEng  = np.sum(energies[structName][treeNum], axis=0)
            structEng /= natoms[structName]

            refEng  = np.sum(energies[refStruct][treeNum], axis=0)
            refEng /= natoms[refStruct]

            ediff = structEng - refEng

            engErrors = abs(ediff - trueValues[structName]['energy'])

            fcs =   forces[structName][treeNum]

            fcsErrors = [
                fcs[ii] - trueValues[structName]['forces_'+el]
                for ii, el in enumerate(elements)
            ]

            fcsErrors = [abs(err) for err in fcsErrors]
            # fcsErrors = [np.average(err, axis=(1, 2)) for err in fcsErrors]
            # fcsErrors = [delayedAvg(err) for err in fcsErrors]
            fcsErrors = [dask.array.average(err, axis=(1,2)) for err in fcsErrors]

            treeErrors.append(engErrors)
            treeErrors.append(sum(fcsErrors))

        # errors.append(np.stack(treeErrors))
        # errors.append(delayedStack(treeErrors))
        errors.append(dask.array.stack(treeErrors))

    # errors = [np.stack(err).T for err in errors]
    # errors = [delayedStackT(err) for err in errors]
    errors = [dask.array.transpose(dask.array.stack((err))) for err in errors]

    return errors


################################################################################
# Script entry point

if __name__ == '__main__':

    size = MPI.COMM_WORLD.Get_size()

    # Load settings
    settings = Settings.from_file(args.settings)
    settings['PROCS_PER_PHYS_NODE'] = args.procs_per_node
    settings['PROCS_PER_MANAGER'] = args.procs_per_manager

    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    print("Current settings:\n")
    settings.printSettings()

    # Prepare save directories
    if os.path.isdir(settings['outputPath']):
        if not settings['overwrite']:
            raise RuntimeError(
                'Save folder "{}" already exists,'\
                ' but `overwrite` is set to False'.format(
                    settings['outputPath']
                    )
            )
        else:
            shutil.rmtree(settings['outputPath'])

    os.mkdir(settings['outputPath'])

    with Client() as client:

        print()
        print(
            'Dask dashboard info: {}'.format(
                client.scheduler_info()['address'],
                client.scheduler_info()['services'],
            ),
            flush=True
        )
        print()

        num = (size-2)//100*100
        print(
            "Waiting for at least {} workers to connect before starting...".format(num),
            flush=True
        )

        client.wait_for_workers(n_workers=(size-2)//100*100)

        # Begin run
        if settings['runType'] == 'GA':
            main(client, settings)
        else:
            polish(client, settings)
