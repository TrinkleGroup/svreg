# Imports
from tree import SVTree
import matplotlib.pyplot as plt
import os
import time
import h5py
import shutil
import argparse

import random
import numpy as np

from dask_mpi import initialize
initialize(
    nthreads=2,
    memory_limit='2 GB',
    #interface='ipogif0',
)

import dask
import dask.array
from dask_jobqueue import PBSCluster
from dask.distributed import Client, LocalCluster

from archive import Archive
from settings import Settings
from database import SVDatabase
from regressor import SVRegressor
from evaluator import SVEvaluator

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

# @profile
def main(client, settings):

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

    regStart = time.time()
    for regStep in range(settings['numRegressorSteps']):

        regressor.printTop10Header(regStep)

        optStart = time.time()
        for optStep in range(settings['numOptimizerSteps']):
            populationDict, rawPopulations = regressor.generatePopulationDict(N)

            if optStep == 0:
                print("Total population shapes:")

                for svName in populationDict:
                    for elem in populationDict[svName]:
                        print(svName, elem, populationDict[svName][elem].shape)

            svResults = evaluator.evaluate(populationDict)

            energies, forces = regressor.evaluateTrees(svResults, N)

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

            printTreeCosts(optStep, costs, penalties, optStart)

            # Update optimizers
            regressor.updateOptimizers(rawPopulations, costs, penalties)

        # Update archive once the optimizers have finished running
        archive.update(
            regressor.trees, costs, errors, rawPopulations,
            regressor.optimizers
        )

        archive.log()
        archive.printStatus([str(t) for t in regressor.trees])

        if regStep + 1 < settings['numRegressorSteps']:

            # Sample tournamentSize number of trees to use as parents
            sampledTrees, opts = archive.sample(settings['tournamentSize'])
            regressor.trees = sampledTrees

            # Finish populating trees by mating/mutating
            newTrees = regressor.evolvePopulation()

            # Remove duplicate trees, and load remaining from archive
            uniqueTrees, uniqueOptimizers = archive.pruneAndLoad(
                sampledTrees, newTrees, opts, regressor
            )

            # # Now pull old trees to ensure population size is the same
            # keys = list(self.keys())
            # while len(uniqueTrees) < self.settings['numberOfTrees']:
            #     randomTree = random.choice(keys)
            #     if randomTree not in uniqueTrees:
            #         uniqueTrees.append(archive[randomTree].tree)
            #         uniqueOptimizers.append(archive[randomTree].optimizer)

            regressor.trees = uniqueTrees
            regressor.optimizers = uniqueOptimizers

    print('Done')


def polish(client, settings):

    # Setup
    with h5py.File(settings['databasePath'], 'r') as h5pyFile:
        database = SVDatabase(h5pyFile)

        evaluator = SVEvaluator(client, database, settings)

    regressor = SVRegressor(settings, database)

    costFxn = buildCostFunction(settings, len(database.attrs['natoms']))

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

    tree.updateSVNodes()

    regressor.trees = [tree]
    regressor.initializeOptimizers()

    N = settings['optimizerPopSize']

    optStart = time.time()
    for optStep in range(settings['numOptimizerSteps']):
        populationDict, rawPopulations = regressor.generatePopulationDict(N)

        if optStep == 0:
            print("Total population shapes:")

            for svName in populationDict:
                for elem in populationDict[svName]:
                    print(svName, elem, populationDict[svName][elem].shape)

        svResults = evaluator.evaluate(populationDict)

        energies, forces = regressor.evaluateTrees(svResults, N)

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

        printTreeCosts(optStep, costs, penalties, optStart)

        # Update optimizers
        regressor.updateOptimizers(rawPopulations, costs, penalties)



################################################################################
# Helper functions

def printTreeCosts(optStep, costs, penalties, startTime):
    first10 = costs[:10]
    first10Pen = penalties[:10]

    string = '\t\t'

    for c, p in zip(first10, first10Pen):
        argmin = np.argmin(c)
        string += '{:.2f} ({:.2f})\t'.format(c[argmin], p[argmin])

    print(
        optStep,
        '({:.2f} s)'.format(time.time() - startTime),
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
        
    # @profile
    def mae(errors):

        costs = []
        for treeErr in errors:
            costs.append(np.average(np.multiply(treeErr, scaler), axis=1))
            # costs.append(
            #     np.average(np.multiply(np.stack(treeErr).T, scaler), axis=1)
            # )

        return np.array(costs)

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


    trueValues = database.trueValues
    natoms = database.attrs['natoms']
    elements = database.attrs['elements']

    keys = list(energies.keys())
    numTrees   = len(energies[keys[0]])
    # numPots    = energies[keys[0]][0][0].shape[0] # [struct][tree][elem]
    # numStructs = len(keys)

    keys = list(energies.keys())
    errors = []

    @dask.delayed
    def delayedAverage(err):
        return dask.array.mean(err, axis=(1, 2))

    errors = []
    for treeNum in range(numTrees):
        treeErrors = []
        for structName in sorted(keys):
            eng = energies[structName][treeNum]/natoms[structName] \
                - energies[refStruct][treeNum]/natoms[refStruct]

            fcs =   forces[structName][treeNum]

            totalEng = sum([eng[ii] for ii in range(len(elements))])
            engErrors = totalEng - trueValues[structName]['energy']

            fcsErrors = [
                fcs[ii] - trueValues[structName]['forces_'+el]
                for ii, el in enumerate(elements)
            ]

            fcsErrors = [abs(err) for err in fcsErrors]
            fcsErrors = [delayedAverage(err) for err in fcsErrors]

            treeErrors.append(engErrors)
            treeErrors.append(sum(fcsErrors))

        errors.append(treeErrors)

    errors = list(dask.compute(errors)[0])
    errors = [np.stack(err).T for err in errors]

    return errors


################################################################################
# Script entry point

if __name__ == '__main__':
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

    # Start Dask client
    # with LocalCluster(
    #     n_workers=4,
    #     processes=True,  # default; need to test out False
    #     threads_per_worker=2,
    #     # worker_dashboard_address='40025'
    # with PBSCluster(
    #         queue='normal',
    #         project='bbas',
    #         local_directory=os.getcwd(),
    #         cores=16,
    #         memory='64 GB',
    #         python='aprun -n 1 -N 1 python',
    #         resource_spec='nodes=1:ppn=32:xe',
    #         env_extra=[
    #             'cd /scratch/sciteam/$USER/svreg/hyojung/$PBS_JOBNAME',
    #             'export PATH=$PATH:/scratch/sciteam/$USER/svreg/hyojung/$PBS_JOBNAME',
    #             'source ~/bin/svregEnv',
    #         ]
    #         
    # ) as cluster, Client(cluster) as client:
    with Client() as client:

        print()
        print(
            'Dask dashboard info: {}'.format(
                client.scheduler_info()
            ),
            flush=True
        )
        print()

        # Begin run
        if settings['runType'] == 'GA':
            main(client, settings)
        else:
            polish(client, settings)
