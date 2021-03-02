# Imports
from scipy.interpolate import CubicSpline  # Importing here helps avoid BW issues for some reason
from svreg.tree import SVTree
import os
import time
import h5py
import shutil
import argparse
from mpi4py import MPI
from copy import deepcopy

import random
import numpy as np

from dask_mpi import initialize
initialize(
    nthreads=2,
    memory_limit='4 GB',
    # interface='ipogif0',
    local_directory=os.getcwd()
)

import dask
import dask.array
from dask.distributed import Client, wait

from svreg.archive import Archive, md5Hash
from svreg.settings import Settings
from svreg.database import SVDatabase
from svreg.regressor import SVRegressor
from svreg.evaluator import SVEvaluator
from svreg.population import Population
from svreg.functions import _function_map
from svreg.tree import MultiComponentTree as MCTree

################################################################################
# Parse all command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-s', '--settings', type=str,
    help='The path to the settings file.'
)
parser.add_argument(
    '-t', '--trees', type=str,
    help='The path to a file containing the names of the trees to use.',
)
parser.add_argument(
    '-n', '--names', type=str,
    help='The path to a file containing the names of the structures to use.',
)
parser.add_argument(
    '-l', '--logfile', type=str,
    help='The path to redirect stdout too. If unspecified, prints to stdout',
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
        wait(database.load(h5pyFile, allSums=settings['allSums']))

        evaluator = SVEvaluator(database, settings)

    regressor = SVRegressor(settings, database)
    archive = Archive(os.path.join(settings['outputPath'], 'archive'))

    costFxn = buildCostFunction(settings, len(database.attrs['natoms']))

    # Begin symbolic regression
    if args.trees is not None:
        with open(args.trees, 'r') as f:
            treeNames = [s.strip() for s in f.readlines()]

        regressor.trees = [
            MCTree.from_str(t, database.attrs['elements'], regressor.svNodePool)
            for t in treeNames
        ]

    regressor.initializeTrees(elements=database.attrs['elements'])
    regressor.initializeOptimizers()

    print("Currently optimizing:")

    for pidx, t in enumerate(regressor.trees):
        print(pidx, t)

    print()
    print()

    N = settings['optimizerPopSize']

    rawPopulations  = None
    errors          = None
    costs           = None

    population = Population(
        settings, regressor.svNodePool, database.attrs['elements']
    )

    numCompletedTrees = 0
    maxNumTrees = settings['numRegressorSteps']*settings['numberOfTrees']

    start = time.time()
    fxnEvals = 1
    while numCompletedTrees < maxNumTrees:

        # Remove any converged trees, update population, and print new results
        staleIndices, messages = regressor.checkStale()

        populationChanged = False

        # A tree has finished optimizing
        for staleIdx, staleMessage in zip(staleIndices, messages):
            candidate   = regressor.trees[staleIdx]
            opt         = regressor.optimizers[staleIdx]

            candidate.cost = opt.result.fbest

            # TODO: this might not agree perfectly with opt.result.xbest
            candidateParamsIdx  = np.argmin(costs[staleIdx])
            # candidate.cost      = costs[staleIdx][candidateParamsIdx]
            err                 = errors[staleIdx][candidateParamsIdx]

            print()
            print()
            print("Completed tree {}:".format(staleIdx))
            print("\t", candidate.cost, candidate)
            print("Stopping criterion:", staleMessage)

            numCompletedTrees += 1

            # Log completed tree
            archive.update(
                candidate, candidate.cost, err, opt.result.xbest, opt
            )

            archive.log()

            # Randomly insert into current population
            inserted = population.attemptInsert(candidate)

            if inserted:
                populationChanged = True

            # Replace completed tree with new tree

            # Make sure new tree isn't already in archive or active population
            currentRegNames = [md5Hash(t) for t in regressor.trees]

            newTree, parent1, parent2 = population.newIndividual()

            generatedNew = False
            while not generatedNew:

                inArchive = False
                inReg = False

                for t in regressor.trees:
                    if newTree == t:
                        inReg = True

                for tname in archive:
                    t = archive[tname].tree

                    if newTree == t:
                        inArchive = True

                if inArchive:
                    print("Already in archive:", newTree)
                elif inReg:
                    print("Already being optimized:", newTree)
                else:
                    generatedNew = True
                
                if not generatedNew:
                    newTree, parent1, parent2 = population.newIndividual()

            print("New tree:")
            print('\t', parent1)
            print('\t+')
            print('\t', parent2)
            print('\t=')
            print('\t', newTree)

            # Insert new tree into list of trees being optimized
            argsCopy = deepcopy(regressor.optimizerArgs)
            path = os.path.join(settings['outputPath'], 'outcmaes', '{}/')
            d = {'verb_filenameprefix': path.format(md5Hash(newTree))}
            d.update(regressor.optimizerArgs[-1])
            argsCopy[-1] = d

            newOpt = regressor.optimizer(
                newTree.populate(N=1)[0],
                *argsCopy
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

        # Continue optimization of currently active trees
        populationDict, rawPopulations = regressor.generatePopulationDict(N)

        svEng = evaluator.evaluate(populationDict, 'energy', useDask=False)
        
        for svName in populationDict:
            for el, pop in populationDict[svName].items():
                populationDict[svName][el] = client.scatter(pop)#, broadcast=True)

        svFcs = evaluator.evaluate(populationDict, 'forces')

        perTreeResults = regressor.evaluateTrees(
            svEng, svFcs, N, database.trueValues
        )

        perTreeResults = client.gather(client.compute(perTreeResults))

        energies = {struct: [] for struct in database.attrs['structNames']}
        forces   = {struct: [] for struct in database.attrs['structNames']}

        for structName in database.attrs['structNames'][::-1]:
            for _ in range(len(regressor.trees)):
                res = perTreeResults.pop()

                energies[structName].append(res[0])
                forces[structName].append(res[1])

            energies[structName] = energies[structName][::-1]
            forces[structName] = forces[structName][::-1]

        # Save the (per-struct) errors and the single-value costs
        errors = computeErrors(
            settings['refStruct'], energies, forces, database
        )

        costs = costFxn(errors)

        # Add ridge regression penalty
        penalties = np.array([
            np.linalg.norm(pop, axis=1)*settings['ridgePenalty']
            for pop in rawPopulations
        ])

        # Update optimizers
        regressor.updateOptimizers(
            rawPopulations, costs, penalties
        )

        printTreeCosts(
            fxnEvals,
            [opt.result.fbest for opt in regressor.optimizers],
            penalties,
            start
        )

        fxnEvals += 1

    print('Done')


def polish(client, settings):

    # Setup
    with h5py.File(settings['databasePath'], 'r') as h5pyFile:
        database = SVDatabase(h5pyFile)
        wait(database.load(h5pyFile, allSums=settings['allSums']))

        evaluator = SVEvaluator(database, settings)

    regressor = SVRegressor(settings, database)
    costFxn = buildCostFunction(settings, len(database.attrs['natoms']))

    if args.trees is not None:
        with open(args.trees, 'r') as f:
            treeNames = [s.strip() for s in f.readlines()]

        regressor.trees = [
            MCTree.from_str(t, database.attrs['elements'], regressor.svNodePool)
            for t in treeNames
        ]

    else:
        from svreg.nodes import FunctionNode
        from svreg.tree import MultiComponentTree as MCTree

        tree = MCTree(['Mo', 'Ti'])

        from copy import deepcopy

        treeMo = SVTree()
        treeMo.nodes = [
            FunctionNode('add'),
            deepcopy(regressor.svNodePool[0]),
            deepcopy(regressor.svNodePool[1]),
        ]

        treeTi = SVTree()
        treeTi.nodes = [
            FunctionNode('add'),
            FunctionNode('add'),
            deepcopy(regressor.svNodePool[3]),
            deepcopy(regressor.svNodePool[3]),
            deepcopy(regressor.svNodePool[4]),
        ]

        tree.chemistryTrees['Mo'] = treeMo
        tree.chemistryTrees['Ti'] = treeTi
        
        tree.updateSVNodes()

        regressor.trees = [tree]

    regressor.initializeOptimizers()

    savePath = os.path.join(settings['outputPath'], 'polished')

    if not os.path.isdir(savePath):
        os.mkdir(savePath)

    for tree in regressor.trees:
        print(tree)

    N = settings['optimizerPopSize']

    from svreg.archive import Entry

    entries = {md5Hash(t): Entry(t, savePath) for t in regressor.trees}
    prevBestCosts = {md5Hash(t): np.inf for t in regressor.trees}

    import pickle

    optStart = time.time()
    for optStep in range(1, settings['maxNumOptimizerSteps']+1):

        staleIndices, messages = regressor.checkStale()
        for staleIdx, staleMessage in zip(staleIndices, messages):
            print('Completed tree {}:'.format(staleIdx))
            print(
                "\t",
                regressor.optimizers[staleIdx].result.fbest,
                regressor.trees[staleIdx]
            )
            print("Stopping criterion:", staleMessage)

            del regressor.trees[staleIdx]
            del regressor.optimizers[staleIdx]

        populationDict, rawPopulations = regressor.generatePopulationDict(N)

        svEng = evaluator.evaluate(populationDict, 'energy', useDask=False)

        for svName in populationDict:
            for el, pop in populationDict[svName].items():
                populationDict[svName][el] = client.scatter(pop, broadcast=True)

        svFcs = evaluator.evaluate(populationDict, 'forces')

        perTreeResults = regressor.evaluateTrees(
            svEng, svFcs, N, database.trueValues
        )

        perTreeResults = client.gather(client.compute(perTreeResults))

        energies = {struct: [] for struct in database.attrs['structNames']}
        forces   = {struct: [] for struct in database.attrs['structNames']}

        for structName in database.attrs['structNames'][::-1]:
            for _ in range(len(regressor.trees)):
                res = perTreeResults.pop()

                energies[structName].append(res[0])
                forces[structName].append(res[1])

            energies[structName] = energies[structName][::-1]
            forces[structName] = forces[structName][::-1]

        # Save the (per-struct) errors and the single-value costs
        errors = computeErrors(
            settings['refStruct'], energies, forces, database
        )

        costs = costFxn(errors)

        # Add ridge regression penalty
        penalties = np.array([
            np.linalg.norm(pop, axis=1)*settings['ridgePenalty']
            for pop in rawPopulations
        ])

        # Update optimizers
        regressor.updateOptimizers(rawPopulations, costs, penalties)

        printTreeCosts(
            optStep,
            [opt.result.fbest for opt in regressor.optimizers],
            penalties,
            optStart
        )

        for treeNum, tree in enumerate(regressor.trees):
            opt = regressor.optimizers[treeNum]
            treeName = md5Hash(tree)

            entry = entries[treeName]

            if opt.result.fbest < prevBestCosts[treeName]:
                prevBestCosts[treeName] = opt.result.fbest

                entry.tree = tree
                entry.optimizer = opt
                entry.cost = opt.result.fbest
                entry.bestParams = opt.result.xbest
                entry.bestErrors = errors[0]

            if (optStep % 10) == 0:
                pickle.dump(
                    entry,
                    open(
                        os.path.join(savePath, treeName, 'entry.pkl'),
                        'wb'
                    )
                )

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

    def mae(errors):

        costs = []
        for treeErr in errors:
            costs.append(np.average(np.multiply(treeErr, scaler), axis=1))

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


def computeErrors(refStruct, energies, forces, database, useDask=True):
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
            A list of the total costs for the populations of each tree. Each
            entry will have a shape of (P, 2*S) where P is the population size
            and S is the number of structures being evaluated.
    """

    trueValues = database.trueValues
    natoms = database.attrs['natoms']

    keys = list(energies.keys())
    numTrees   = len(energies[keys[0]])

    keys = list(energies.keys())

    errors = []
    for treeNum in range(numTrees):
        for structName in sorted(keys):

            structEng  = energies[structName][treeNum]
            structEng /= natoms[structName]

            refEng  = energies[refStruct][treeNum]
            refEng /= natoms[refStruct]

            ediff = structEng - refEng

            # Stored true values should already be per-atom energies
            # Note that if the database alreayd did subtract off a reference
            # energy, this won't cause any issues since it will be 0
            trueEdiff = trueValues[structName]['energy']
            trueEdiff -= trueValues[refStruct]['energy']

            engErrors = abs(ediff - trueEdiff)

            fcsErrors = forces[structName][treeNum]

            errors.append(engErrors)
            errors.append(fcsErrors)

    errors = np.stack(errors).T
    errors = np.split(errors, numTrees, axis=1)

    return errors


################################################################################
# Script entry point

if __name__ == '__main__':

    size = MPI.COMM_WORLD.Get_size()

    # Load settings
    settings = Settings.from_file(args.settings)

    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    print("Current settings:\n")
    settings.printSettings()

    if settings['allSums']:
        for key in _function_map:
            if key != 'add':
                raise RuntimeError(
                    "allSums == True, but function map includes other functions"
                )

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
        print()

        client.wait_for_workers(n_workers=(size-2)//100*100)

        if args.logfile is not None:
            # Redirects stdout to a logfile instead of 
            from contextlib import redirect_stdout

            logfile = os.path.join(settings['outputPath'], args.logfile)

            print("Redirecting stdout to '{}'".format(logfile))

            with open(logfile, 'w') as f:
                with redirect_stdout(f):
                    # Begin run
                    if settings['runType'] == 'GA':
                                main(client, settings)
                    else:
                        polish(client, settings)
        else:
            # Begin run
            if settings['runType'] == 'GA':
                        main(client, settings)
            else:
                polish(client, settings)