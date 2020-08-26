"""
The main entry point for the svreg module.

TODO: change this into a GA class, then make a new __main__.py
"""
################################################################################
# Imports

import sys
import cma
import time
import h5py
import random
import argparse
import numpy as np
from mpi4py import MPI

from settings import Settings
from database import SVDatabase
from regressor import SVRegressor
from evaluator import SVEvaluator
from nodes import SVNode


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


@profile
def main():
    # Prepare usual MPI stuff
    worldComm = MPI.COMM_WORLD
    isMaster = (worldComm.Get_rank() == 0)

    # Load settings
    settings = Settings.from_file(args.settings)
    settings['PROCS_PER_PHYS_NODE'] = args.procs_per_node
    settings['PROCS_PER_MANAGER'] = args.procs_per_manager

    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    # Load database, build evaluator, and prepare list of structNames
    with SVDatabase(settings['databasePath'], 'r') as database:
        if args.names:
            # Read in structure names if available
            with open(args.names, 'r') as f:
                lines = f.readlines()
                structNames = [line.strip() for line in lines]
        else:
            # Otherwise just use all structures in the database
            structNames = list(database.keys())
        
        if settings['refStruct'] not in structNames:
            raise RuntimeError(
                "The reference structure must be included in structNames."
            )

        evaluator = SVEvaluator(worldComm, structNames, settings)
        evaluator.distributeDatabase(database)

        # Prepare svNodePool for use in tree construction
        svNodePool = []
        group = database[settings['refStruct']]
        for svName in group:
            svGroup = group[svName]
            svNodePool.append(
                SVNode(
                    description=svName,
                    numParams=svGroup.attrs['numParams'],
                    paramRange=svGroup.attrs['paramRange']\
                        if 'paramRange' in group[svName].attrs else None
                )
            )

        trueValues = database.loadTrueValues()
    
    # Prepare regressor
    if settings['optimizer'] == 'CMA':
        optimizer = cma.CMAEvolutionStrategy
        optimizerArgs = [
            1.0,  # defaulted sigma0 value
            {'verb_disp': 0, 'popsize':settings['optimizerPopSize']}
        ]
    else:
        raise NotImplementedError(
            'The only available optimizer is CMA.'\
                'Modify ga.py to add one.'
        )

    if isMaster:
        regressor = SVRegressor(
            settings, svNodePool, optimizer, optimizerArgs
        )

        regressor.initializeTrees()
        print()

    N = settings['optimizerPopSize']

    # Begin optimization
    start = time.time()
    for regStep in range(settings['numRegressorSteps']):
        if isMaster:
            print(regStep, flush=True)

            # Optimizers need to be re-initialized when trees change
            regressor.initializeOptimizers()


        for optStep in range(settings['numOptimizerSteps']):
            # TODO: is it an issue that this can give diff fits for same tree?
            if isMaster:
                rawPopulations = [
                    np.array(opt.ask(N)) for opt in regressor.optimizers
                ]

                # Need to parse populations so that they can be grouped easily
                # treePopulation = [{svName: population} for tree in trees]
                treePopulations = []
                for pop, tree in zip(rawPopulations, regressor.trees):
                    treePopulations.append(tree.parseArr2Dict(pop))

                # Group all dictionaries into one
                # {svName: [tree populations for tree in trees]}
                populationDict = {
                    svName: [] for svName in [n.description for n in svNodePool]
                }

                for treeDict in treePopulations:
                    for svName, pop in treeDict.items():
                        populationDict[svName].append(pop)

            else:
                populationDict = None

            # Energies
            svEng = evaluator.evaluate(populationDict, evalType='energy')
            svFcs = evaluator.evaluate(populationDict, evalType='forces')

            if isMaster:
                # Eneriges/forces = {structName: [pop for tree in trees]}
                energies, forces = regressor.evaluateTrees(svEng, svFcs, N)

                costs = cost(energies, forces, trueValues)

                # Print the cost of the best paramaterization of the best tree
                # print('\t', optStep, min([min(c) for c in costs]))

                # Update optimizers
                for treeIdx in range(len(regressor.optimizers)):
                    opt = regressor.optimizers[treeIdx]
                    opt.tell(rawPopulations[treeIdx], costs[treeIdx])

        if isMaster:
            # Set tree costs as the lowest costs at the final Optimizer step
            bestCosts = [min(c) for c in costs]
            for tree, c in zip(regressor.trees, bestCosts):
                tree.cost = c

            for t in sorted(regressor.trees, key=lambda t: t.cost):
                print('\t{:.2f}'.format(t.cost), t)

            print(flush=True)

            if regStep + 1 < settings['numRegressorSteps']:
                regressor.evolvePopulation(svNodePool)


    print('Done')


def cost(energies, forces, trueValues):
    """
    Takes in dictionaries of energies and forces and returns a single cost
    value. Currently uses MAE.

    Args:
        energies (dict):
            {structName: [(P,) for tree in trees]} where P is population size.

        energies (dict):
            {structName: [(P, 3, N) for tree in trees]} where P is population
            size and N is number of atoms.

        trueValues (dict):
            {structName: {'energy': eng, 'forces':fcs}}

    Returns:
        costs (list):
            A list of the total costs for the populations of each tree.
    """

    keys = list(energies.keys())
    numTrees   = len(energies[keys[0]])
    numPots    = energies[keys[0]][0].shape[0]
    numStructs = len(keys)

    keys = list(energies.keys())
    costs = []

    for treeNum in range(numTrees):
        treeCosts = np.zeros((numPots, 2*numStructs))
        for i, structName in enumerate(keys):
            eng = energies[structName][treeNum]
            fcs =   forces[structName][treeNum]

            engErrors = eng - trueValues[structName]['energy']
            fcsErrors = fcs - trueValues[structName]['forces']

            treeCosts[:,   i] = abs(engErrors)
            treeCosts[:, 2*i] = np.average(np.abs(fcsErrors), axis=(1, 2))

        costs.append(treeCosts.sum(axis=1))

    return costs


if __name__ == '__main__':
    main()