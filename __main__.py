"""
The main entry point for the svreg module.
"""
################################################################################
# Imports

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

import sys
import cma
import h5py
import argparse
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


def main():
    # Prepare usual MPI stuff
    worldComm = MPI.COMM_WORLD
    isMaster = (worldComm.Get_rank() == 0)

    # Load settings
    settings = Settings.from_file(args.settings)
    settings['PROCS_PER_PHYS_NODE'] = args.procs_per_node
    settings['PROCS_PER_MANAGER'] = args.procs_per_manager

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
                'Modify __main__.py to add one.'
        )

    if isMaster:
        regressor = SVRegressor(
            settings, svNodePool, optimizer, optimizerArgs
        )

        regressor.initializeTrees()
        regressor.initializeOptimizers()

    N = settings['optimizerPopSize']

    # Begin optimization
    for regStep in range(settings['numRegressorSteps']):
        for optStep in range(settings['numOptimizerSteps']):
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
            values = evaluator.evaluate(populationDict, evalType='energy')
            energies = regressor.evaluateTrees(values, N)

            # Forces
            values = evaluator.evaluate(populationDict, evalType='forces')
            forces = regressor.evaluateTrees(values, N)

            # Eneriges/forces = {structName: [val for tree in regressor.trees]}

            # TODO: convert to errors
            # TODO: compute cost function
            # TODO: update optimizers for each tree


    print('Done')

if __name__ == '__main__':
    main()