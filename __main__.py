"""
The main entry point for the svreg module.

TODO: change this into a GA class, then make a new __main__.py
"""
################################################################################
# Imports

import os
import sys
import cma, comocma
import time
import h5py
import shutil
import pickle
import random
import argparse
import numpy as np
from mpi4py import MPI
from copy import deepcopy

from settings import Settings
from database import SVDatabase
from regressor import SVRegressor
from evaluator import SVEvaluator
from nodes import SVNode, FunctionNode
from tree import SVTree
from archive import Archive, Entry
from optimizers import GAWrapper, SofomoreWrapper


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


# @profile
def main(settings, worldComm, isMaster):
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

        if isMaster:
            print("Loaded structures:")
            for n in structNames:
                print('\t', n)
        
        if settings['refStruct'] not in structNames:
            raise RuntimeError(
                "The reference structure must be included in structNames."
            )

        evaluator = SVEvaluator(worldComm, structNames, settings)
        evaluator.distributeDatabase(database)

        svNodePool = buildSVNodePool(database[settings['refStruct']])
        trueValues = database.loadTrueValues()

        numStructs = len(trueValues)
    
    # Prepare regressor
    if settings['optimizer'] == 'CMA':
        optimizer = cma.CMAEvolutionStrategy
        optimizerArgs = [
            1.0,  # defaulted sigma0 value
            {'verb_disp': 0, 'popsize':settings['optimizerPopSize']}
        ]
    elif settings['optimizer'] == 'GA':
        optimizer = GAWrapper
        optimizerArgs = [
            {
                'verb_disp': 0,
                'popsize': settings['optimizerPopSize'],
                'pointMutateProb': settings['pointMutateProb']
            }
        ]
    elif settings['optimizer'] == 'Sofomore':
        optimizer = SofomoreWrapper
        optimizerArgs = {
            'numStructs': numStructs,
            'paretoDimensionality': 2,
            'CMApopSize': settings['optimizerPopSize'],
            'SofomorePopSize': settings['numberOfTrees'],  # Placeholder
            # 'threads_per_node': settings['PROCS_PER_PHYS_NODE'],
            'threads_per_node': None,
        }
    else:
        raise NotImplementedError('Must be one of `GA`, `CMA`, or `Sofomore`.')

    if isMaster:
        archive = Archive(os.path.join(settings['outputPath'], 'archive'))

        regressor = SVRegressor(
            settings, svNodePool, optimizer, optimizerArgs
        )

        regressor.initializeTrees()
        regressor.initializeOptimizers()
        print()

    N = settings['optimizerPopSize']

    # Begin optimization
    start = time.time()
    for regStep in range(settings['numRegressorSteps']):
        if isMaster:
            print(regStep, flush=True)

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
                # {svName: {bondType: [tree populations for tree in trees]}}
                # populationDict = {
                #     svName: {
                #         [] for svName in [n.description for n in svNodePool]
                #     }
                # }

                populationDict = {}
                for svNode in svNodePool:
                    populationDict[svNode.description] = {}
                    for bondType in svNode.bonds:
                        populationDict[svNode.description][bondType] = []

                for treeDict in treePopulations:
                    for svName in treeDict.keys():
                        for bondType, pop in treeDict[svName].items():
                            populationDict[svName][bondType].append(pop)

            else:
                populationDict = None

            # Distribute energy/force calculations across compute nodes
            svEng = evaluator.evaluate(populationDict, evalType='energy')
            svFcs = evaluator.evaluate(populationDict, evalType='forces')

            if isMaster:
                # Eneriges/forces = {structName: [pop for tree in trees]}
                energies, forces = regressor.evaluateTrees(svEng, svFcs, N)

                # Save the (per-struct) errors and the single-value costs
                errors = computeErrors(
                    settings['refStruct'], energies, forces, trueValues
                )

                costs = [el.sum(axis=1) for el in errors]

                # Print the cost of the best paramaterization of the best tree
                print(
                    '\t', optStep, min([np.min(c) for c in costs]), flush=True
                )

                # Update optimizers
                for treeIdx in range(len(regressor.optimizers)):
                    opt = regressor.optimizers[treeIdx]
                    opt.tell(rawPopulations[treeIdx], costs[treeIdx])

        if isMaster:
            archive.update(
                regressor.trees, costs, errors, rawPopulations,
                regressor.optimizers
            )

            archive.log()

            print()
            printNames = list(archive.keys())
            printCosts = [archive[n].cost for n in printNames]
            regressorNames = [str(t) for t in regressor.trees]

            for idx in np.argsort(printCosts):
                indicator = ''
                if printNames[idx] in regressorNames:
                    indicator = '->'

                print(
                    '\t{}{:.2f}'.format(indicator, printCosts[idx]),
                    printNames[idx],
                )
            print(flush=True)

            if regStep + 1 < settings['numRegressorSteps']:

                # Sample tournamentSize number of trees to use as parents
                trees, opts = archive.sample(settings['tournamentSize'])
                regressor.trees = trees

                # Finish populating trees by mating/mutating
                newTrees = regressor.evolvePopulation(svNodePool)

                # Check if each tree is in the archive
                currentTreeNames = [str(t) for t in trees]

                uniqueTrees = trees
                uniqueOptimizers = opts
                for tree in newTrees:
                    treeName = str(tree)
                    if treeName not in currentTreeNames:
                        # Not a duplicate
                        uniqueTrees.append(tree)

                        if treeName in archive:
                            # Load archived optimizer
                            uniqueOptimizers.append(archive[treeName].optimizer)
                        else:
                            # Create new optimizer
                            uniqueOptimizers.append(
                                regressor.optimizer(
                                    tree.populate(N=1)[0],
                                    *regressor.optimizerArgs
                                )
                            )

                regressor.trees = uniqueTrees
                regressor.optimizers = uniqueOptimizers

    print('Done')


def fixedExample(settings, worldComm, isMaster):
    """A function for running basic tests on example trees."""

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

        svNodePool = buildSVNodePool(database[settings['refStruct']])
        trueValues = database.loadTrueValues()
    
        numStructs = len(trueValues)

    # Prepare regressor
    if settings['optimizer'] == 'CMA':
        optimizer = cma.CMAEvolutionStrategy
        optimizerArgs = [
            100.0,  # defaulted sigma0 value
            {'verb_disp': 1, 'popsize':settings['optimizerPopSize']}
        ]
    elif settings['optimizer'] == 'GA':
        optimizer = GAWrapper
        optimizerArgs = [
            {
                'verb_disp': 1,
                'popsize': settings['optimizerPopSize'],
                'pointMutateProb': settings['pointMutateProb']
            }
        ]
    elif settings['optimizer'] == 'Sofomore':
        optimizer = SofomoreWrapper
        optimizerArgs = {
            'numStructs': numStructs,
            'structNames': structNames,
            'paretoDimensionality': 2,
            'CMApopSize': settings['optimizerPopSize'],
            'SofomorePopSize': settings['numberOfTrees'],  # Placeholder
            'threads_per_node': settings['PROCS_PER_PHYS_NODE'],
            'opts': {'verb_disp': 1}
        }
    else:
        raise NotImplementedError('Must be one of `GA`, `CMA`, or `Sofomore`.')

    if isMaster:
        tree = SVTree()

        ffgNode = svNodePool[0]
        rhoNode = svNodePool[1]

        tree.nodes = [
            FunctionNode('mul'),
            FunctionNode('sqrt'),
            FunctionNode('sqrt'), deepcopy(rhoNode), deepcopy(rhoNode)
            # FunctionNode('inv'), deepcopy(ffgNode),
            # FunctionNode('log'), deepcopy(rhoNode),
            # FunctionNode('add'), deepcopy(rhoNode),
            # FunctionNode('mul'), deepcopy(rhoNode), deepcopy(ffgNode)
        ]

        tree.svNodes = [n for n in tree.nodes if isinstance(n, SVNode)]

        # archive = pickle.load(
        #     open(
        #         os.path.join('results', 'archive', 'archive.pkl'),
        #         'rb'
        #     )
        # )

        # entry = archive['mul(add(inv(ffg), log(rho)), add(rho, mul(rho, ffg)))']
        # tree = entry.tree

        regressor = SVRegressor(
            settings, svNodePool, optimizer, optimizerArgs
        )

        regressor.trees = [tree]
        regressor.initializeOptimizers()

        print('Tree:\n\t', regressor.trees[0], flush=True)

    if settings['optimizer'] == 'Sofomore':
        # Sofomore.ask() returns the incumbents of each kernels,
        # plus the populations of each kernel
        N = settings['numberOfTrees']*(1 + settings['optimizerPopSize'])
    else:
        N = settings['optimizerPopSize']

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
            populationDict = {}
            for svNode in svNodePool:
                populationDict[svNode.description] = {}
                for bondType in svNode.bonds:
                    populationDict[svNode.description][bondType] = []

            for treeDict in treePopulations:
                for svName in treeDict.keys():
                    for bondType, pop in treeDict[svName].items():
                        populationDict[svName][bondType].append(pop)

        else:
            populationDict = None

        # Distribute energy/force calculations across compute nodes
        svEng = evaluator.evaluate(populationDict, evalType='energy')
        svFcs = evaluator.evaluate(populationDict, evalType='forces')

        if isMaster:
            # Eneriges/forces = {structName: [pop for tree in trees]}
            energies, forces = regressor.evaluateTrees(svEng, svFcs, N)

            errors = computeErrors(
                settings['refStruct'], energies, forces, trueValues
            )


            costs = np.array([el.sum(axis=1) for el in errors])

            # Add ridge regression penalty
            penalties = np.array([
                np.linalg.norm(pop, axis=1)*settings['ridgePenalty']
                for pop in rawPopulations
            ])

            # Print the cost of the best paramaterization of the best tree
            # print('\t', optStep, min([min(c) for c in costs]), flush=True)

            # Update optimizers
            for treeIdx in range(len(regressor.optimizers)):
                fullCost = costs[treeIdx] + penalties[treeIdx]

                opt = regressor.optimizers[treeIdx]

                if settings['optimizer'] == 'Sofomore':
                    opt.tell(
                        rawPopulations[treeIdx], errors[treeIdx], penalties
                    )
                else:
                    opt.tell(rawPopulations[treeIdx], fullCost)

                opt.disp()

            if optStep % 10 == 0:
                treeOutfilePath = os.path.join(
                    settings['outputPath'], 'tree_{}.pkl'.format(optStep)
                )

                bestIdx = np.argmin(costs)
                bestTree = bestIdx//N
                saveTree = deepcopy(regressor.trees[bestTree])
                saveTree.bestParams = saveTree.fillFixedKnots(
                    rawPopulations[bestTree][bestIdx]
                )[0]

                errorsOutfilePath = os.path.join(
                    settings['outputPath'], 'errors_{}.pkl'.format(optStep)
                )

                with open(treeOutfilePath, 'wb') as outfile:
                    pickle.dump(saveTree, outfile)

                with open(errorsOutfilePath, 'wb') as outfile:
                    pickle.dump(errors[bestTree][bestIdx], outfile)


def buildSVNodePool(group):
    """Prepare svNodePool for use in tree construction"""

    svNodePool = []

    # `group` is a pointer to an entry for a structure in the database
    for svName in sorted(group):
        svGroup = group[svName]

        restrictions = None
        if 'restrictions' in svGroup.attrs:
            restrictions = []
            resList = svGroup.attrs['restrictions'].tolist()
            for num in svGroup.attrs['numRestrictions']:
                tmp = []
                for _ in range(num):
                    tmp.append(tuple(resList.pop()))
                restrictions.append(tmp)

        svNodePool.append(
            SVNode(
                description=svName,
                components=svGroup.attrs['components'],
                numParams=svGroup.attrs['numParams'],
                bonds={
                    k:svGroup[k].attrs['components'] for k in svGroup.keys()
                },
                restrictions=restrictions,
                paramRanges=svGroup.attrs['paramRanges']\
                    if 'paramRanges' in group[svName].attrs else None
            )
        )

    return svNodePool


def computeErrors(refStruct, energies, forces, trueValues):
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

        trueValues (dict):
            {structName: {'energy': eng, 'forces':fcs}}. Note that it is
            assumed that the true 'energies' are actually energy differences
            where the energy of the reference structure has already been
            subtracted off.

    Returns:
        costs (list):
            A list of the total costs for the populations of each tree. Eeach
            entry will have a shape of (P, 2*S) where P is the population size
            and S is the number of structures being evaluated.
    """

    keys = list(energies.keys())
    numTrees   = len(energies[keys[0]])
    numPots    = energies[keys[0]][0].shape[0]
    numStructs = len(keys)

    keys = list(energies.keys())
    errors = []

    for treeNum in range(numTrees):
        treeErrors = np.zeros((numPots, 2*numStructs))
        for i, structName in enumerate(sorted(keys)):
            eng = energies[structName][treeNum] - energies[refStruct][treeNum]
            fcs =   forces[structName][treeNum]

            engErrors = eng - trueValues[structName]['energy']
            fcsErrors = fcs - trueValues[structName]['forces']

            treeErrors[:,   2*i] = abs(engErrors)
            treeErrors[:, 2*i+1] = np.average(np.abs(fcsErrors), axis=(1, 2))

        treeErrors[:, ::2]  *= settings['energyWeight']
        treeErrors[:, 1::2] *= settings['forcesWeight']

        # Could not sum here to preserve per-struct energy/force errors
        # costs.append(treeCosts.sum(axis=1))
        errors.append(treeErrors)


    return errors


if __name__ == '__main__':
    # Prepare usual MPI stuff
    worldComm = MPI.COMM_WORLD
    isMaster = (worldComm.Get_rank() == 0)

    # Load settings
    settings = Settings.from_file(args.settings)
    settings['PROCS_PER_PHYS_NODE'] = args.procs_per_node
    settings['PROCS_PER_MANAGER'] = args.procs_per_manager

    random.seed(settings['seed'])
    np.random.seed(settings['seed'])

    if isMaster:
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

    if settings['runType'] == 'GA':
        main(settings, worldComm, isMaster)
    else:
        fixedExample(settings, worldComm, isMaster)
