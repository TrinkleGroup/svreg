import cma#, comocma
import time
from cma.evolution_strategy import CMAEvolutionStrategy
import numpy as np
from deap import base, creator, tools


"""
All optimizer obects will be instantiated by calling
optimizer(tree.populate(N=1)[0], **optimizerArgs) for each tree in the
regressor. They must also implement an ask()/tell() interface.
"""


class Timer:
    def __init__(self):
        self._prev_printed = None

    
    @property
    def elapsed(self):
        if self._prev_printed is None:
            self._prev_printed = time.time()

        return time.time() - self._prev_printed


class DummyFitClass:
    def __init__(self):
        self.median0 = None
        self.fit = None


class GAWrapper(CMAEvolutionStrategy):
    PRESERVE_FRAC = 5
    NUM_ALLOWED_STALE_STEPS = 100

    def __init__(self, parameters, *args):
        self.toolbox, self.creator = buildToolbox(len(parameters))

        self.opts = args[0]

        # the DEAP toolbox requires a population to be a collection of Individuals
        population = self.toolbox.population(self.opts['popsize'])
        self.population = []
        for ind in population:
            self.population.append(self.creator.Individual(ind))

        self.offspringStartIndex = len(self.population) // self.PRESERVE_FRAC

        # Stuff needed for disp()
        self.countiter = 0
        self.countevals = 0
        self.timer = Timer()
        self.fit = 0


    def disp(self):
        print(
            self.countiter, self.countevals, self.fit, self.timer.elapsed,
            flush=True
        )


    def ask(self, N):
        """
        Update (in place) the GA population using mate/mutate operations.
        """

        for pot_num in range(self.offspringStartIndex, N):
            momIdx = np.random.randint(1, self.offspringStartIndex)

            dadIdx = momIdx
            while dadIdx == momIdx:
                dadIdx = np.random.randint(1, self.offspringStartIndex)

            mom = self.population[momIdx]
            dad = self.population[dadIdx]

            kid, _ = self.toolbox.mate(
                self.toolbox.clone(mom), self.toolbox.clone(dad)
            )

            self.population[pot_num] = kid

        for mutIndiv in self.population[self.offspringStartIndex:]:
            if np.random.random() >= self.opts['pointMutateProb']:
                self.toolbox.mutate(mutIndiv)

        return self.population[:N]


    def tell(self, population, fitnesses):
        """
        Update the fitness values of the individuals in the population. Note
        I am using the DEAP toolbox, so the fitness values need to be tuples.

        The 'population' argument is unused; it's a filler because moes needs it
        """

        for idx in range(len(population)):
            ind = self.population[idx]
            fit = fitnesses[idx]

            ind.fitness.values = fit,

        self.population = tools.selBest(self.population, len(self.population))

        newAvgFitness = np.average(fitnesses, axis=0)

        self.countiter += 1
        self.countevals += len(population)
        self.fit = min(fitnesses)


    def stop(self):
        return {}

    @property
    def incumbent(self):
        return self.population[0]

    def __getstate__(self):
        """Avoid pickling errors when using Pool in MOES"""

        self_dict = self.__dict__.copy()
        del self_dict['toolbox']
        del self_dict['creator']

        return self_dict


def buildToolbox(numParams):
    """ Initializes GA toolbox with desired functions """

    creator.create("CostFunctionMinimizer", base.Fitness, weights=(-1.,))
    creator.create("Individual", np.ndarray, fitness=creator.CostFunctionMinimizer)

    def ret_pvec(arr_fxn):
        return arr_fxn(np.random.random(numParams))

    toolbox = base.Toolbox()
    toolbox.register('parameter_set', ret_pvec, creator.Individual)
    toolbox.register('population', tools.initRepeat, list, toolbox.parameter_set)

    toolbox.register(
        'mutate', tools.mutGaussian,
        mu=0, sigma=1, indpb=0.1
    )

    toolbox.register('mate', tools.cxTwoPoint)

    return toolbox, creator


# class SofomoreWrapper(comocma.Sofomore):
# 
#     def __init__(self, parameters, settings):
# 
#         # `parameters` is unused; just a dummy to satisfy Optimizer API
#         # TODO: need a good way to 
#         allParameters = np.random.normal(
#             loc=parameters,
#             size=(settings['SofomorePopSize'], parameters.shape[0])
#         )
# 
#         self.groupIndices = self.buildErrorIndices(settings['structNames'])
# 
#         # Prepare Sofomore MOES optimizer
#         reference_point = [1]*2*len(self.groupIndices)
#         ideal_hypervolume = np.prod(settings['paretoDimensionality'])
# 
#         solvers = [
#             comocma.como.CmaKernel(
#                 sol,
#                 1.0,
#                 {
#                     'popsize': settings['CMApopSize'],
#                     'verb_disp': 0,
#                 }
#             )
#             for sol in allParameters
#         ]
# 
#         comocma.Sofomore.__init__(
#             self,
#             solvers,
#             reference_point=reference_point,
#             opts=settings['opts'],
#             threads_per_node=settings['threads_per_node']
#         )
# 
#     
#     def buildErrorIndices(self, structNames):
#         """
#         It is expected that error vectors will be of length numStructs. In
#         order to convert that into the proper dimensionality, groups of indices
#         must be defined based on the names of the structures.
#         """
# 
#         groupConditions = [
#                 # lambda n: 1 if 'strain' in n else 0,
#                 lambda n: 1 if 'surface' in n else 0,
#                 # lambda n: 1 if 'Vacancy' in n else 0,
#                 # lambda n: 1 if ('6000_K' in n) and ('Vacancy' not in n) else 0,
#         ]
# 
#         groupIndices = []
# 
#         for cond in groupConditions:
#             indices = np.array([
#                 # 1 if group_name in n else 0 for n in all_struct_names
#                 cond(n) for n in structNames
#             ])
# 
#             indices = np.where(indices)[0].tolist()
# 
#             groupIndices.append(indices)
# 
# 
#         everythingElse = np.arange(len(structNames)).tolist()
# 
#         delIndices = []
#         for group in groupIndices:
#             delIndices += group
# 
#         delIndices = sorted(delIndices)
# 
#         for ind in delIndices[::-1]:
#             del everythingElse[ind]
# 
#         groupIndices.append(everythingElse)
# 
#         return groupIndices
# 
# 
# 
#     
#     def tell(self, population, errors, penalties=None):
#         """
#         Convert full database errors into correct dimensionality, then update
#         kernels.
# 
#         Args:
#             population (np.arr):
#                 The list of parameter sets corresponding to each error vector.
# 
#             errors (np.arr):
#                 The full (P, 2*M) array of errors for each P parameter set and M
#                 structures.
# 
#             penalties (np.arr):
#                 Optional array of P penalties.
#         """
# 
#         energyCosts  = np.atleast_2d(errors[:, ::2])
#         forcesCosts  = np.atleast_2d(errors[:, 1::2])
# 
#         energyCostGroups = []
#         forcesCostGroups = []
# 
#         for group in self.groupIndices:
#             groupEngCosts = energyCosts[:, group]
#             groupFcsCosts = forcesCosts[:, group]
# 
#             energyCostGroups.append(np.average(groupEngCosts, axis=1))
#             forcesCostGroups.append(np.average(groupFcsCosts, axis=1))
# 
#         energyCostGroups = [
#             np.atleast_2d(grpCost) for grpCost in energyCostGroups
#         ]
# 
#         forcesCostGroups = [
#             np.atleast_2d(grpCost) for grpCost in forcesCostGroups
#         ]
# 
#         energyCostGroups = np.vstack(energyCostGroups)
#         forcesCostGroups = np.vstack(forcesCostGroups)
# 
#         newCosts = np.vstack([
#            energyCostGroups, forcesCostGroups
#         ]).T
# 
#         comocma.Sofomore.tell(
#             self, population, newCosts, penalties=penalties[0]
#         )
# 
#     
#     def ask(self, N):
#         return comocma.Sofomore.ask(self, 'all')
