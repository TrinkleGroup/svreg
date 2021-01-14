"""
Module for Node objects, which are the elements in an equation tree.
"""

import random
import numpy as np

from functions import _function_map, _arities, _latex
from exceptions import StaleValueException

# collection of types of nodes that can be added; used for growing trees
_node_types = ['function', 'parameter', 'sv']


class Node:
    """
    The base class for all node objects.

    Attributes:
        description (str):
            Description of the node (e.g. 'rho_sv', 'scalar', etc.)
    """

    def __init__(self, description):
        self.description = description


class FunctionNode(Node):
    """
    A node sub-class that is used for function operators.

    Attributes:
        function (_Function):
            The function to use for evaluation; handles closure and arity
    """

    _num_avail_functions = len(_function_map)

    def __init__(self, key):
        Node.__init__(self, description=key)
        self.function = _function_map[key]
        self.latex = _latex[key]


    @classmethod
    def random(cls, arity=None):
        if arity is None:
            key = random.sample(_function_map.keys(), 1)[0]
        else:
            key = random.sample(_arities[arity], 1)[0]

        return cls(key=key)

class SVNode(Node):
    """
    A Node that is specifically designed to represent structure vector (SV)
    operations. Doesn't have anything new yet.

    Attributes:
        numParams (int):
            The number of fitting parameters for the corresponding SV.

        components (list):
            A list of names of components that make up the SV.

        numParams (list):
            A list integers corresponding to the number of parameters for each
            component. It's assumed that this does NOT take into account any
            non-free knots specified by `restrictions`.

        constructor (list):
            A list of components for building a parameter vector out of the node
            components. For example, if components = ['f_A', 'g_AA'] and
            constructor = ['f_A', 'f_A', 'g_AA'], then the node would build the
            full parameter vector by generating vectors for 'f_A' and 'g_AA',
            then computing np.outer(np.outer('f_A', 'f_A'), 'g_AA').

        population (np.arr):
            An array of shape (P, self.numParams) where each row
            corresponds to a different parameter set. P depends on the most
            recent populate() call.

        restrictions (list):
            A list of lists of tuples for each component specifying any non-free
            knots, and their values.

            e.g. [[(knotIdx, value), ...] for each component]

        paramRanges (dict):
            A dictionary of length-2 tuples of the (low, high) range of allowed
            parameters. If paramRanges is None, each component is automatically
            given a default range of (0, 1).

        values (np.arr):
            The values of the evaluated SVs (i.e. the energies/forces).
            Initialized to None, and reset to None every time they are returned
            to avoid using stale values.
    """

    def __init__(
        self, description, components, constructor, numParams,
        restrictions=None, paramRanges=None, inputTypes=None
        ):

        Node.__init__(self, description)
        self.components = components
        self.constructor = constructor

        # Used for tracking allowed neighbor types during direct evaluations
        self.inputTypes = inputTypes

        # Load any restricted knot values
        tmp = restrictions
        if tmp is None:
            tmp = [[]]*len(self.components)

        self.restrictions = {comp: res for comp, res in zip(components, tmp)}

        # Store number of free parameters
        self.numFreeParams = {}
        self.numParams = {}
        for compName, num in zip(components, numParams):
            numFixed = len(self.restrictions[compName])
            self.numFreeParams[compName] = num - numFixed
            self.numParams[compName] = num

        self.totalNumParams = sum(self.numParams.values())
        self.totalNumFreeParams = sum(self.numFreeParams.values())

        # Load any limits on the parameter ranges for each component type
        self.paramRanges = paramRanges
        if self.paramRanges is None:
            paramRanges = [(0, 1)]*len(components)
            self.paramRanges = {
                comp: rng for comp, rng in zip(components, paramRanges)
            }

        self._values = None

        self.convolutions = 0

    
    def populate(self, popSize):
        """
        Generate a random population of `popSize` fitting parameters.

        Args:
            popSize (int):
                The number of parameter sets to generate

        Return:
            population (np.arr):
                (P, np.mul(self.numParams)). Parameters are ordered according
                to the order of self.components.
        """

        population = []
        for componentName in self.components:
            numParams = self.numFreeParams[componentName]

            pop = np.random.random(size=(popSize, numParams))

            # shift into expected range
            paramRange = self.paramRanges[componentName]
            pop *= (paramRange[1] - paramRange[0])
            pop += paramRange[0]

            population.append(pop)

        return np.hstack(population)

    
    def fillFixedKnots(self, pop, compName):
        if len(self.restrictions[compName]) < 1:
            return pop

        restrictedKnots, specifiedValues = zip(
            *self.restrictions[compName]
        )

        # Shape including number of fixed knots
        fullShape = (pop.shape[0], pop.shape[1]+len(restrictedKnots))
        knotMask = np.array([
            True if k in restrictedKnots else False
            for k in range(fullShape[1])
        ])

        fullPop = np.empty(fullShape)
        fullPop[:, ~knotMask] = pop
        fullPop[:,  knotMask] = specifiedValues

        return fullPop


    @property
    def values(self):
        """
        Return current values and reset to None to avoid using stale values in
        the future. Expected to be a 2-tuple of (value, derivative).
        """

        prev = self._values

        if prev is None:
            raise StaleValueException()

        self._values = None
        return prev

    @values.setter
    def values(self, values):
        self._values = values