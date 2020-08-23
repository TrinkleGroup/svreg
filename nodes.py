"""
Module for Node objects, which are the elements in an equation tree.
"""

import random
import numpy as np
from gplearn.functions import _function_map as _gp_function_map


# choose the allowed functions
_function_map = {
    key: _gp_function_map[key] for key in [
        'add', 'mul', 'sqrt', 'log', 'inv', 'sin', 'cos', 'tan'
    ]
}

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
    A node sub-class that is used for function operators. Leverages the
    gplearn.functions module to handle function 'closure' (handling bad inputs)
    and function arity (number of inputs).

    gplearn.functions documentation:
    https://gplearn.readthedocs.io/en/stable/_modules/gplearn/functions.html

    Attributes:
        function (gplearn.functions._Function):
            The function to use for evaluation; handles closure and arity
    """

    _num_avail_functions = len(_function_map)

    def __init__(self, key):
        Node.__init__(self, description=key)
        self.function = _function_map[key]


    @classmethod
    def random(cls):
        key = random.sample(_function_map.keys(), 1)[0]
        return cls(key=key)


class SVNode(Node):
    """
    A Node that is specifically designed to represent structure vector (SV)
    operations. Doesn't have anything new yet.

    Attributes:
        numParams (int):
            The number of fitting parameters for the corresponding SV.

        population (np.arr):
            An array of shape (P, self.numParams) where each row
            corresponds to a different parameter set. P depends on the most
            recent populate() call.

        TODO: there's no reason for a node to store its population

        paramRange (tuple):
            A length-2 tuple of the (low, high) range of allowed parameters.
            Default is (0, 1).

        values (np.arr):
            The values of the evaluated SVs (i.e. the energies/forces).
            Initialized to None, and reset to None every time they are returned
            to avoid using stale values.
    """

    class StaleValueException(Exception):
        """Used to avoid pullling SVNode values that have already been used"""

        def __init__(self):
            Exception.__init__(self, "Trying to use stale SV value.")

    def __init__(self, description, numParams, paramRange=None):
        Node.__init__(self, description)
        self.numParams = numParams

        self.paramRange = paramRange
        if self.paramRange is None:
            self.paramRange = (0, 1)

        self._values = None

    
    def populate(self, popSize):
        """Generate a random population of `popSize` parameter sets"""
        pop = np.random.random(size=(popSize, self.numParams))

        # shift into expected range
        pop *= (self.paramRange[1] - self.paramRange[0])
        pop += self.paramRange[0]

        return pop

    @property
    def values(self):
        """
        Return current values and reset to None to avoid using stale values in
        the future.
        """

        prev = self._values

        if prev is None:
            raise self.StaleValueException()

        self._values = None
        return prev

    @values.setter
    def values(self, values):
        self._values = values