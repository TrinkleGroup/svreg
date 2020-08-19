"""
Module for Node objects, which are the elements in an equation tree.
"""

class Node:
    """
    The base class for all node objects.

    Attributes:
        id (int):
            a unique integer for identifier for the node

        description (str):
            description of the node (e.g. 'rho_sv', 'scalar', etc.)

        arity (int):
            number of allowed children

        children (list):
            a list of child nodes
    """

    def __init__(self, id, description, arity):
        self.id = id
        self.description = description
        self.arity = arity
        self.children = []


class FunctionNode(Node):
    """
    A node sub-class that is used for function operators.

    Attributes:
        function (callable):
            the function to use for evaluation

        allowedChildrenTypes (list):
            an ordered list of allowed types for child
            nodes, with a length equal to self.arity. Useful for things like
            exponential functions, where the second argument can't be an SVNode
    """

    def __init__(self, id, description, arity, function, allowedChildrenTypes):
        Node.__init__(id, description, arity)
        self.function = function
        self.allowedChildrenTypes = allowedChildrenTypes


class ParameterNode(Node):
    """
    A node sub-class used for storing basic single-value parameters. For
    example: scalar multipliers, powers on exponents, etc.

    Attributes:
        population (np.arr):
            an array of parameter choices

        optimalParamIndex (int):
            indexer into `population` that identifies the best parameter choice

        value (float):
            the value that will be passed back to a parent node
    """

    def __init__(self, id, description, arity):
        Node.__init__(id, description, arity)
        self.optimalParamIndex = None


class SVNode(ParameterNode):
    """
    An extension of a ParameterNode that is specifically designed to represent
    structure vector (SV) operations.

    Attributes:
        svName (str):
            a unique identifier to the name of the structure vector
    """

    def __init__(self, id, description, arity, svName):
        ParameterNode.__init__(id, description, arity)
        self.svName = svName