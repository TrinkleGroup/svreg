import random
from copy import deepcopy

from nodes import FunctionNode, SVNode, _node_types


class SVTree(list):
    """
    A class for organizing and evaluating a structure vector equation tree that 
    represents the tree using a 1D list. This module draws heavily from the
    gplearn._program module, though it has some critical changes to account for
    the use of structure vectors.

    Tree evaluation is performed by recursively converting the 1D list of nodes
    into a 2D list of lists depending on the node type.

    https://github.com/trevorstephens/gplearn/blob/master/gplearn/_program.py

    Attributes:
        nodes (list):
            The tree itself is represented by a 1D list of nodes.

        popsize (int):
            The number of parameter sets to generate for each SVNode. Used
            during fitting.

        svNodes (list):
            A list that points to the SVNodes in the tree; used for
            easily accessing the parameters of the SV nodes.
    """

    def __init__(self, nodes=None, popSize=10):
        if nodes is None:
            nodes = []
        self.nodes = nodes

        self.popSize = popSize
        self.svNodes = [node for node in nodes if isinstance(node, SVNode)]


    @classmethod
    def random(cls, svNodePool, maxDepth=1, singleNodeTreeProb=0.1):
        """
        Generates a random tree with a maximum depth of maxDepth by randomly
        adding nodes from a pool of function nodes and the given svNodes list.

        Args:
            maxDepth (int):
                The maximum allowed depth.

            svNodePool (list):
                The collection of svNodes that can be used to generate the tree.

            singleNodeTreeProb (float):
                The probability of generating a tree that is a single SVNode.
        """

        if maxDepth < 1:
            raise RuntimeError("maxDepth must be >= 1")

        if (singleNodeTreeProb == 1) and (maxDepth > 1):
            raise RuntimeError(
                "Can't specify singleNodeTreeProb=1 and maxDepth != 1"
            )
        
        tree = cls()

        numSVNodes = len(svNodePool)
        numFuncs = FunctionNode._num_avail_functions
        numNodeChoices = numFuncs + numSVNodes

        # first node shold either be a function or an SVNode
        if random.random() < singleNodeTreeProb:
            # choose random SVNode to use as the only term in the tree
            newSVNode = deepcopy(random.choice(svNodePool))
            tree.nodes.append(newSVNode)
            tree.svNodes.append(newSVNode)
            return tree
        else:
            # choose a random function as the head, then continue with building
            tree.nodes.append(FunctionNode.random())

        # track how many children need to be added to the current node
        # the tree is complete once this list is empty
        nodesToAdd = [tree.nodes[0].function.arity]

        while nodesToAdd:
            depth = len(nodesToAdd)

            choice = random.randint(0, numNodeChoices)

            if (depth < maxDepth) and (choice <= numFuncs):
                # chose to add a FunctionNode
                tree.nodes.append(FunctionNode.random())
                nodesToAdd.append(tree.nodes[-1].function.arity)
            else:
                # add an SVNode
                sv = random.choice(deepcopy(svNodePool))
                tree.nodes.append(sv)
                tree.svNodes.append(sv)

                # see if you need to add any more nodes to the sub-tree
                nodesToAdd[-1] -= 1
                while nodesToAdd[-1] == 0:
                    # pop counters from the stack if sub-trees are complete
                    nodesToAdd.pop()
                    if not nodesToAdd:
                        return tree
                    else:
                        nodesToAdd[-1] -= 1

        # impossible to get here; should always return in while-loop
        raise RuntimeError("Something went wrong in tree construction")


    def eval(self, forces=False):
        """
        Evaluates the tree. Assumes that each SVNode has already been updated to
        contain the SV of the desired structure. If forces=True, computes the
        value of the tree using the derivatives of each node (analytical
        derivatives for FunctionNode objects, force SVs for SVNode objects).

        Args:
            forces (bool):
                If True, computes the derivative of the tree (i.e. the forces).
                Default=False.

        Returns:
            results (np.arr):
                An array of results for self.popSize different
                parameterizations. If forces=False, this will be an array of
                energies of size (self.popSize,). If forces=True, it will
                be an array of forces of size (self.popSize, N, 3) where N is
                the number of atoms in the currently-loaded structure.
        """

        # Check for single-node tree
        if isinstance(self.nodes[0], SVNode):
            return self.nodes[0].values


        # Constructs a list-of-lists where each sub-list is a sub-tree for a
        # function at a given recursion depth. The first node of a sub-tree
        # should always be a FunctionNode

        subTrees = []

        for node in self.nodes:
            if isinstance(node, FunctionNode):
                # start a new sub-tree
                subTrees.append([node])
            else:
                # grow the current deepest sub-tree
                subTrees[-1].append(node)

            # if the sub-tree is complete, evaluate its function
            while len(subTrees[-1]) == subTrees[-1][0].function.arity + 1:
                args = [
                    n.values if isinstance(n, SVNode) # terminal is SVNode
                    else n  # terminal is intermediate result
                    for n in subTrees[-1][1:]
                ]

                intermediateResult = subTrees[-1][0].function(*args)

                if len(subTrees) != 1:  # still some left to evaluate
                    subTrees.pop()
                    subTrees[-1].append(intermediateResult)
                else:  # done evaluating all sub-trees
                    return intermediateResult

        raise RuntimeError("Something went wrong in tree evaluation")


    def populate(self, N):
        """
        Assign random populations to all ParameterNode objects in the tree, then
        return a 2D array of all of these parameters concatenated together.

        Args:
            N (int):
                Number of parameter sets to generate for each ParameterNode

        Return:
            population (np.arr):
                2D array of all ParameterNode parameters
        """

        raise NotImplementedError


    def getPopulation(self):
        """Return a 2D array of all ParameterNode parameters"""

        raise NotImplementedError
    

    def setPopulation(self, population):
        """
        Parse a 2D array of parameters (formatted the same as in
        getPopulation()), then update the parameters corresponding nodes.

        Args:
            population (np.arr):
                The population to be assigned to the ParameterNode objects
        """

        raise NotImplementedError


    def parsePopulation(self, population):
        """
        Convert a 2D array of parameters into a dictionary, where the key is the
        unique node identifier, and the value is an array of parameters.

        Args:
            population (np.arr):
                The population to be parsed

        Returns:
            parameters (dict):
                {Node.id: array of parameters}
        """

        raise NotImplementedError


    def getSVParams(self):
        """
        A helper function for extracting only the parameters that
        correspond to the SVNode objects in the tree. Returns a dictionary where
        the key is the unique node identifier of the SVNode objects in the tree,
        and the value is the populations of each node.

        Returns:
            svParams (dict):
                {Node.id: population of parameters}
        """

        raise NotImplementedError


    def updateSVValues(self, values):
        """
        A helper function for updating the `value` attributes of all SVNode
        objthe tree. 

        Args:
            values (list):
                A list of values to be passed to the SVNode objects the tree.
                Assumed to be ordered the same as self.svNodes.
        """

        raise NotImplementedError


    def __str__(self):
        """Improve print functionality"""

        terminals = [0]
        output = ''

        for i, node in enumerate(self.nodes):
            if isinstance(node, FunctionNode):
                terminals.append(node.function.arity)
                output += node.description + '('
            else:
                output += node.description

                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'

                if i != len(self.nodes) - 1:
                    output += ', '

        return output