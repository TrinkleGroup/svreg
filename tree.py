import random
import numpy as np
from copy import deepcopy
from scipy.interpolate import CubicSpline

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

        svNodes (list):
            A list that points to the SVNodes in the tree; used for
            easily accessing the parameters of the SV nodes.
    """

    def __init__(self, nodes=None):

        if nodes is None:
            nodes = []
        self.nodes = nodes

        self.svNodes = [node for node in nodes if isinstance(node, SVNode)]
        self.cost = np.inf


    @classmethod
    def random(cls, svNodePool, maxDepth=1):
        """
        Generates a random tree with a maximum depth of maxDepth by randomly
        adding nodes from a pool of function nodes and the given svNodes list.

        Args:
            maxDepth (int):
                The maximum allowed depth.

            svNodePool (list):
                The collection of svNodes that can be used to generate the tree.
        """

        if maxDepth < 1:
            raise RuntimeError("maxDepth must be >= 1")
       
        tree = cls()

        numSVNodes = len(svNodePool)
        numFuncs = FunctionNode._num_avail_functions
        numNodeChoices = numFuncs + numSVNodes

        # Choose a random depth from [1, maxDepth]
        if maxDepth == 1:
            # Choose random SVNode to use as the only term in the tree
            newSVNode = deepcopy(random.choice(svNodePool))
            tree.nodes.append(newSVNode)
            tree.svNodes.append(newSVNode)
            return tree
        else:
            # Choose a random function as the head, then continue with building
            tree.nodes.append(FunctionNode.random())

        # Track how many children need to be added to the current node. The tree
        # is complete once this list is empty
        nodesToAdd = [tree.nodes[0].function.arity]

        while nodesToAdd:
            depth = len(nodesToAdd)

            choice = random.randint(0, numNodeChoices)

            if (depth < maxDepth) and (choice <= numFuncs):
                # Chose to add a FunctionNode
                tree.nodes.append(FunctionNode.random())
                nodesToAdd.append(tree.nodes[-1].function.arity)
            else:
                # Add an SVNode
                sv = random.choice(deepcopy(svNodePool))
                tree.nodes.append(sv)
                tree.svNodes.append(sv)

                # See if you need to add any more nodes to the sub-tree
                nodesToAdd[-1] -= 1
                while nodesToAdd[-1] == 0:
                    # Pop counters from the stack if sub-trees are complete
                    nodesToAdd.pop()
                    if not nodesToAdd:
                        return tree
                    else:
                        nodesToAdd[-1] -= 1

        # Impossible to get here; should always return in while-loop
        raise RuntimeError("Something went wrong in tree construction")


    def eval(self):
        """
        Evaluates the tree. Assumes that each SVNode has already been updated to
        contain the SV of the desired structure.

        Returns:
            energies, forces (tuple):
                A tuple of arrays of results for P different parameterizations,
                where P depeneds on the most recent self.populate() call. If
                forces=False, this will be an array of energies of size (P,).
                If forces=True, it will be an array of forces of size (P, N, 3)
                where N is the number of atoms in the current structure.
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
                # Start a new sub-tree
                subTrees.append([node])
            else:
                # Grow the current deepest sub-tree
                subTrees[-1].append(node)

            # If the sub-tree is complete, evaluate its function
            while len(subTrees[-1]) == subTrees[-1][0].function.arity + 1:
                args = [
                    n.values if isinstance(n, SVNode)
                    else n  # Terminal is intermediate result
                    for n in subTrees[-1][1:]
                ]

                intermediateEng = subTrees[-1][0].function(*args)
                intermediateFcs = subTrees[-1][0].function.derivative(*args)

                if len(subTrees) != 1:  # Still some left to evaluate
                    subTrees.pop()
                    subTrees[-1].append(
                        (intermediateEng, intermediateFcs)
                    )
                else:  # Done evaluating all sub-trees
                    return intermediateEng, intermediateFcs

        raise RuntimeError("Something went wrong in tree evaluation")


    def populate(self, N):
        """
        Generate a random population from all SVNode objects in the tree, then
        return an ordered horizontal array of shape (N, ?) where the second
        dimension depends on the structure of the tree.

        Note that an array form is useful for working with Optimizer objects. It
        will be parsed into a dictionary form when passing to an Evaluator.

        Args:
            N (int):
                Number of parameter sets to generate for each SVNode

        Return:
            population (np.arr):
                Array of shape (N, ?), where the length of the second dimension
                depends on the number of SVNode objects in the tree.
        """

        # Get raw per-component parameters for each svNode
        population = []
        for svNode in self.svNodes:
            population.append(svNode.populate(N))

        return np.hstack(population)


    def getPopulation(self):
        """Return a 2D array of all SVNode parameters"""

        raise NotImplementedError
    

    def setPopulation(self, population):
        """
        Parse a 2D array of parameters (formatted the same as in
        getPopulation()), then update the parameters corresponding nodes.

        Args:
            population (np.arr):
                The population to be assigned to the SVNode objects
        """

        raise NotImplementedError


    def parseDict2Arr(self, population, N):
        """
        Converts a dictionary of {svName: np.vstack-ed array of parameters} to
        a 2D array form. Useful for passing to Optimizer objects.

        Args:
            population (dict):
                {svName: np.vstack-ed array of parameters}

            N (int):
                The number of parameter sets generated for each node.
        """

        # Split the populations for each SV type
        for svName in population:
            population[svName] = np.split(population[svName], N)

        # Now build an ordered horizontal array of shape (N, ?)
        # where the second dimension depends on the structure of the tree.
        array = []
        for svNode in self.svNodes:
            array.append(population[svNode.description].pop())

        # Error checking to see if something went wrong
        for svName in population:
            leftovers = len(population[svName])
            if leftovers > 0:
                raise RuntimeError(
                    'SV {} had {} extra parameter set(s)'.format(
                        svName, leftovers
                    )
                )

        return np.hstack(array)

    
    def fillFixedKnots(self, population):
        """Inserts any specified values into correct columns of `population`."""

        population = np.atleast_2d(population)

        fullShape = (
            population.shape[0],
            population.shape[1]+sum([len(n.restrictions) for n in self.svNodes])
        )

        fullPop = np.empty(fullShape)

        splitPop = np.array_split(
            population, np.cumsum([n.totalNumParams for n in self.svNodes]),
            axis=1
        )

        fullPop = []

        for svNode, nodePopSplit in zip(self.svNodes, splitPop):
            compPopSplit = np.array_split(
                nodePopSplit,
                np.cumsum([svNode.numParams[k] for k in svNode.components]),
                axis=1
            )
            for compName, pop in zip(svNode.components, compPopSplit):
                fullPop.append(svNode.fillFixedKnots(pop, compName))

        return np.hstack(fullPop)


    def parseArr2Dict(self, rawPopulation, fillFixedKnots=True):
        """
        Convert a 2D array of parameters into a dictionary, where the key is the
        structure vector type, and the value is a dictionary of array of
        parameters of all bond types for each SVNode object of that type in
        self.svNodes. Useful for storing in an organized manner.

        Populates any fixed knots with specified values.

        Args:
            rawPopulation (np.arr):
                The population to be parsed

            fillFixedKnots (bool):
                If True, inserts fixed knots into rawPopulation.

        Returns:
            parameters (dict):
                {svName: {bondType: np.vstack-ed array of parameters}}
        """

        if fillFixedKnots:
            splitIndices = np.cumsum([
                n.totalNumParams for n in self.svNodes
            ])[:-1]
        else:
            splitIndices = np.cumsum([
                n.totalNumParams+sum([len(r) for r in n.restrictions.values()])
                for n in self.svNodes
            ])[:-1]

        splitPop = np.split(rawPopulation, splitIndices, axis=1)

        # Convert raw parameters into SV node parameters (using outer products)
        parameters = {}
        for svNode, rawParams in zip(self.svNodes, splitPop):

            if svNode.description not in parameters:
                parameters[svNode.description] = {
                    bondType: [] for bondType in svNode.bonds
                }

            # Split the parameters for each component type
            if fillFixedKnots:
                splitParams = np.array_split(
                    rawParams, np.cumsum(
                        [svNode.numParams[c] for c in svNode.components]
                    )[:-1],
                    axis=1
                )
            else:
                splitParams = np.array_split(
                    rawParams, np.cumsum([
                        len(svNode.restrictions[c]) + sum([svNode.numParams[c]])
                        for c in svNode.components
                    ])[:-1],
                    axis=1
                )

            # Organize parameters by component type for easy indexing
            # Fill any fixed values
            componentParams = {}
            for compName, pop in zip(svNode.components, splitParams):
                if fillFixedKnots:
                    componentParams[compName] = svNode.fillFixedKnots(
                            pop, compName
                        )
                else:
                    componentParams[compName] = pop

            for bondType, bondComponents in svNode.bonds.items():
                # Take outer products to form SV params out of bond components

                cart = None
                for componentName in bondComponents:
                    tmp = componentParams[componentName]

                    if cart is None:
                        cart = tmp
                    else:
                        cart = np.einsum('ij,ik->ijk', cart, tmp)
                        cart = cart.reshape(
                            cart.shape[0], cart.shape[1]*cart.shape[2]
                        )

                parameters[svNode.description][bondType].append(cart)

        # Stack populations of same bond type; can be split by N later
        for svName in parameters:
            for bondType in parameters[svName]:
                parameters[svName][bondType] = np.vstack(
                    parameters[svName][bondType]
                )

        return parameters


    def roughnessPenalty(self, population):
        """
        Computes a roughness penalty for the tree by integrating the second
        derivative of the spline over the input domain for each population.

        Args:
            population (np.arr):
                A PxK array of P separater parameter sets. Assumed not to
                already included any fixed knots.
        """
        splits = []

        for svNode in self.svNodes:
            for comp in svNode.components:
                splits.append(
                    svNode.numParams[comp]+len(svNode.restrictions[comp])
                )

        # Split the population by SVNode (i.e. by splines)
        splitParams = np.array_split(
            self.fillFixedKnots(population), np.cumsum(splits)[:-1],
            axis=1
        )

        # Since it's assumed that all splines have the same number of knots,
        # we can prepare for the smoothness calculations here
        n = splitParams[0].shape[1] - 2
        x = np.arange(n)
        dx = 1  # because we're ignoring actual knot spacing
        D = np.zeros((n-2, n))
        for i in range(D.shape[0]):
            D[i,i] = 1
            D[i,i+1] = -2
            D[i,i+2] = 1
        
        W = np.zeros((n-2, n-2))
        for i in range(W.shape[0]):
            W[i,i] = 2/3.
            if i > 0:
                W[i-1, i]=1/6.
            if i < W.shape[0]-1:
                W[i, i+1] = 1/6.

        A = D.T @ (np.linalg.inv(W) @ D)

        penalties = np.zeros(population.shape[0])

        for splinePop in splitParams:
            m = splinePop[:, :-2].T
            penalties += (m.T @ (A @ m)).sum(axis=0)

        return penalties


    def latex(self):
        """Return LaTeX string"""

        if len(self.nodes) == 1:
            return self.nodes[0].description

        subTrees = []
        output = []

        for node in self.nodes:
            if isinstance(node, FunctionNode):
                # Start a new sub-tree
                subTrees.append([node])
                output.append([node.latex])
            else:
                # Grow the current deepest sub-tree
                subTrees[-1].append(node)
                output[-1].append(node.description)

            # If the sub-tree is complete, evaluate its function
            while len(output[-1]) == subTrees[-1][0].function.arity + 1:
                args = output[-1][1:]

                intermediate = subTrees[-1][0].latex.format(*args)

                if len(subTrees) != 1:  # Still some left to evaluate
                    subTrees.pop()
                    output.pop()
                    output[-1].append(intermediate)
                else:  # Done evaluating all sub-trees
                    return intermediate


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


    def getSubtree(self):
        """
        Get a random sub-tree. As in gplearn, uses Koza's (1992) approach.

        Returns:
            start, end (tuple):
                Indices of start/end of sub-tree.
        """

        probs = np.array([
            0.9 if isinstance(node, FunctionNode) else 0.1
            for node in self.nodes
        ])

        probs = np.cumsum(probs/probs.sum())
        start = np.searchsorted(probs, random.random())

        stack = 1
        end = start
        while stack > end - start:
            node = self.nodes[end]
            if isinstance(node, FunctionNode):
                stack += node.function.arity
            end += 1
        
        return start, end


    def crossover(self, donor):
        """
        Performs an in-place crossover operation between self and `donor`.

        TODO: this can also result in trees with depths > maxDepth

        Args:
            donor (SVTree):
                The tree to cross self with.

        Returns:
            The list of nodes for the new tree.
        """

        # Choose sub-tree for removal
        start, end = self.getSubtree()

        # Choose sub-tree from donor to donate
        donorStart, donorEnd = donor.getSubtree()

        self.nodes = self.nodes[:start]\
            + donor.nodes[donorStart:donorEnd]\
                + self.nodes[end:]

    
    def mutate(self, svNodePool, maxDepth=1):
        """Does an in-place mutation of the current tree."""

        raise NotImplementedError

        # TODO: will likely grow deeper than max depth...

        randomDonor = self.random(svNodePool, maxDepth)


    def hoistMutate(self):
        """Implemented in gplearn. Supposedly helps to avoid bloat."""
        raise NotImplementedError


    def pointMutate(self, svNodePool, mutProb):
        """
        Perform an in-place mutation of the current set of nodes.

        Args:
            svNodePool (list):
                The collection of SVNode pools used to generate trees.

            mutProb (float):
                The probability of mutating each node.
        """
        
        # Randomly choose nodes to mutate
        mutantIndices = np.where(
            np.random.random(size=len(self.nodes)) < mutProb
        )[0]

        for mutIdx in mutantIndices:
            node = self.nodes[mutIdx]
            if isinstance(node, FunctionNode):
                # Choose a function with the same arity
                arity = node.function.arity
                self.nodes[mutIdx] = FunctionNode.random(arity=arity)
            else:
                # Choose a random SV node
                self.nodes[mutIdx] = deepcopy(random.choice(svNodePool))

    
    def updateSVNodes(self):
        """Updates self.svNodes. Useful after crossovers/mutations."""
        self.svNodes = [node for node in self.nodes if isinstance(node, SVNode)]
