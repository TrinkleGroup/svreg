import random
import numpy as np
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

        # choose a random depth from [1, maxDepth]
        if maxDepth == 1:
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
                # start a new sub-tree
                subTrees.append([node])
            else:
                # grow the current deepest sub-tree
                subTrees[-1].append(node)

            # if the sub-tree is complete, evaluate its function
            while len(subTrees[-1]) == subTrees[-1][0].function.arity + 1:
                args = [
                    n.values if isinstance(n, SVNode)
                    else n  # terminal is intermediate result
                    for n in subTrees[-1][1:]
                ]

                intermediateEng = subTrees[-1][0].function(*args)
                intermediateFcs = subTrees[-1][0].function.derivative(*args)

                if len(subTrees) != 1:  # still some left to evaluate
                    subTrees.pop()
                    subTrees[-1].append(
                        (intermediateEng, intermediateFcs)
                    )
                else:  # done evaluating all sub-trees
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
                {svName: population}
        """

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


    def parseArr2Dict(self, population):
        """
        Convert a 2D array of parameters into a dictionary, where the key is the
        structure vector type, and the value is an array of parameters of all
        SVNode objects of that type in self.svNodes. Useful for storing in an
        organized manner.

        Args:
            population (np.arr):
                The population to be parsed

        Returns:
            parameters (dict):
                {svName: np.vstack-ed array of parameters}
        """

        # Split by node
        splitIndices = np.cumsum([n.numParams for n in self.svNodes])[:-1]
        splitPop = np.split(population, splitIndices, axis=1)

        # Group by SV type
        parameters = {
            svName: [] for svName in set([n.description for n in self.svNodes])
        }

        for svNode, pop in zip(self.svNodes, splitPop):
            parameters[svNode.description].append(pop)

        for svName in parameters:
            parameters[svName] = np.vstack(parameters[svName])

        return parameters


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

        self.nodes = self.crossover(randomDonor)


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