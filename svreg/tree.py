import random
import itertools
import numpy as np
from copy import deepcopy

from svreg.nodes import FunctionNode, SVNode
from svreg.summation import Summation, _implemented_sums
from svreg.functions import _function_map


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


    def __eq__(self, other):
        """
        Logic to check for equality of two chemistry trees:

        Inequal if:
            - different number of nodes
            - different frequency count of each node type
            - they don't evaluate to the same thing with dummy values
        """

        n = len(self.nodes)

        # Check incorrect length
        if len(other.nodes) != n:
            return False

        # Check mis-matched nodes
        selfCounts = {}
        otherCounts = {}

        for node in self.nodes:
            name = node.description
            selfCounts[name] = selfCounts.get(name, 0) + 1

        for node in other.nodes:
            name = node.description
            otherCounts[name] = otherCounts.get(name, 0) + 1

        for name in selfCounts:
            if name not in otherCounts:
                return False

            if selfCounts[name] != otherCounts[name]:
                return False

        # Check dummy evaluation
        nsv = len(self.svNodes)

        dummyEnergies = np.random.random((nsv, 1, 1))
        dummyForces   = np.random.random((nsv, 1, 1, 1, 1))

        for i in range(nsv):
            self.svNodes[i].values = (dummyEnergies[i], dummyForces[i])
            other.svNodes[i].values = (dummyEnergies[i], dummyForces[i])

        if self.eval(useDask=False) != other.eval(useDask=False):
            return False
        
        return True


    @classmethod
    def random(cls, svNodePool, maxDepth=1, method='grow', maxNumSVs=1, allSums=False):
        """
        Generates a random tree with a maximum depth of maxDepth by randomly
        adding nodes from a pool of function nodes and the given svNodes list.

        Args:
            svNodePool (list):
                The collection of svNodes that can be used to generate the tree.

            maxDepth (int):
                The maximum allowed depth. The depth of a tree is defined by the
                maximum number of nested functions that it has. A tree with only
                one node has a depth of 0.

            method (str):
                'full' or 'grow'. Following the terminology used in the gplearn
                package, 'grow' means that nodes are randomly selected, often
                leading to asymmetrical trees. 'full' means that only functions
                are chosen until the maximum depth is reached, then the
                "terminal" (SV) nodes are chosen, often resulting in
                symmetrical, "bushy" trees.
        """

        if maxDepth < 0:
            raise RuntimeError("maxDepth must be >= 0")
       
        tree = cls()

        numSVNodes = len(svNodePool)
        numFuncs = FunctionNode._num_avail_functions
        numNodeChoices = numFuncs + numSVNodes

        # Choose a random depth from [0, maxDepth]
        if maxDepth == 0:
            # Choose random SVNode to use as the only term in the tree
            newSVNode = deepcopy(random.choice(svNodePool))
            tree.nodes.append(newSVNode)
            tree.svNodes.append(newSVNode)

            tree.totalNumParams = sum([n.totalNumParams for n in tree.svNodes])

            tree.totalNumFreeParams = sum([
                n.totalNumFreeParams for n in tree.svNodes
            ])


            return tree
        else:
            # Choose a random function as the head, then continue with building
            if allSums:
                tree.nodes.append(FunctionNode('add'))
            else:
                tree.nodes.append(FunctionNode.random())

        # Track how many children need to be added to the current node. The tree
        # is complete once this list is empty
        nodesToAdd = [tree.nodes[0].function.arity]

        numNestedFunctions = int(tree.nodes[0].description != 'add')
        numSVs = 0

        while nodesToAdd:
            # depth = len(nodesToAdd)

            choice = random.randint(0, numNodeChoices)

            if (numNestedFunctions < maxDepth) and (
                (choice <= numFuncs) or (method == 'full')) and (
                    numSVs < maxNumSVs
                ):

                # Chose to add a FunctionNode
                if allSums:
                    tree.nodes.append(FunctionNode('add'))
                else:
                    tree.nodes.append(FunctionNode.random())

                nodesToAdd.append(tree.nodes[-1].function.arity)

                # Only count embedding functions (or *) when considering depth
                numNestedFunctions += int(tree.nodes[-1].description != 'add')
            else:
                # Add an SVNode
                sv = random.choice(deepcopy(svNodePool))
                tree.nodes.append(sv)
                tree.svNodes.append(sv)

                numSVs += 1

                # See if you need to add any more nodes to the sub-tree
                nodesToAdd[-1] -= 1
                while nodesToAdd[-1] == 0:
                    # Pop counters from the stack if sub-trees are complete
                    nodesToAdd.pop()
                    if not nodesToAdd:

                        tree.totalNumParams = sum([
                            n.totalNumParams for n in tree.svNodes
                        ])

                        tree.totalNumFreeParams = sum([
                            n.totalNumFreeParams for n in tree.svNodes
                        ])

                        return tree
                    else:
                        nodesToAdd[-1] -= 1

        # Impossible to get here; should always return in while-loop
        raise RuntimeError("Something went wrong in tree construction")


    @classmethod
    def from_str(cls, treeStr, svNodePool):
        """
        Returns a tree object that matches the corresponding input string.

        Args:
            treeStr (str):

            The string representation of a tree. Should match the format
            used in SVTree.__str__, where chemistry trees are printed
            alphabetically, each chemistry tree is prefixed by "<<element>>",
            and multiple chemistry trees are separated by "|".

            Examples:
                'ffg_AB'
                'softplus(rho_A)'
                'add(softplus(rho_B), rho_A)'
                'add(sv0, add(sv1, sv2))'
        
        Pseudocode:
            Two cases:
                1) The first node is a structure vector
                2) The first node is a function

            Note: mathematical symbols (+, -, *) should never be used here; all
            functions should be specified by name.

            Case 1:
                - If there are no parentheses then this is a single-node tree
                - Check if node exists in node pool; raise error if no
                - Add node to tree and return

            Case 2:
                - If there is a parentheses then the first node must be a
                function
        """

        tree = cls()

        nodePoolNames = [n.description for n in svNodePool]

        # Case 1: single-node tree
        if '(' not in treeStr:
            nodeName = treeStr.strip()
            if nodeName not in nodePoolNames:
                raise RuntimeError(
                    "Node '{}' not found in node pool: {}".format(
                        nodeName, nodePoolNames
                    )
                )
            else:
                tree.nodes.append(
                    deepcopy(svNodePool[nodePoolNames.index(nodeName)])
                )
        else:  # Case 2: multi-node tree, where first node should be a function
            cleanNames = treeStr.split('(')
            cleanNames = [s.replace(')', '') for s in cleanNames]
            cleanNames = [s.split(', ') for s in cleanNames]
            cleanNames = list(itertools.chain.from_iterable(cleanNames))
            cleanNames = [s.strip() for s in cleanNames]

            if cleanNames[0] not in _function_map:
                raise RuntimeError(
                    "First node '{}' is not an allowed function".format(
                        cleanNames[0]
                    )
                )

            for s in cleanNames:
                if s in _function_map:
                    tree.nodes.append(FunctionNode(s))
                elif s in nodePoolNames:
                    tree.nodes.append(
                        deepcopy(svNodePool[nodePoolNames.index(s)])
                    )
                else:
                    raise RuntimeError("Node name '{}' is invalid".format(s))

        tree.updateSVNodes()

        return tree


    def eval(self, useDask=True, allSums=False):
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

        import dask.array as da

        # Check for single-node tree
        if isinstance(self.nodes[0], SVNode):

            # eng = self.nodes[0].values[0].sum(axis=1)
            # fcs = self.nodes[0].values[0]
            # fcs = np.einsum('ijkl->ikl', fcs)

            # eng = client.submit(np.sum, self.nodes[0].values[0], axis=1)
            # fcs = client.submit(np.einsum, 'ijkl->ikl', self.nodes[0].values[0])

            values = self.nodes[0].values

            # eng = dask.delayed(np.sum)(values[0], axis=1)
            eng = np.sum(values[0], axis=1)

            if allSums:
                fcs = values[1]
            else:
                if useDask:
                    # fcs = dask.delayed(np.einsum)('ijkl->ikl', values[1])
                    fcs = da.einsum('ijkl->ikl', values[1])
                else:
                    fcs = np.einsum('ijkl->ikl', values[1])

            return eng, fcs

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

                # intermediateEng = subTrees[-1][0].function(*args)
                # intermediateFcs = subTrees[-1][0].function.derivative(*args)

                # intermediateEng = dask.delayed(subTrees[-1][0].function)(*args)
                # intermediateFcs = dask.delayed(subTrees[-1][0].function.derivative)(*args)
                # intermediateEng = client.submit(subTrees[-1][0].function, *args)
                # intermediateFcs = client.submit(subTrees[-1][0].function.derivative, *args)

                # intermediateEng = dask.delayed(subTrees[-1][0].function)(*args)
                intermediateEng = subTrees[-1][0].function(*args)

                if useDask:
                    # intermediateFcs = dask.delayed(subTrees[-1][0].function.derivative)(*args)
                    intermediateFcs = subTrees[-1][0].function.derivative(*args)
                else:
                    intermediateFcs = subTrees[-1][0].function.derivative(*args)

                if len(subTrees) != 1:  # Still some left to evaluate
                    subTrees.pop()
                    subTrees[-1].append(
                        (intermediateEng, intermediateFcs)
                    )
                else:  # Done evaluating all sub-trees

                    # Ne = num atoms of given element type; N = total num atoms
                    # intermediateEng = (P, Ne)
                    # intermediateFcs = (P, Ne, N, 3)

                    # intermediateEng = intermediateEng.sum(axis=1)
                    # intermediateEng = dask.delayed(np.sum)(intermediateEng, axis=1)
                    # intermediateFcs = dask.delayed(np.einsum)('ijkl->ikl', intermediateFcs)

                    # intermediateEng = client.submit(np.sum, intermediateEng, axis=1)
                    # intermediateFcs = client.submit(np.einsum, 'ijkl->ikl', intermediateFcs)

                    # intermediateEng = dask.delayed(np.sum)(intermediateEng, axis=1)
                    intermediateEng = np.sum(intermediateEng, axis=1)

                    if not allSums:
                        if useDask:
                            # intermediateFcs = dask.delayed(np.einsum)('ijkl->ikl', intermediateFcs)
                            intermediateFcs = da.einsum('ijkl->ikl', intermediateFcs)
                        else:
                            intermediateFcs = np.einsum('ijkl->ikl', intermediateFcs)
                    
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


    # def parseDict2Arr(self, population, N):
    #     """

    #     TODO: I think this function isn't actually needed; I instead have just
    #     been keeping track of the "rawPopulation" that was originally parsed
    #     into a dictionary


    #     Converts a dictionary of {svName: np.vstack-ed array of parameters} to
    #     a 2D array form. Useful for passing to Optimizer objects.

    #     Args:
    #         population (dict):
    #             {svName: np.vstack-ed array of parameters}

    #         N (int):
    #             The number of parameter sets generated for each node.
    #     """

    #     # Split the populations for each SV type
    #     for svName in population:
    #         population[svName] = np.split(
    #             population[svName], population[svName].shape[0]//N
    #         )

    #     # Now build an ordered horizontal array of shape (N, ?)
    #     # where the second dimension depends on the structure of the tree.
    #     array = []
    #     for svNode in self.svNodes:
    #         array.append(population[svNode.description].pop())

    #     # Error checking to see if something went wrong
    #     for svName in population:
    #         leftovers = len(population[svName])
    #         if leftovers > 0:
    #             raise RuntimeError(
    #                 'SV {} had {} extra parameter set(s)'.format(
    #                     svName, leftovers
    #                 )
    #             )

    #     return np.hstack(array)

    
    def fillFixedKnots(self, population):
        """Inserts any specified values into correct columns of `population`."""

        population = np.atleast_2d(population)

        fullShape = (
            population.shape[0],
            population.shape[1]+sum([len(n.restrictions) for n in self.svNodes])
        )

        fullPop = np.empty(fullShape)

        splitPop = np.array_split(
            population, np.cumsum([n.totalNumFreeParams for n in self.svNodes]),
            axis=1
        )

        fullPop = []

        for svNode, nodePopSplit in zip(self.svNodes, splitPop):
            compPopSplit = np.array_split(
                nodePopSplit,
                np.cumsum([svNode.numFreeParams[k] for k in svNode.components]),
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
                If True, inserts fixed knots into rawPopulation. Note that
                fillFixedKnots==True does not change the shape of the population,
                it just updates the fixed knots with their specified values.

        Returns:
            parameters (dict):
                # {svName: {bondType: np.vstack-ed array of parameters}}
                {svName: np.vstack-ed array of parameters}
        """

        if fillFixedKnots:
            splitIndices = np.cumsum([
                n.totalNumFreeParams for n in self.svNodes
            ])
        else:
            splitIndices = np.cumsum([
                n.totalNumParams#+sum([len(r) for r in n.restrictions.values()])
                for n in self.svNodes
            ])

        splitIndices = np.concatenate([[0], splitIndices])
        # splitPop = np.split(rawPopulation, splitIndices, axis=1)

        splitPop = []
        for i in range(splitIndices.shape[0]-1):
            start = splitIndices[i]
            stop  = splitIndices[i+1]

            splitPop.append(rawPopulation[:, start:stop])

        # Convert raw parameters into SV node parameters (using outer products)
        parameters = {}
        for svNode, rawParams in zip(self.svNodes, splitPop):

            # Prepare dictionary if entry doesn't exist yet
            if svNode.description not in parameters:
                parameters[svNode.description] = []

            # Split the parameters for each component type
            if fillFixedKnots:
                # splitParams = np.array_split(
                #     rawParams, np.cumsum(
                #         [svNode.numFreeParams[c] for c in svNode.components]
                #     )[:-1],
                #     axis=1
                # )

                splits = np.cumsum([
                    svNode.numFreeParams[c] for c in svNode.components
                ])
            else:
                # splitParams = np.array_split(
                #     rawParams, np.cumsum([
                #         svNode.numParams[c] for c in svNode.components
                #     ])[:-1],
                #     axis=1
                # )

                splits = np.cumsum([
                    svNode.numParams[c] for c in svNode.components
                ])

            splits = np.concatenate([[0], splits])

            splitParams = []
            for i in range(splits.shape[0]-1):
                start = splits[i]
                stop  = splits[i+1]

                splitParams.append(rawParams[:, start:stop])

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

            cart = None
            for componentName in svNode.constructor:
                tmp = componentParams[componentName]

                if cart is None:
                    cart = tmp
                else:
                    cart = np.einsum('ij,ik->ijk', cart, tmp)
                    cart = cart.reshape(
                        cart.shape[0], cart.shape[1]*cart.shape[2]
                    )

            parameters[svNode.description].append(cart)

        # Stack populations of same bond type; can be split by N later
        for svName in parameters:
            parameters[svName] = np.concatenate(parameters[svName], axis=0)

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
                W[i, i-1] = 1/6.

        A = D.T @ (np.linalg.inv(W) @ D)

        penalties = np.zeros(population.shape[0])

        for splinePop in splitParams:
            m = splinePop[:, :-2].T
            penalties += np.diagonal(m.T @ (A @ m))

        return penalties/len(splitParams)  # normalize by number of splines


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


    def __repr__(self):
        return str(self)


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


    def pointMutate(self, svNodePool, mutProb, allSums=False):
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

                if not allSums:
                    # if allSums, can't mutate function nodes
                    self.nodes[mutIdx] = FunctionNode.random(arity=arity)
            else:
                # Choose a random SV node
                self.nodes[mutIdx] = deepcopy(random.choice(svNodePool))

    
    def updateSVNodes(self):
        """Updates self.svNodes. Useful after crossovers/mutations."""
        self.svNodes = [node for node in self.nodes if isinstance(node, SVNode)]

        self.totalNumFreeParams = sum([
            n.totalNumFreeParams for n in self.svNodes
        ])
        
        self.totalNumParams = sum([
            n.totalNumParams for n in self.svNodes
        ])


    def directEvaluation(self, y, atoms, allElements, evalType, bc_type, cutoffs, hostType=None):
        """
        Evaluates a tree by performing SV summations directly, rather than
        using the SV representation.

        Args:
            y (np.arr):
                The full set of parameters for the tree nodes.

            atoms (ase.Atoms):
                The atomic structure to evaluate

            evalType (str):
                One of 'energy' or 'forces'.

            bc_type (str):
                One of 'natural' or 'fixed'. 'natural' uses natural boundary
                conditions for LHS boundaries of radial functions and for both
                boundaries of non-radial functions.

            cutoffs (list):
                The low/high cutoff ranges for the splines. Note that splines
                perform linear extrapolation outside of this range.

            hostType (str):
                Used to work with multi-component trees, where a Summation
                object might be intended for use with only a given host atom
                type. If hostType is not None, then loop() only iterates over
                atoms with the given hostType.

        Returns:
            If `evalType` == 'energy', returns the total energy of the system.
            If `evalType` == 'forces', returns the atomic forces.
        """

        if evalType not in ['energy', 'forces']:
            raise RuntimeError("evalType must be one of 'energy' or 'forces'.")

        splits = np.cumsum([n.totalNumParams for n in self.svNodes])[:-1]
        splitParams = np.array_split(y, splits)

        # Clone the tree, but replace SVNode objects with Summation objects
        nodes = []
        for node in self.nodes[::-1]:

            # TODO: node.description could be something like ffg_AB

            nodeType = node.description.split('_')[0]

            # if node.description in _implemented_sums:
            if nodeType in _implemented_sums:
                nelem = None
                if nodeType == 'rho':
                    nelem = 1
                if nodeType == 'ffg':
                    if len(node.components) == 2:
                        nelem = 1
                    elif len(node.components) == 3:
                        nelem = 2
                    else:
                        raise NotImplementedError(
                            "Only numElements <=2 supported currently"
                        )

                nodes.append(
                    _implemented_sums[nodeType](
                        name=node.description,
                        allElements=allElements,
                        neighborElements=list(set(itertools.chain.from_iterable(
                            node.inputTypes.values()))),
                        components=node.components,
                        inputTypes=node.inputTypes,
                        numParams=node.numParams,
                        restrictions=node.restrictions,
                        paramRanges=node.paramRanges,
                        bonds=None,
                        bondMapping='lambda x: x',
                        numElements=nelem,
                        cutoffs=cutoffs,
                        bc_type=bc_type,
                    )
                )

                nodes[-1].setParams(splitParams.pop())
            else:
                nodes.append(node)

        nodes = nodes[::-1]

        # Resume: figure out, for each node, how to specify neighTypes

        # Check for single-node tree
        if isinstance(nodes[0], Summation):
            if len(nodes) > 1:
                raise RuntimeError(
                    "First node isn't a funciton, but there is more than one\
                    node."
                )
            res = nodes[0].loop(atoms, evalType, hostType)

            if evalType == 'energy':
                return np.sum(res)
            elif evalType == 'forces':
                return np.einsum('ijkl->kl', res[0])

        # Constructs a list-of-lists where each sub-list is a sub-tree for a
        # function at a given recursion depth. The first node of a sub-tree
        # should always be a FunctionNode

        subTrees = []

        for node in nodes:
            if isinstance(node, FunctionNode):
                # Start a new sub-tree
                subTrees.append([node])
            else:
                # Grow the current deepest sub-tree
                subTrees[-1].append(node)

            # If the sub-tree is complete, evaluate its function
            while len(subTrees[-1]) == subTrees[-1][0].function.arity + 1:

                args = []
                for n in subTrees[-1][1:]:
                    if isinstance(n, Summation):

                        if evalType == 'forces':
                            fcs, eng = n.loop(atoms, evalType, hostType)
                        else:
                            eng = n.loop(atoms, 'energy', hostType)
                            fcs = None

                        args.append((eng, fcs))

                    else:
                        args.append(n)

                print([[el.shape if el is not None else 'None' for el in l] for l in args])
                intermediateEng = subTrees[-1][0].function(*args)

                if evalType == 'forces':
                    intermediateFcs = subTrees[-1][0].function.derivative(*args)
                else:
                    # Functions expect tuple inputs
                    intermediateFcs = None

                if len(subTrees) != 1:  # Still some left to evaluate
                    subTrees.pop()
                    subTrees[-1].append(
                        (intermediateEng, intermediateFcs)
                    )
                else:  # Done evaluating all sub-trees

                    # Ne = num atoms of given element type; N = total num atoms
                    # intermediateEng = (1, Ne)
                    # intermediateFcs = (1, Ne, N, 3)

                    if evalType == 'energy':
                        intermediateEng = np.sum(intermediateEng)
                        return intermediateEng
                    else:
                        intermediateFcs = np.einsum('ijkl->kl', intermediateFcs)
                        return intermediateFcs

        raise RuntimeError("Something went wrong in tree evaluation")


class MultiComponentTree(SVTree):
    """
    An extension of SVTree that accounts for additional complexities associated
    with dealing with multi-component systems.

    "MultiComponentTree" will be shorthanded as "MCTree"

    The main differences between MCT and SVTree:
        1)  The MCT has different trees for each chemistry.
        2)  When generating a random tree using a pool of SVNode objects, the
            pool for MCT is extended to include each of the bond types
            separately (e.g. FFG -> FFG_AA, FFG_AB, and FFG_BB).
        3)  MCT will track the active "components" of each SVNode, that way it
            can generate correct populations that don't include inactive
            components (e.g. an MCT that only uses FFG_AA won't generate
            parameters for the f_B component).
        4)  The MCT needs some additional logic for parsing results and passing
            them to the correct chemistry tree.
        5)  Mate/mutate operations in an MCT can operate between chemistry
            trees.

    Note: when doing any looping over nodes or chemistry trees, it's assumed
    that the looop will go over the trees in alphabetical order of chemical
    name.

    Attributes:
        elements (list):
            A sorted list of element names (e.g. ['Mo', 'Ti']). Since this list
            is sorted, it will be used when looping over any dictionaries in the
            MCT.
    """

    def __init__(self, elements):#, nodes=None):

        self.elements = sorted(elements)
        self.chemistryTrees = {el: None for el in self.elements}
        self.treeNumParams = {el: None for el in self.elements}
        self.cost = np.inf


    @classmethod
    def random(cls, svNodePool, elements, maxDepth=1, allSums=False):
        """
        Overloads SVTree.random() to generate random trees for each chemistry,
        then to update the corresponding class attributes.
        """

        tree = cls(elements)

        tree.chemistryTrees = {
            el: SVTree.random(svNodePool, maxDepth=maxDepth, allSums=allSums)
            for el in tree.elements
        }

        tree.updateSVNodes()

        return tree

    
    @classmethod
    def from_str(cls, treeStr, elements, svNodePool):
        """
        Generates a multi-component tree from a string representation.

        Expected string format:
        - SV node names match names in svNodePool
        - Function names match names from svreg.functions._function_map
        - Functions with 2 inputs have inputs separated by ','
        - Strings for each chemistry tree are prefixed by '<(element)> '
        - Strings for each chemistry tree are separated by ' | '

        Example strings:
            '<Mo> softplus(ffg_BB)'
            '<Mo> softplus(ffg_BB) + <Ti> add(rho_A, rho_B)'
        """

        chemistryTreeStrings = treeStr.split(' | ')

        tree = cls(elements)

        for chemStr in chemistryTreeStrings:
            tag = chemStr.find('>')

            el = chemStr[:tag][1:]

            if el not in elements:
                raise RuntimeError(
                    "Element '{}' not in elements list".format(el)
                )

            tree.chemistryTrees[el] = SVTree.from_str(
                chemStr[tag+1:], svNodePool
            )
        
        tree.updateSVNodes()
        return tree

    
    @classmethod
    def from_file(cls, fileName, elements, svNodePool):
        """
        Reads a parameterized tree from a parameter file. The first line of
        the file should be the string representation of the tree. Every line
        after that should be a 2-tuple, where the first value is 0 or 1
        (0 = knot is fixed and shouldn't be free for fitting), and the second
        value is a knot parameter.

        Note that this function is currently only intended to be used as a
        helper function during fitting, so it's assumed that the structure
        vector nodes (names, neighbors, cutoffs, num_knots) have already been
        defined.
        """

        with open(fileName, 'r') as f:
            tree = MultiComponentTree.from_str(
                f.readline().strip(), elements, svNodePool
            )

            params = np.genfromtxt(f)

        svi = 0
        ci = 0
        ki = 0
        sv = tree.svNodes[svi]
        sv.restrictions[sv.components[ci]] = []
        sv.numFreeParams[sv.components[ci]] = sv.numParams[sv.components[ci]]
        sv.totalNumFreeParams = sum(sv.numFreeParams.values())
        for p in params:

            fixed, val = p
            if fixed:
                if ki == sv.numParams[sv.components[ci]]:
                    # Move to next component if finished loading
                    ci += 1
                    ki = 0

                    # if ki == sv.totalNumParams:
                    if ci == len(sv.components):
                        sv.totalNumFreeParams = sum(sv.numFreeParams.values())
                        # Move to next node if finished loading
                        svi += 1
                        sv = tree.svNodes[svi]
                        ci = 0
                        ki = 0
                        sv.restrictions[sv.components[ci]] = []
                        sv.numFreeParams[sv.components[ci]] = sv.numParams[sv.components[ci]]
                        sv.totalNumFreeParams = sum(sv.numFreeParams.values())
                    else:
                        sv.restrictions[sv.components[ci]] = []
                        sv.numFreeParams[sv.components[ci]] = sv.numParams[sv.components[ci]]
                        sv.totalNumFreeParams = sum(sv.numFreeParams.values())

                if fixed:
                    sv.restrictions[sv.components[ci]].append((ki, val))
                    sv.numFreeParams[sv.components[ci]] -= 1
                    sv.totalNumFreeParams -= 1

            ki += 1

        return tree


    def write_to_lammps(self, fname, y, cutoffs):
        writtenTemplates = []

        with open(fname, 'w') as f:
            # Write SV templates
            f.write("#"*80 + "\n")
            f.write("# STRUCTURE VECTORS\n")

            for svn in self.svNodes:

                if svn.description in writtenTemplates: continue
                writtenTemplates.append(svn.description)

                nodeType = svn.description.split('_')[0]

                if nodeType == 'rho':
                    f.write("Rho\n")
                elif nodeType == 'ffg':
                    f.write("FFG\n")

                f.write("\tname: {}\n".format(svn.description))

                neighborElements = list(set(itertools.chain.from_iterable(
                    svn.inputTypes.values())))
                
                neighborElements = ' '.join(sorted(neighborElements))

                f.write("\tneighbors: {}\n".format(neighborElements))
                f.write("\tcutoffs: {}\n".format(' '.join([str(c) for c in cutoffs])))

                numKnots = ' '.join(
                    [str(svn.numParams[c] - 2) for c in svn.components]
                )

                f.write("\tnum_knots: {}\n".format(numKnots))
                f.write("\n")

            f.write("#"*80 + "\n")
            f.write("# TREE\n")
            f.write("Tree\n")
            f.write("{}\n".format(str(self)))

            f.write("#"*80 + "\n")
            f.write("# PARAMETERS\n")
            f.write("Parameters\n")

            for y_i in y:
                f.write("{}\n".format(y_i))
    

    def eval(self, useDask=True, allSums=False):
        vals = [
            self.chemistryTrees[el].eval(useDask=useDask, allSums=allSums)
            for el in self.elements
        ]

        eng, fcs = zip(*vals)

        # eng = [(P,) for _ in chemistryTrees]
        # fcs = [(P, N, 3) for _ in chemistryTrees]

        return eng, fcs


    def populate(self, N):
        return np.hstack(
            [self.chemistryTrees[el].populate(N) for el in self.elements]
        )


    def fillFixedKnots(self, population):
        splits = np.cumsum([
            self.chemistryTrees[el].totalNumFreeParams
            for el in self.elements
        ])
        splitPop = np.array_split(population, splits, axis=1)

        return np.hstack([
            self.chemistryTrees[el].fillFixedKnots(splitPop[i])
            for i,el in enumerate(self.elements)
        ])

    
    def parseArr2Dict(self, rawPopulation, fillFixedKnots=True):
        splitIndices = np.cumsum([
            self.chemistryTrees[el].totalNumFreeParams if fillFixedKnots
            else self.chemistryTrees[el].totalNumParams
            for el in self.elements
        ])

        splitIndices = np.concatenate([[0], splitIndices])

        # splitPop = np.split(rawPopulation, splitIndices, axis=1)
        splitPop = []
        for i in range(splitIndices.shape[0]-1):
            start = splitIndices[i]
            stop  = splitIndices[i+1]

            splitPop.append(rawPopulation[:, start:stop])

        subDicts = {
            el: self.chemistryTrees[el].parseArr2Dict(
                splitPop[i], fillFixedKnots
            ) for i,el in enumerate(self.elements)
        }

        return subDicts


    def crossover(self, donor):
        """
        For MCTree, crossover does not necessarily need to be within trees of
        the same host element type. Crossover is performed by choosing a random
        from both parents.
        """

        parentTree1 = random.choice(list(self.chemistryTrees.values()))
        parentTree2 = random.choice(list(donor.chemistryTrees.values()))

        parentTree1.crossover(parentTree2)


    def pointMutate(self, svNodePool, mutProb, allSums=False):
        for tree in self.chemistryTrees.values():
            tree.pointMutate(svNodePool, mutProb, allSums=allSums)


    def updateSVNodes(self):

        self.nodes = {
            el: self.chemistryTrees[el].nodes
            for el in self.elements
        }

        for tree in self.chemistryTrees.values():
            tree.updateSVNodes()

        self.svNodes = list(itertools.chain.from_iterable(
            [self.chemistryTrees[el].svNodes for el in self.elements]
        ))

        self.numFreeParams = {
            el: sum([
                n.totalNumFreeParams
                for n in self.chemistryTrees[el].svNodes
            ])
            for el in self.elements
        }

        self.numParams = {
            el: sum([
                n.totalNumParams
                for n in self.chemistryTrees[el].svNodes
            ])
            for el in self.elements
        }

        self.totalNumFreeParams = sum(self.numFreeParams.values())
        self.totalNumParams = sum(self.numParams.values())


    def directEvaluation(self, y, atoms, evalType, bc_type, cutoffs):
        splits = np.cumsum([
            self.chemistryTrees[el].totalNumParams for el in self.elements
        ])

        splitPop = np.array_split(y, splits)

        res = [
            self.chemistryTrees[el].directEvaluation(
                splitPop[ii], atoms, self.elements, evalType, bc_type, cutoffs,
                hostType=el
            )
            for ii, el in enumerate(self.elements)
        ]

        return sum(res)


    def __str__(self):
        return ' | '.join([
            '<{}> {}'.format(
                el, str(self.chemistryTrees[el])
            )
            for el in self.elements
        ])

    
    def __repr__(self):
        return str(self)


    def __eq__(self, other):
        if sorted(self.elements) != sorted(other.elements):
            return False

        return all([
            self.chemistryTrees[el] == other.chemistryTrees[el]
            for el in self.elements
        ])


    def latex(self):
        return {el: self.chemistryTrees[el].latex() for el in self.elements}
        # return ' + '.join([
        #     '{}: {}'.format(
        #         el, self.chemistryTrees[el].latex()
        #     )
        #     for el in self.elements
        # ])
