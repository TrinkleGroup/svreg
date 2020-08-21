from tree import SVTree

class SVRegressor:
    """
    A class for running a genetic algorithm to perform symbolic
    regression using structure vectors.

    Attributes:

        numberOfTrees (int):
            The number of equation trees to construct (i.e. the GA population
            size).

        treePopSize (int):
            The number of parameter sets to generate (per tree) when optimizing
            the trees.

        maxTreeDepth (int):
            Maximum allowed tree depth (a tree with one node has a depth of 0)

        maxSteps (int):
            The maximum number of allowed GA steps.

        optimizerSteps (int):
            The maximum number of steps the optimizers are allowed to take when
            optimizing the trees.gg

        structureNames (list):
            A list of unique strings defining the names of the atomic structures
            that are being fitted to.

        optimizer (object):
            The constructor for an optimizer object that will be used to
            optimize tree parameters.
    """

    def __init__(
        self,
        numberOfTrees=100,
        treePopSize=100,
        maxTreeDepth=,
        maxSteps=1000,
        optimizerSteps=10,
        structureNames=[],
        optimizer=None
        )

        self.numberOfTrees = numberOfTrees
        self.treePopSize = treePopSize

        if maxTreeDepth < 1:
            raise AttributeError("maxTreeDepth must be >= 1")
        self.maxTreeDepth = maxTreeDepth

        self.maxSteps = maxSteps
        self.optimizerSteps = optimizerSteps
        self.structureNames = structureNames
        self.optimizer = optimizer

        self.trees = []

    def initializeTrees(self):
        for _ in range(self.numberOfTrees):
            tree = SVTree()
            tree.headNode = FunctionNode.random()

            # generate up to self.maxTreeDepth layers
            for _ in range(random.randint(1, self.maxTreeDepth)):
                numChildren = random.randint()