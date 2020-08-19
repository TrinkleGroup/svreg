class SVRegressor:
    """
    A class for running a genetic algorithm to perform symbolic
    regression using structure vectors.

    Attributes:

        numberOfTrees (int):
            the number of equation trees to construct (i.e. the GA population
            size).

        treePopSize (int):
            the number of parameter sets to generate (per tree) when optimizing
            the trees.

        maxSteps (int):
            the maximum number of allowed GA steps.

        optimizerSteps (int):
            the maximum number of steps the optimizers are allowed to take when
            optimizing the trees.

        structureNames (list):
            a list of unique strings defining the names of the atomic structures
            that are being fitted to.

        optimizer (object):
            the constructor for an optimizer object that will be used to
            optimize tree parameters.
    """

    pass