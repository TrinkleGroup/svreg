class SVTree:
    """
    A class for organizing and evaluating a structure vector equation tree.

    Attributes:
        headNode (Node):
            the top node in the equation tree

        svNodes (list):
            a list that points to the SVNodes in the tree; used for
            easily accessing the parameters of the SV nodes.
    """

    def __init__(self):
        self.headNode = None


    def eval(self):
        pass


    def initializePopulation(self, N):
        """
        Assign random populations to all ParameterNode objects in the tree, then
        return a 2D array of all of these parameters concatenated together.

        Args:
            N (int):
                number of parameter sets to generate for each ParameterNode

        Return:
            population (np.arr):
                2D array of all ParameterNode parameters
        """

        pass


    def getPopulation(self):
        """Return a 2D array of all ParameterNode parameters"""

        pass
    

    def setPopulation(self, population):
        """
        Parse a 2D array of parameters (formatted the same as in
        getPopulation()), then update the parameters corresponding nodes.

        Args:
            population (np.arr):
                the population to be assigned to the ParameterNode objects
        """

        pass


    def parsePopulation(self, population):
        """
        Convert a 2D array of parameters into a dictionary, where the key is the
        unique node identifier, and the value is an array of parameters.

        Args:
            population (np.arr):
                the population to be parsed

        Returns:
            parameters (dict):
                {Node.id: array of parameters}
        """

        pass


    def getSVParams(self):
        """
        A helper function for extracting only the parameters that
        correspond to the SVNode objects in the tree. Returns a dictionary where
        the key is the unique node identifier of the SVNode objects in the tree, and
        the value is the populations of each node.

        Returns:
            svParams (dict):
                {Node.id: population of parameters}
        """

        pass


    def updateSVValues(self, values):
        """
        A helper function for updating the `value` attributes of all SVNode
        objthe tree. 

        Args:
            values (list):
                a list of values to be passed to the SVNode objects the tree.
                Assumed to be ordered the same as self.svNodes.
        """

        pass