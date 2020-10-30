import unittest
import numpy as np

from tree import SVTree
from nodes import FunctionNode, SVNode


class Test_SVTree(unittest.TestCase):

    def setUp(self):
        self.svNodePool = [
            SVNode(description='sv{}'.format(i), numParams=10)
            for i in range(5)
        ]

        for i, svNode in enumerate(self.svNodePool):
            svNode.values = i
            
        self.tree = SVTree(
            nodes=[
                FunctionNode('add'),
                self.svNodePool[0],
                FunctionNode('mul'),
                self.svNodePool[1],
                self.svNodePool[2],
            ],
        )

    def test_verify_constructor(self):
        self.assertEqual(5, len(self.tree.nodes))
        self.assertEqual(3, len(self.tree.svNodes))

    
    def test_tree_print(self):
        self.assertEqual(str(self.tree), 'add(sv0, mul(sv1, sv2))')

    
    def test_eval(self):
        self.assertEqual(self.tree.eval(), 2)


    def test_1d_eval(self):
        for node in self.tree.svNodes:
            node.values = np.array([1, 2, 3])

        np.testing.assert_array_equal(self.tree.eval(), [2, 6, 12])


    def test_2d_eval(self):
        for node in self.tree.svNodes:
            node.values = np.tile([1,2,3], reps=(2,))

        np.testing.assert_array_equal(
            self.tree.eval(), np.tile([2, 6, 12], reps=(2,))
        )


    def test_repeat_eval_error(self):
        self.assertEqual(self.tree.eval(), 2)

        with self.assertRaises(SVNode.StaleValueException):
            self.tree.eval()


    def test_random_no_errors(self):
        try:
            SVTree.random(svNodePool=self.svNodePool, maxDepth=3)
        except RuntimeError:
            self.fail("Unexpected RuntimeError in SVTree.random()")


    def test_depth_error(self):
        with self.assertRaises(RuntimeError):
            randTree = SVTree.random(
                svNodePool=self.svNodePool, maxDepth=0
            )


    def test_verify_random_one(self):
        randTree = SVTree.random(
            svNodePool=self.svNodePool, maxDepth=1
        )

        self.assertEqual(1, len(randTree.nodes))
        self.assertEqual(1, len(randTree.svNodes))

    
    def test_populate(self):
        population = self.tree.populate(100)

        totalNumParams = sum([svNode.numParams for svNode in self.tree.svNodes])

        self.assertEqual(population.shape, (100, totalNumParams))


if __name__ == '__main__':
    unittest.main()