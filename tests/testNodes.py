import random
import unittest
import numpy as np

from svreg.nodes import Node, FunctionNode, SVNode, _function_map
from svreg.exceptions import StaleValueException


class Test_Node(unittest.TestCase):

    def test_basic_constructor(self):
        try:
            node = Node('basic_node')
        except:
            self.fail("Something went wrong in Node.__init__()")

        self.assertEqual(node.description, 'basic_node')


class Test_FunctionNode(unittest.TestCase):

    def test_basic_constructor(self):
        for key in _function_map:
            try:
                node = FunctionNode(key)
            except:
                self.fail(
                    "Something went wrong in FunctionNode.__init__()"\
                    "with key={}".format(key)
                )

            self.assertEqual(node.description, key)
            self.assertEquals(node.function.arity, _function_map[key].arity)

            args = [
                (np.random.random(size=(10, 7)), np.random.random((10, 7, 7, 3)))
                for _ in range(node.function.arity)
            ]

            np.testing.assert_allclose(
                node.function(*args), _function_map[key](*args)
            )


    def test_random_no_errors(self):
        try:
            FunctionNode.random()
        except:
            self.fail("Something went wrong in FunctionNode.random()")


class Test_SVNode(unittest.TestCase):

    def setUp(self):
        try:
            self.node = SVNode(
                description='dummy_sv_node',
                components=['comp1', 'comp2'],
                constructor=['comp1', 'comp2', 'comp1'],
                numParams=[7, 3],
                restrictions=[[(5, 0), (6, 0)], []],
                paramRanges={'comp1': (1, 2), 'comp2':(3, 4)}
            )
        except:
            self.fail("Something went wrong in SVNode.__init__()")


    def test_verify_constructor(self):
        self.assertEquals(self.node.description, 'dummy_sv_node')

        self.assertEqual(self.node.numParams['comp1'], 7)
        self.assertEqual(self.node.numParams['comp2'], 3)

        self.assertEqual(self.node.numFreeParams['comp1'], 5)
        self.assertEqual(self.node.numFreeParams['comp2'], 3)

        with self.assertRaises(StaleValueException):
            self.node.values


    def test_generate_population(self):
        pop = self.node.populate(100)

        self.assertEqual(pop.shape, (100, 8))
        self.assertTrue(np.min(pop) >= 1)
        self.assertTrue(np.max(pop) <= 4)

    def test_value_invalidation(self):
        newValues = np.random.random(size=10)

        self.node.values = newValues

        np.testing.assert_array_equal(self.node.values, newValues)

        with self.assertRaises(StaleValueException):
            self.node.values

        
    def test_fillFixedKnots(self):
        pop = self.node.populate(100)

        filledPop = self.node.fillFixedKnots(pop, 'comp2')
        self.assertEqual(filledPop.shape, (100, 8))
   
        filledPop = self.node.fillFixedKnots(pop, 'comp1')
        self.assertEqual(filledPop.shape, (100, 10))

        np.testing.assert_allclose(filledPop[:, 5], np.zeros(100))
        np.testing.assert_allclose(filledPop[:, 6], np.zeros(100))


if __name__ == '__main__':
    unittest.main()