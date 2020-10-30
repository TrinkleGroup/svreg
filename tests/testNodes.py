import random
import unittest
import numpy as np

from nodes import Node, FunctionNode, SVNode, _function_map
from exceptions import StaleValueException


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
                (random.random(), random.random())
                for _ in range(node.function.arity)
            ]

            self.assertEquals(
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
                'dummy_sv_node', ['comp1', 'comp2'],
                bonds=[['comp1'], ['comp2']],
                numParams=[10, 10],
                restrictions=None,
                paramRanges={'comp1': (1, 2), 'comp2':(3, 4)}
            )
        except:
            self.fail("Something went wrong in SVNode.__init__()")


    def test_verify_constructor(self):
        self.assertEquals(self.node.description, 'dummy_sv_node')
        self.assertEqual(self.node.numParams['comp1'], 10)
        self.assertEqual(self.node.numParams['comp2'], 10)

        with self.assertRaises(StaleValueException):
            self.node.values


    def test_generate_population(self):
        pop = self.node.populate(100)

        self.assertEqual(pop.shape, (100, 20))
        self.assertTrue(np.min(pop) >= 1)
        self.assertTrue(np.max(pop) <= 4)

    def test_value_invalidation(self):
        newValues = np.random.random(size=10)

        self.node.values = newValues

        np.testing.assert_array_equal(self.node.values, newValues)

        with self.assertRaises(StaleValueException):
            self.node.values
   

if __name__ == '__main__':
    unittest.main()