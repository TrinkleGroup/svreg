import random
import unittest
import numpy as np

from nodes import Node, FunctionNode, SVNode, _function_map


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

            args = [random.random() for _ in range(node.function.arity)]
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
                'dummy_sv_node', numParams=10, paramRange=(1, 2)
            )
        except:
            self.fail("Something went wrong in SVNode.__init__()")


    def test_verify_constructor(self):
        self.assertEquals(self.node.description, 'dummy_sv_node')
        self.assertEqual(self.node.numParams, 10)

        with self.assertRaises(SVNode.StaleValueException):
            self.node.values


    def test_generate_population(self):
        pop = self.node.populate(100)

        self.assertEqual(pop.shape, (100, 10))
        self.assertTrue(np.min(pop) >= 1)
        self.assertTrue(np.max(pop) <= 2)

    def test_value_invalidation(self):
        newValues = np.random.random(size=10)

        self.node.values = newValues

        np.testing.assert_array_equal(self.node.values, newValues)

        with self.assertRaises(SVNode.StaleValueException):
            self.node.values
   

if __name__ == '__main__':
    unittest.main()