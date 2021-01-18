import unittest
import numpy as np
from copy import deepcopy

from ase import Atoms

from svreg.tree import SVTree
from svreg.nodes import FunctionNode, SVNode
from svreg.exceptions import StaleValueException


class Test_SVTree(unittest.TestCase):

    def setUp(self):
        self.svNodePool = [
            SVNode(
                description='sv{}'.format(i),
                components=['comp1', 'comp2'],
                constructor=['comp1', 'comp2', 'comp1'],
                numParams=[7, 3],
                restrictions=[[(5, 0), (6, 0)], []],
                paramRanges={'comp1': (1, 2), 'comp2':(3, 4)}
            )
            for i in range(5)
        ]

        for i, svNode in enumerate(self.svNodePool):
            svNode.values = (i, 0)
            
        self.tree = SVTree(
            nodes=[
                FunctionNode('add'),
                self.svNodePool[0],
                FunctionNode('add'),
                self.svNodePool[1],
                self.svNodePool[2],
            ],
        )

    def test_verify_constructor(self):
        self.assertEqual(5, len(self.tree.nodes))
        self.assertEqual(3, len(self.tree.svNodes))

    
    def test_tree_print(self):
        self.assertEqual(str(self.tree), 'add(sv0, add(sv1, sv2))')

    
    def test_eval(self):
        self.assertEqual(self.tree.eval(), (3, 0))


    def test_1d_eval(self):
        for node in self.tree.svNodes:
            node.values = (np.array([1, 2, 3]), 0)

        results = self.tree.eval()

        np.testing.assert_array_equal(results[0], [3, 6, 9])
        np.testing.assert_array_equal(results[1], 0)


    def test_2d_eval(self):
        for node in self.tree.svNodes:
            node.values = (np.tile([1,2,3], reps=(2,)), 0)

        results = self.tree.eval()

        np.testing.assert_array_equal(
            results[0], np.tile([3, 6, 9], reps=(2,))
        )

        np.testing.assert_array_equal(results[1], 0)


    def test_repeat_eval_error(self):
        self.assertEqual(self.tree.eval(), (3, 0))

        with self.assertRaises(StaleValueException):
            self.tree.eval()


    def test_random_no_errors(self):
        try:
            SVTree.random(svNodePool=self.svNodePool, maxDepth=3)
        except RuntimeError:
            self.fail("Unexpected RuntimeError in SVTree.random()")


    def test_depth_error(self):
        with self.assertRaises(RuntimeError):
            SVTree.random(
                svNodePool=self.svNodePool, maxDepth=-1
            )


    def test_verify_random_one(self):
        randTree = SVTree.random(
            svNodePool=self.svNodePool, maxDepth=0
        )

        self.assertEqual(1, len(randTree.nodes))
        self.assertEqual(1, len(randTree.svNodes))

    
    def test_populate(self):
        population = self.tree.populate(100)

        self.assertEqual(population.shape, (100, 24))


    def test_fillFixedKnots(self):
        population = self.tree.populate(100)
        self.assertEqual(population.shape, (100, 24))

        population = self.tree.fillFixedKnots(population)
        self.assertEqual(population.shape, (100, 30))



    def test_parseDict2ArrNoFill(self):
        """
        This is for the case where the input population should already have the
        filled in knots.
        """

        population  = self.tree.populate(100)
        population  = self.tree.fillFixedKnots(population)
        popDict     = self.tree.parseArr2Dict(population, fillFixedKnots=False)

        for node in self.tree.svNodes:
            self.assertEqual(
                popDict[node.description].shape,
                (100, np.prod([
                    node.numParams[c] for c in node.constructor
                ]))
            )


    def test_parseDict2ArrYesFill(self):
        population  = self.tree.populate(100)
        popDict     = self.tree.parseArr2Dict(population, fillFixedKnots=True)

        for node in self.tree.svNodes:
            self.assertEqual(
                popDict[node.description].shape,
                (100, np.prod([node.numParams[c] for c in node.constructor]))
            )

            for comp in node.components:
                for rest in node.restrictions[comp]:
                    k, v = rest

                    np.testing.assert_allclose(
                        popDict[node.description][:, k],
                        np.ones(100)*v

                    )


    def test_parseDict2Arr(self):
        population  = self.tree.populate(100)
        popDict     = self.tree.parseArr2Dict(population, fillFixedKnots=True)

        population2 = self.tree.parseDict2Arr(popDict, 100)

        np.testing.assert_equal(population, population2)


class Test_SVTree_Real(unittest.TestCase):

    def setUp(self):
        self.rhoNode = SVNode(
            description='rho',
            components=['rho_A'],
            constructor=['rho_A'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes=['Mo'],
        )

        self.ffgNode = SVNode(
            description='ffg',
            components=['f_A', 'g_AA'],
            constructor=['f_A', 'f_A', 'g_AA'],
            numParams=[9, 9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes=['Mo'],
        )

        r0 = 3.0
        dimer_aa = Atoms([1, 1], positions=[[0, 0, 0], [r0, 0, 0]]) 
        dimer_aa.set_chemical_symbols(['Mo', 'Mo'])

        self.dimer = dimer_aa

    
    def test_directEval_onerho_dimer(self):

        tree = SVTree(
            nodes=[
                deepcopy(self.rhoNode),
            ],
        )


        eng = tree.directEvaluation(
            np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            self.dimer,
            evalType='energy',
            bc_type='fixed',
            elements=['Mo'],
        )

        self.assertEqual(eng, 2.0)

    
    def test_directEval_tworho_dimer(self):

        tree = SVTree(
            nodes=[
                FunctionNode('add'),
                deepcopy(self.rhoNode),
                deepcopy(self.rhoNode),
            ],
        )


        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            ]),
            self.dimer,
            evalType='energy',
            bc_type='fixed',
            elements=['Mo'],
        )

        self.assertEqual(eng, 4.0)


if __name__ == '__main__':
    unittest.main()