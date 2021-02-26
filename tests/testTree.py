import unittest
import numpy as np
from copy import deepcopy

from ase.neighborlist import NeighborList

from svreg.tree import SVTree
from svreg.tree import MultiComponentTree as MCTree
from svreg.nodes import FunctionNode, SVNode
from svreg.exceptions import StaleValueException

from tests._testStructs import _all_test_structs

atol = 1e-10

def flat(x=None):
    if x is None:
        x = np.random.random()

    rand = np.ones(9)*x
    rand[-2] = rand[-1] = 0
    return rand


def angled(x=None):
    if x is None:
        x = np.random.random()

    rand = np.linspace(2.4, 5.2, 9)*x
    rand[-2] = rand[-1] = x
    return rand

def wiggly():
    rand = np.random.random(9)
    return rand


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
            svNode.values = (np.ones((1, 5))*i, np.zeros((1, 5, 5, 3)))
            
        self.tree = SVTree(
            nodes=[
                FunctionNode('add'),
                self.svNodePool[0],
                FunctionNode('add'),
                self.svNodePool[1],
                self.svNodePool[2],
            ],
        )

    def test_equality(self):
        tree1 = SVTree(
            nodes=[
                FunctionNode('add'),
                deepcopy(self.svNodePool[0]),
                FunctionNode('add'),
                deepcopy(self.svNodePool[1]),
                deepcopy(self.svNodePool[2]),
            ]
        )

        tree2 = SVTree(
            nodes=[
                FunctionNode('add'),
                deepcopy(self.svNodePool[0]),
                deepcopy(self.svNodePool[1]),
            ]
        )


        tree3 = SVTree(
            nodes=[
                FunctionNode('add'),
                deepcopy(self.svNodePool[0]),
                FunctionNode('add'),
                deepcopy(self.svNodePool[1]),
                FunctionNode('softplus'),
                deepcopy(self.svNodePool[1]),
            ]
        )

        tree4 = SVTree(
            nodes=[
                FunctionNode('add'),
                deepcopy(self.svNodePool[0]),
                FunctionNode('add'),
                deepcopy(self.svNodePool[1]),
                deepcopy(self.svNodePool[1]),
            ]
        )

        self.assertTrue(self.tree == tree1)

        self.assertFalse(self.tree == tree2)
        self.assertFalse(self.tree == tree3)
        self.assertFalse(self.tree == tree4)



    def test_verify_constructor(self):
        self.assertEqual(5, len(self.tree.nodes))
        self.assertEqual(3, len(self.tree.svNodes))

    
    def test_tree_print(self):
        self.assertEqual(str(self.tree), 'add(sv0, add(sv1, sv2))')

    
    def test_from_tree(self):
        treeStr = str(self.tree)
        newTree = SVTree.from_str(treeStr, self.svNodePool)

        self.assertTrue(self.tree == newTree)


    def test_1d_eval(self):
        for node in self.tree.svNodes:
            node.values = (np.array([[1, 2, 3]]), np.zeros((1, 5, 5, 3)))

        results = self.tree.eval()

        np.testing.assert_array_equal(results[0], np.array([18]))
        np.testing.assert_array_equal(results[1], np.zeros((1, 5, 3)))


    def test_2d_eval(self):
        for node in self.tree.svNodes:
            node.values = (
                np.array([[1, 2, 3], [1, 2, 3]]), np.zeros((2, 5, 5, 3))
            )

        results = self.tree.eval()

        np.testing.assert_array_equal(results[0], np.array([18, 18]))
        np.testing.assert_array_equal(results[1], np.zeros((2, 5, 3)))


    def test_repeat_eval_error(self):
        self.tree.eval()

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
        population  = self.tree.populate(3)
        popDict     = self.tree.parseArr2Dict(population, fillFixedKnots=True)

        # TODO: dummy test for now to assure no errors thrown; it's hard to
        # verify that popDict is correct without manually re-creating it, which
        # seems dumb to do


class Test_SVTree_Real(unittest.TestCase):

    def setUp(self):
        self.rhoNode = SVNode(
            description='rho',
            components=['rho_A'],
            constructor=['rho_A'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_A': ['H']},
        )

        self.ffgNode = SVNode(
            description='ffg',
            components=['f_A', 'g_AA'],
            constructor=['f_A', 'f_A', 'g_AA'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_A': ['H'], 'g_AA': ['H', 'H']},
        )

        self.cutoffs = [1.0, 3.0]
        self.elements = ['H', 'He']
    
    def test_directEval_onerho_dimer(self):

        tree = SVTree(
            nodes=[
                deepcopy(self.rhoNode),
            ],
        )


        eng = tree.directEvaluation(
            np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            _all_test_structs['aa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 2.0)

    
    def test_directEval_oneffg_dimer(self):
        tree = SVTree(nodes=[deepcopy(self.ffgNode)])
 
        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
            ]),
            _all_test_structs['aa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 0.0)

    
    def test_directEval_oneffg_trimer(self):
        tree = SVTree(nodes=[deepcopy(self.ffgNode)])
 
        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
            ]),
            _all_test_structs['aaa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 6.0)
    
    
    def test_directEval_twoffg_trimer(self):
        tree = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.ffgNode),
            deepcopy(self.ffgNode),
        ])
 
        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
                np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),
            ]),
            _all_test_structs['aaa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 114.0)
    

    def test_directEval_onerho_bvo1(self):
        tree = SVTree(
            nodes=[
                deepcopy(self.rhoNode),
            ],
        )


        eng = tree.directEvaluation(
            np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            _all_test_structs['bulk_vac_ortho_type1'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        # Expected energy = 2*number_of_bonds
        # number_of_bonds = N*(N-1)
        n = len(_all_test_structs['bulk_vac_ortho_type1'])

        nl = NeighborList(
            np.ones(n)*(self.cutoffs[-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(_all_test_structs['bulk_vac_ortho_type1'])

        self.assertEqual(eng, np.sum(nl.get_connectivity_matrix()))


    def test_directEval_tworho_bvo1(self):
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
            _all_test_structs['bulk_vac_ortho_type1'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        n = len(_all_test_structs['bulk_vac_ortho_type1'])

        nl = NeighborList(
            np.ones(n)*(self.cutoffs[-1]/2.),
            self_interaction=False, bothways=True, skin=0.0
        )

        nl.update(_all_test_structs['bulk_vac_ortho_type1'])

        self.assertEqual(eng, 2*np.sum(nl.get_connectivity_matrix()))


     
    def test_directEval_onerho_trimer(self):

        tree = SVTree(
            nodes=[
                deepcopy(self.rhoNode),
            ],
        )

        eng = tree.directEvaluation(
            np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            _all_test_structs['aaa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 6.0)

       
    def test_directEval_tworho_trimer(self):

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
            _all_test_structs['aaa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 12.0)

    
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
            _all_test_structs['aa'],
            self.elements,
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 4.0)


class Test_MCTree_Real(unittest.TestCase):

    # rzm: get Rho working without the nhost dimension

    def setUp(self):
        self.rho_A = SVNode(
            description='rho_A',
            components=['rho_A'],
            constructor=['rho_A'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_A': ['H']},
        )

        self.rho_B = SVNode(
            description='rho_B',
            components=['rho_B'],
            constructor=['rho_B'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_B': ['He']},
        )

        self.ffg_AA = SVNode(
            description='ffg_AA',
            components=['f_A', 'g_AA'],
            constructor=['f_A', 'f_A', 'g_AA'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_A': ['H'], 'g_AA': ['H', 'H']},
        )


        self.ffg_AB = SVNode(
            description='ffg_AB',
            components=['f_A', 'f_B', 'g_AB'],
            constructor=['f_A', 'f_B', 'g_AB'],
            numParams=[9, 9, 9],
            restrictions=[[(6, 0), (8, 0)], [(6, 0), (8, 0)], []],
            paramRanges=[None, None, None],
            inputTypes={'f_A': ['H'], 'f_B': ['He'], 'g_AB': ['H', 'He']},
        )


        self.ffg_BB = SVNode(
            description='ffg_BB',
            components=['f_B', 'g_BB'],
            constructor=['f_B', 'f_B', 'g_BB'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_B': ['He'], 'g_BB': ['He', 'He']},
        )

        self.cutoffs = [1.0, 3.0]


    def test_equality_element_mismatch(self):

        tree1 = MCTree(['H', 'He'])
        tree2 = MCTree(['H', 'Li'])

        self.assertFalse(tree1 == tree2)

    
    def test_equality(self):

        tree1 = MCTree(['H', 'He'])

        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_A)])

        tree1.chemistryTrees['H']    = subtree1
        tree1.chemistryTrees['He']   = subtree2

        tree1.updateSVNodes()

        tree2 = deepcopy(tree1)

        self.assertTrue(tree1 == tree2)

        tree3 = MCTree(['H', 'He'])

        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(
            nodes=[
                FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.ffg_AB)
            ]
        )

        tree3.chemistryTrees['H']    = subtree1
        tree3.chemistryTrees['He']   = subtree2

        tree3.updateSVNodes()

        self.assertFalse(tree1 == tree3)

        tree4 = MCTree(['H', 'He'])

        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(
            nodes=[
                FunctionNode('add'), deepcopy(self.ffg_AB), deepcopy(self.rho_A)
            ]
        )

        tree4.chemistryTrees['H']    = subtree1
        tree4.chemistryTrees['He']   = subtree2

        tree4.updateSVNodes()

        self.assertTrue(tree3 == tree4)


    def test_from_str(self):

        tree = MCTree(['H', 'He'])

        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(
            nodes=[
                FunctionNode('add'), deepcopy(self.ffg_AB), deepcopy(self.rho_A)
            ]
        )

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        treeStr = str(tree)
        
        svNodePool = [self.rho_A, self.rho_B, self.ffg_AA, self.ffg_AB, self.ffg_BB]
        newTree = MCTree.from_str(treeStr, ['H', 'He'], svNodePool)

        self.assertTrue(tree == newTree)


    def test_directEval_rho_a_rho_a_dimer_aa(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_A)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['aa'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 2.0)
 
        # fcs = tree.directEvaluation(
        #     np.concatenate([
        #         np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
        #         np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
        #     ]),
        #     _all_test_structs['aa'],
        #     evalType='forces',
        #     bc_type='fixed',
        #     cutoffs=self.cutoffs,
        # )

        # np.testing.assert_allclose(fcs, 0.0)
                   
    def test_directEval_rho_a_rho_a_dimer_ab(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_A)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['ab'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 2.0)

             
    def test_directEval_rho_a_rho_a_dimer_bb(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_A)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['bb'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 0.0)
     
              
    def test_directEval_rho_a_rho_b_dimer_aa(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_B)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['aa'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 2.0)

             
    def test_directEval_rho_a_rho_b_dimer_ab(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_B)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['ab'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 0.0)

              
    def test_directEval_rho_a_rho_b_dimer_bb(self):

        tree = MCTree(['H', 'He'])
        
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree2 = SVTree(nodes=[deepcopy(self.rho_B)])

        tree.chemistryTrees['H']    = subtree1
        tree.chemistryTrees['He']   = subtree2

        tree.updateSVNodes()

        eng = tree.directEvaluation(
            np.concatenate([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
            ]),
            _all_test_structs['bb'],
            evalType='energy',
            bc_type='fixed',
            cutoffs=self.cutoffs,
        )

        self.assertEqual(eng, 4.0)
   
              
    def test_directEval_rho_rho_trimers(self):


        rhoPairs = [
            (deepcopy(self.rho_A), deepcopy(self.rho_A)),
            (deepcopy(self.rho_A), deepcopy(self.rho_B)),
            (deepcopy(self.rho_B), deepcopy(self.rho_B)),
        ]

        # A looking for *, B looking for *

        expectedResults = [
            [6.0,  0.0, 4.0, 4.0, 6.0, 6.0],
            [6.0, 12.0, 4.0, 4.0, 2.0, 2.0],
            [0.0, 12.0, 6.0, 6.0, 2.0, 2.0],
        ]

        for pair, expected in zip(rhoPairs, expectedResults):
            tree = MCTree(['H', 'He'])

            subtree0 = SVTree(nodes=[pair[0]])
            subtree1 = SVTree(nodes=[pair[1]])
        
            tree.chemistryTrees['H']    = subtree0
            tree.chemistryTrees['He']   = subtree1

            tree.updateSVNodes()

            for tri, true in zip(
                ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
                expected
            ):

                eng = tree.directEvaluation(
                    np.concatenate([
                        np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # H
                        np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # He
                    ]),
                    _all_test_structs[tri],
                    evalType='energy',
                    bc_type='fixed',
                    cutoffs=self.cutoffs,
                )

                self.assertEqual(eng, true)
    

    def test_directEval_ffg_AA_ffg_AA_trimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[deepcopy(self.ffg_AA)])
        subtree1 = SVTree(nodes=[deepcopy(self.ffg_AA)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [6.0,    0.0,    0.0,  0.0,  36.0, 36.0],
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per triplet
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 36 per triplet
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)


    def test_directEval_ffg_AB_ffg_AB_trimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[deepcopy(self.ffg_AB)])
        subtree1 = SVTree(nodes=[deepcopy(self.ffg_AB)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [0.0,    0.0,  72.0,  72.0,   4.0,   4.0],
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per triplet
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 36 per triplet
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)


    def test_directEval_ffg_AA_ffg_AB_trimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[deepcopy(self.ffg_AA)])
        subtree1 = SVTree(nodes=[deepcopy(self.ffg_AB)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [6.0,    0.0,  72.0,  72.0,   0.0,   0.0],
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per triplet
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 36 per triplet
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)


    def test_directEval_mixed_rho_dimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
                FunctionNode('add'),
                deepcopy(self.rho_A),
                deepcopy(self.rho_B)
        ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            deepcopy(self.rho_B)
        ])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aa', 'ab', 'bb'],
            [2.0,  5.0,  8.0]
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # 1 per AA
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # 2 per AB
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),  # 3 per BA
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 4 per BB
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)



    def test_directEval_mixed_rho_trimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
                FunctionNode('add'),
                deepcopy(self.rho_A),
                deepcopy(self.rho_B)
        ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            deepcopy(self.rho_B)
        ])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [6.0,   24.0,  18.0,  18.0,  12.0,   12.0],
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  # 1 per AA
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  # 2 per AB
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),  # 3 per BA
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 4 per BB
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)


    def test_directEval_mixed_ffg_trimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
                FunctionNode('add'),
                deepcopy(self.ffg_AA),
                deepcopy(self.ffg_AB)
        ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
        ])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [6.0,  108.0,  72.0,  72.0,   4.0,   4.0],
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per triplet
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per triplet
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 36 per triplet
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  # 36 per triplet
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)



    def test_directEval_mixed_rho_ffg_dimers(self):

        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB),
        ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB),
        ])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()
 
        for tri, true in zip(
            ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba'],
            [15.0,  72.0,  53.0,  53.0,  34.0,  34.0]
        ):

            eng = tree.directEvaluation(
                np.concatenate([
                    # subtree 1
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),  #  1 per AA
                    np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),  #  2 per AB
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),  #  3 per AAA
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),  #  4 per AAB
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([5, 5, 5, 5, 5, 5, 5, 0, 0]),  #  5 per BAB
                    # subtree 2
                    np.array([6, 6, 6, 6, 6, 6, 6, 0, 0]),  #  6 per BA
                    np.array([7, 7, 7, 7, 7, 7, 7, 0, 0]),  #  7 per BB
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([8, 8, 8, 8, 8, 8, 8, 0, 0]),  #  8 per ABA
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([9, 9, 9, 9, 9, 9, 9, 0, 0]),  #  9 per ABB
                    np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                    np.array([10, 10, 10, 10, 10, 10, 10, 0, 0]),  # 10 per BBB
                ]),
                _all_test_structs[tri],
                evalType='energy',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            self.assertEqual(eng, true)


class Test_Direct_vs_SV_Ti48Mo80_type1_c10(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        import os
        from ase.io import read
        from svreg.summation import Rho, FFG

        ffg = FFG(
            name='ffg',
            allElements=['Mo', 'Ti'],
            neighborElements=['Mo', 'Ti'],
            components=['f_A', 'f_B', 'g_AA', 'g_BB', 'g_AB'],
            inputTypes={'f_A': ['Mo'], 'f_B': ['Ti'], 'g_AA': ['Mo', 'Mo'], 'g_AB': ['Mo', 'Ti'], 'g_BB': ['Ti', 'Ti']},
            numParams={'f_A': 7, 'f_B': 7, 'g_AA': 9, 'g_BB': 9, 'g_AB': 9},
            restrictions={
                'f_A': [(6, 0), (8, 0)],
                'f_B': [(6, 0), (8, 0)],
                'g_AA':[],
                'g_AB':[],
                'g_BB':[],
            },
            paramRanges={'f_A': None, 'f_B': None, 'g_AA': None, 'g_AB': None, 'g_BB': None},
            bonds={
                'ffg_AA': ['f_A', 'f_A', 'g_AA'],
                'ffg_AB': ['f_A', 'f_B', 'g_AB'],
                'ffg_BB': ['f_B', 'f_B', 'g_BB'],
            },
            bondMapping="lambda i,j: 'ffg_AA' if i+j==0 else ('ffg_AB' if i+j==1 else 'ffg_BB')",
            cutoffs=[2.4, 5.2],
            numElements=2,
            bc_type='fixed',
        )

        rho = Rho(
            name='rho',
            allElements=['Mo', 'Ti'],
            neighborElements=['Mo', 'Ti'],
            components=['rho_A', 'rho_B'],
            inputTypes={'rho_A': ['Mo'], 'rho_B': ['Ti']},
            numParams={'rho_A': 7, 'rho_B': 7},
            restrictions={'rho_A': [(6, 0), (8, 0)], 'rho_B': [(6, 0), (8, 0)]},
            paramRanges={'rho_A': None, 'rho_B': None},
            bonds={
                'rho_A': ['rho_A'],
                'rho_B': ['rho_B'],
            },
            bondMapping="lambda i: 'rho_A' if i == 0 else 'rho_B'",
            cutoffs=[2.4, 5.2],
            numElements=2,
            bc_type='fixed',
        )

        engSV = {'rho': None, 'ffg': None}
        fcsSV = {'rho': None, 'ffg': None}

        struct = 'Ti48Mo80_type1_c10'

        atoms = read(
            os.path.join('examples/{}.data'.format(struct)),
            format='lammps-data',
            style='atomic'
        )

        types = np.array([
            'Ti' if t == 'H' else 'Mo' for t in atoms.get_chemical_symbols()
        ])

        atoms.set_chemical_symbols(types)

        engSV['rho'], fcsSV['rho'] = rho.loop(atoms, evalType='vector')
        engSV['ffg'], fcsSV['ffg'] = ffg.loop(atoms, evalType='vector')

        elements = ['Mo', 'Ti']

        miniDatabase = {
            el: {
                evalType: {
                    'rho': {'rho_A': None, 'rho_B': None},
                    'ffg': {'ffg_AA': None, 'ffg_AB': None, 'ffg_BB': None}
                } for evalType in ['energy', 'forces']
            }
            for el in elements
        }

        n = len(atoms)

        for svName in ['rho', 'ffg']:
            k = 729 if svName == 'ffg' else 9

            for bondType in engSV[svName].keys():
  
                for elem in ['Mo', 'Ti']:
                    where = np.where(types == elem)[0]

                    miniDatabase[elem]['energy'][svName][bondType] = engSV[svName][bondType][where, :]
          
                    fcsSplit = fcsSV[svName][bondType]
                    fcsSplit = fcsSplit.reshape((n, n, 3, k))
                    fcsSplit = fcsSplit[where, :, :, :]
                    miniDatabase[elem]['forces'][svName][bondType] = fcsSplit.copy()
                   
        from svreg.tree import SVTree
        from svreg.nodes import FunctionNode
        from svreg.tree import MultiComponentTree as MCTree

        rho_A = SVNode(
            description='rho_A',
            components=['rho_A'],
            constructor=['rho_A'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_A': ['Mo']},
        )

        rho_B = SVNode(
            description='rho_B',
            components=['rho_B'],
            constructor=['rho_B'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_B': ['Ti']},
        )

        ffg_AA = SVNode(
            description='ffg_AA',
            components=['f_A', 'g_AA'],
            constructor=['f_A', 'f_A', 'g_AA'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_A': ['Mo'], 'g_AA': ['Mo', 'Mo']},
        )


        ffg_AB = SVNode(
            description='ffg_AB',
            components=['f_A', 'f_B', 'g_AB'],
            constructor=['f_A', 'f_B', 'g_AB'],
            numParams=[9, 9, 9],
            restrictions=[[(6, 0), (8, 0)], [(6, 0), (8, 0)], []],
            paramRanges=[None, None, None],
            inputTypes={'f_A': ['Mo'], 'f_B': ['Ti'], 'g_AB': ['Mo', 'Ti']},
        )


        ffg_BB = SVNode(
            description='ffg_BB',
            components=['f_B', 'g_BB'],
            constructor=['f_B', 'f_B', 'g_BB'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_B': ['Ti'], 'g_BB': ['Ti', 'Ti']},
        )

        dummyTree = MCTree(['Mo', 'Ti'])

        from copy import deepcopy

        treeMo = SVTree()
        treeMo.nodes = [
            FunctionNode('add'),
            deepcopy(rho_A),
            FunctionNode('add'),
            deepcopy(rho_B),
            FunctionNode('add'),
            deepcopy(ffg_AA),
            FunctionNode('add'),
            deepcopy(ffg_AB),
            deepcopy(ffg_BB),
        ]

        treeTi = SVTree()
        treeTi.nodes = [
            FunctionNode('add'),
            deepcopy(rho_A),
            FunctionNode('add'),
            deepcopy(rho_B),
            FunctionNode('add'),
            deepcopy(ffg_AA),
            FunctionNode('add'),
            deepcopy(ffg_AB),
            deepcopy(ffg_BB),
        ]

        dummyTree.chemistryTrees['Mo'] = treeMo
        dummyTree.chemistryTrees['Ti'] = treeTi

        dummyTree.updateSVNodes()

        cls.dummyTree = dummyTree
        cls.miniDatabase = miniDatabase
        cls.atoms = atoms


    def test_flat_splines(self):
        dummyParams = np.concatenate([flat() for _ in range(18)])

        popDict = self.dummyTree.parseArr2Dict(
            np.atleast_2d(dummyParams), fillFixedKnots=False
        )

        totalMo = 0

        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalMo = sum(totalMo)

        totalTi = 0

        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        totalTi = sum(totalTi)

        engSVmethod = totalMo + totalTi

        engDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'energy',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        np.testing.assert_allclose(engSVmethod, engDirectMethod)

        totalMoForces = 0

        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalTiForces = 0

        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        fcsDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'forces',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        svForces = np.sum(totalMoForces, axis=0) + np.sum(totalTiForces, axis=0)
        svForces = np.moveaxis(svForces, -1, 0)
        np.testing.assert_allclose(fcsDirectMethod, svForces[0], atol=atol)


    def test_angled_splines(self):

        dummyParams = np.concatenate([angled() for _ in range(18)])

        popDict = self.dummyTree.parseArr2Dict(
            np.atleast_2d(dummyParams), fillFixedKnots=False
        )

        totalMo = 0

        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalMo = sum(totalMo)

        totalTi = 0

        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        totalTi = sum(totalTi)

        engSVmethod = totalMo + totalTi

        engDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'energy',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        np.testing.assert_allclose(engSVmethod, engDirectMethod)

        totalMoForces = 0

        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalTiForces = 0

        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        fcsDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'forces',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        svForces = np.sum(totalMoForces, axis=0) + np.sum(totalTiForces, axis=0)
        svForces = np.moveaxis(svForces, -1, 0)
        np.testing.assert_allclose(fcsDirectMethod, svForces[0], atol=atol)


    def test_wiggly_splines(self):

        dummyParams = np.concatenate([wiggly() for _ in range(18)])

        popDict = self.dummyTree.parseArr2Dict(
            np.atleast_2d(dummyParams), fillFixedKnots=False
        )

        totalMo = 0

        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalMo = sum(totalMo)

        totalTi = 0

        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        totalTi = sum(totalTi)

        engSVmethod = totalMo + totalTi

        engDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'energy',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        np.testing.assert_allclose(engSVmethod, engDirectMethod)

        totalMoForces = 0

        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMoForces += self.miniDatabase['Mo']['forces']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalTiForces = 0

        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTiForces += self.miniDatabase['Ti']['forces']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        fcsDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'forces',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        svForces = np.sum(totalMoForces, axis=0) + np.sum(totalTiForces, axis=0)
        svForces = np.moveaxis(svForces, -1, 0)
        np.testing.assert_allclose(fcsDirectMethod, svForces[0], atol=atol)


    def test_linear_rho_wiggly_ffg_0end(self):
        # rho_A rho_B ffg_AA ffg_AB ffg_BB
        dummyParams = np.concatenate([
            flat(),
            flat(),
            wiggly(), wiggly(),
            wiggly(), wiggly(), wiggly(),
            wiggly(), wiggly(),
            flat(),
            flat(),
            wiggly(), wiggly(),
            wiggly(), wiggly(), wiggly(),
            wiggly(), wiggly(),
        ])

        popDict = self.dummyTree.parseArr2Dict(
            np.atleast_2d(dummyParams), fillFixedKnots=False
        )

        totalMo = 0

        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalMo = sum(totalMo)

        totalTi = 0

        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        totalTi = sum(totalTi)

        engSVmethod = totalMo + totalTi

        engDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'energy',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        np.testing.assert_allclose(engSVmethod, engDirectMethod)


    def test_wiggly_rho_linear_0end(self):

        # rho_A rho_B ffg_AA ffg_AB ffg_B
        dummyParams = np.concatenate([
            wiggly(),
            wiggly(),
            flat(), flat(),
            flat(), flat(), flat(),
            flat(), flat(),
            wiggly(),
            wiggly(),
            flat(), flat(),
            flat(), flat(), flat(),
            flat(), flat(),
        ])

        popDict = self.dummyTree.parseArr2Dict(
            np.atleast_2d(dummyParams), fillFixedKnots=False
        )

        totalMo = 0

        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AA'] @ popDict['Mo']['ffg_AA'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_AB'] @ popDict['Mo']['ffg_AB'].T
        totalMo += self.miniDatabase['Mo']['energy']['ffg']['ffg_BB'] @ popDict['Mo']['ffg_BB'].T

        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_A'] @ popDict['Mo']['rho_A'].T
        totalMo += self.miniDatabase['Mo']['energy']['rho']['rho_B'] @ popDict['Mo']['rho_B'].T

        totalMo = sum(totalMo)

        totalTi = 0

        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AA'] @ popDict['Ti']['ffg_AA'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_AB'] @ popDict['Ti']['ffg_AB'].T
        totalTi += self.miniDatabase['Ti']['energy']['ffg']['ffg_BB'] @ popDict['Ti']['ffg_BB'].T

        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_A'] @ popDict['Ti']['rho_A'].T
        totalTi += self.miniDatabase['Ti']['energy']['rho']['rho_B'] @ popDict['Ti']['rho_B'].T

        totalTi = sum(totalTi)

        engSVmethod = totalMo + totalTi

        engDirectMethod = self.dummyTree.directEvaluation(
            dummyParams,
            self.atoms,
            'energy',
            'fixed',
            cutoffs=[2.4, 5.2]
        )

        np.testing.assert_allclose(engSVmethod, engDirectMethod)


class Test_MCTree_Real_Forces(unittest.TestCase):

    def setUp(self):
        self.rho_A = SVNode(
            description='rho_A',
            components=['rho_A'],
            constructor=['rho_A'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_A': ['H']},
        )

        self.rho_B = SVNode(
            description='rho_B',
            components=['rho_B'],
            constructor=['rho_B'],
            numParams=[9],
            restrictions=[(6, 0), (8, 0)],
            paramRanges=[None],
            inputTypes={'rho_B': ['He']},
        )

        self.ffg_AA = SVNode(
            description='ffg_AA',
            components=['f_A', 'g_AA'],
            constructor=['f_A', 'f_A', 'g_AA'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_A': ['H'], 'g_AA': ['H', 'H']},
        )


        self.ffg_AB = SVNode(
            description='ffg_AB',
            components=['f_A', 'f_B', 'g_AB'],
            constructor=['f_A', 'f_B', 'g_AB'],
            numParams=[9, 9, 9],
            restrictions=[[(6, 0), (8, 0)], [(6, 0), (8, 0)], []],
            paramRanges=[None, None, None],
            inputTypes={'f_A': ['H'], 'f_B': ['He'], 'g_AB': ['H', 'He']},
        )


        self.ffg_BB = SVNode(
            description='ffg_BB',
            components=['f_B', 'g_BB'],
            constructor=['f_B', 'f_B', 'g_BB'],
            numParams=[9, 9],
            restrictions=[[(6, 0), (8, 0)], []],
            paramRanges=[None, None],
            inputTypes={'f_B': ['He'], 'g_BB': ['He', 'He']},
        )

        self.cutoffs = [1.0, 50.0]

        from svreg.summation import Rho, FFG

        ffg = FFG(
            name='ffg',
            allElements=['H', 'He'],
            neighborElements=['H', 'He'],
            components=['f_A', 'f_B', 'g_AA', 'g_BB', 'g_AB'],
            inputTypes={'f_A': ['H'], 'f_B': ['He'], 'g_AA': ['H', 'H'], 'g_AB': ['H', 'He'], 'g_BB': ['He', 'He']},
            numParams={'f_A': 7, 'f_B': 7, 'g_AA': 9, 'g_BB': 9, 'g_AB': 9},
            restrictions={
                'f_A': [(6, 0), (8, 0)],
                'f_B': [(6, 0), (8, 0)],
                'g_AA':[],
                'g_AB':[],
                'g_BB':[],
            },
            paramRanges={'f_A': None, 'f_B': None, 'g_AA': None, 'g_AB': None, 'g_BB': None},
            bonds={
                'ffg_AA': ['f_A', 'f_A', 'g_AA'],
                'ffg_AB': ['f_A', 'f_B', 'g_AB'],
                'ffg_BB': ['f_B', 'f_B', 'g_BB'],
            },
            bondMapping="lambda i,j: 'ffg_AA' if i+j==0 else ('ffg_AB' if i+j==1 else 'ffg_BB')",
            cutoffs=self.cutoffs,
            numElements=2,
            bc_type='fixed',
        )

        rho = Rho(
            name='rho',
            allElements=['H', 'He'],
            neighborElements=['H', 'He'],
            components=['rho_A', 'rho_B'],
            inputTypes={'rho_A': ['H'], 'rho_B': ['He']},
            numParams={'rho_A': 7, 'rho_B': 7},
            restrictions={'rho_A': [(6, 0), (8, 0)], 'rho_B': [(6, 0), (8, 0)]},
            paramRanges={'rho_A': None, 'rho_B': None},
            bonds={
                'rho_A': ['rho_A'],
                'rho_B': ['rho_B'],
            },
            bondMapping="lambda i: 'rho_A' if i == 0 else 'rho_B'",
            cutoffs=self.cutoffs,
            numElements=2,
            bc_type='fixed',
        )


        def miniDb(atoms):

            engSV = {'rho': None, 'ffg': None}
            fcsSV = {'rho': None, 'ffg': None}

            engSV['rho'], fcsSV['rho'] = rho.loop(atoms, evalType='vector')
            engSV['ffg'], fcsSV['ffg'] = ffg.loop(atoms, evalType='vector')

            elements = ['H', 'He']

            miniDatabase = {
                el: {
                    evalType: {
                        'rho': {'rho_A': None, 'rho_B': None},
                        'ffg': {'ffg_AA': None, 'ffg_AB': None, 'ffg_BB': None}
                    } for evalType in ['energy', 'forces']
                }
                for el in elements
            }

            types = np.array(atoms.get_chemical_symbols())

            n = len(atoms)

            for svName in ['rho', 'ffg']:
                k = 729 if svName == 'ffg' else 9

                for bondType in engSV[svName].keys():

                    for elem in ['H', 'He']:
                        where       = np.where(types == elem)[0]

                        miniDatabase[elem]['energy'][svName][bondType] = engSV[svName][bondType][where, :]

                        fcsSplit = fcsSV[svName][bondType]
                        fcsSplit = fcsSplit.reshape((n, n, 3, k))
                        fcsSplit = fcsSplit[where, :, :, :]
                        miniDatabase[elem]['forces'][svName][bondType] = fcsSplit.copy()
                    
            return miniDatabase

        self.miniDb = miniDb
        
  
    def test_directEval_rho_A_rho_A_dimers(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aa', 'ab', 'bb']:
            fcs = tree.directEvaluation(
                np.concatenate([
                    flat(1),  # H
                    flat(2),  # He
                ]),
                _all_test_structs[struct],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0)

      
    def test_directEval_rho_A_rho_A_trimers(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[deepcopy(self.rho_A)])
        subtree1 = SVTree(nodes=[deepcopy(self.rho_A)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            fcs = tree.directEvaluation(
                np.concatenate([
                    flat(1),  # H
                    flat(2),  # He
                ]),
                _all_test_structs[struct],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)
    
      
    def test_directEval_rho_A_rho_B_trimers_linear(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])
        subtree1 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])
        
        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            fcs = tree.directEvaluation(
                np.concatenate([
                    angled(10),  # H on H
                    angled(20),  # He on H
                    angled(10),  # H on He
                    angled(20),  # He on He
                ]),
                _all_test_structs[struct+'_lin'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)
          

    def test_direct_vs_SV_rho_A_rho_B_trimers_linear_angled(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])
        subtree1 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([
                angled(10),  # H on H
                angled(20),  # He on H
                angled(10),  # H on He
                angled(20),  # He on He
            ])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_lin'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_lin'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_symmetric_angled(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([angled() for _ in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_linear_angled(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([angled() for _ in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_lin'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_lin'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_asymmetric_angled(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([angled(i) for i in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_asym'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_asym'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_rho_A_rho_B_trimers_linear_wiggly(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])
        subtree1 = SVTree(nodes=[FunctionNode('add'), deepcopy(self.rho_A), deepcopy(self.rho_B)])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([
                wiggly(),  # H on H
                wiggly(),  # He on H
                wiggly(),  # H on He
                wiggly(),  # He on He
            ])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_lin'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
        )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_lin'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_symmetric_wiggly(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([wiggly() for _ in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_linear_wiggly(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([wiggly() for _ in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_lin'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_lin'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)


    def test_direct_vs_SV_mixed_trimers_asymmetric_wiggly(self):
        tree = MCTree(['H', 'He'])
        
        subtree0 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        subtree1 = SVTree(nodes=[
            FunctionNode('add'),
            deepcopy(self.rho_A),
            FunctionNode('add'),
            deepcopy(self.rho_B),
            FunctionNode('add'),
            deepcopy(self.ffg_AA),
            FunctionNode('add'),
            deepcopy(self.ffg_AB),
            deepcopy(self.ffg_BB)
            ])

        tree.chemistryTrees['H']    = subtree0
        tree.chemistryTrees['He']   = subtree1

        tree.updateSVNodes()

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            y = np.concatenate([wiggly() for _ in range(18)])

            fcs = tree.directEvaluation(
                y,
                _all_test_structs[struct+'_asym'],
                evalType='forces',
                bc_type='fixed',
                cutoffs=self.cutoffs,
            )

            miniDatabase = self.miniDb(_all_test_structs[struct+'_asym'])

            popDict = tree.parseArr2Dict(np.atleast_2d(y), fillFixedKnots=False)

            totalHForces = 0

            totalHForces += miniDatabase['H']['forces']['rho']['rho_A'] @ popDict['H']['rho_A'].T
            totalHForces += miniDatabase['H']['forces']['rho']['rho_B'] @ popDict['H']['rho_B'].T

            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AA'] @ popDict['H']['ffg_AA'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_AB'] @ popDict['H']['ffg_AB'].T
            totalHForces += miniDatabase['H']['forces']['ffg']['ffg_BB'] @ popDict['H']['ffg_BB'].T

            totalHeForces = 0

            totalHeForces += miniDatabase['He']['forces']['rho']['rho_A'] @ popDict['He']['rho_A'].T
            totalHeForces += miniDatabase['He']['forces']['rho']['rho_B'] @ popDict['He']['rho_B'].T

            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AA'] @ popDict['He']['ffg_AA'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_AB'] @ popDict['He']['ffg_AB'].T
            totalHeForces += miniDatabase['He']['forces']['ffg']['ffg_BB'] @ popDict['He']['ffg_BB'].T

            np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)

            svForces = np.sum(totalHForces, axis=0) + np.sum(totalHeForces, axis=0)
            svForces = np.moveaxis(svForces, -1, 0)
            np.testing.assert_allclose(fcs, svForces[0], atol=atol)

      

    def test_nonlinear_embedders_dont_crash(self):
        from svreg.functions import _function_map

        for fxn in _function_map:
            tree = MCTree(['H', 'He'])
            
            subtree0 = SVTree(nodes=[
                FunctionNode(fxn),
                deepcopy(self.rho_A),
                deepcopy(self.ffg_AB),
                ])

            subtree1 = SVTree(nodes=[
                FunctionNode(fxn),
                deepcopy(self.rho_A),
                deepcopy(self.ffg_AB),
                ])

            tree.chemistryTrees['H']    = subtree0
            tree.chemistryTrees['He']   = subtree1

            tree.updateSVNodes()

            for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
                y = np.concatenate([wiggly() for _ in range(18)])

                fcs = tree.directEvaluation(
                    y,
                    _all_test_structs[struct+'_asym'],
                    evalType='forces',
                    bc_type='fixed',
                    cutoffs=self.cutoffs,
                )

                np.testing.assert_allclose(np.sum(fcs, axis=0), 0.0, atol=atol)
         

if __name__ == '__main__':
    unittest.main()