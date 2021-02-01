import unittest
import numpy as np
from copy import deepcopy

from ase.neighborlist import NeighborList

from svreg.tree import SVTree
from svreg.tree import MultiComponentTree as MCTree
from svreg.nodes import FunctionNode, SVNode
from svreg.exceptions import StaleValueException

from tests._testStructs import _all_test_structs


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



if __name__ == '__main__':
    unittest.main()