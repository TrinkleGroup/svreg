import unittest
import numpy as np

from svreg.summation import Rho, FFG

from tests._testStructs import _all_test_structs


class Test_Summation(unittest.TestCase):

    def setUp(self):

        self.ffg = FFG(
            name='ffg',
            allElements=['H', 'He'],
            neighborElements=['H', 'He'],
            components=['f_A', 'f_B', 'g_AA', 'g_BB', 'g_AB'],
            inputTypes={
                'f_A': ['H'],
                'f_B': ['He'],
                'g_AA': ['H', 'H'],
                'g_AB': ['H', 'He'],
                'g_BB': ['He', 'He']
            },
            numParams={'f_A': 7, 'f_B': 7, 'g_AA': 9, 'g_BB': 9, 'g_AB': 9},
            restrictions={
                'f_A': [(6, 0), (8, 0)],
                'f_B': [(6, 0), (8, 0)],
                'g_AA':[],
                'g_AB':[],
                'g_BB':[],
            },
            paramRanges={
                'f_A': None,
                'f_B': None,
                'g_AA': None,
                'g_AB': None,
                'g_BB': None
            },
            bonds={
                'ffg_AA': ['f_A', 'f_A', 'g_AA'],
                'ffg_AB': ['f_A', 'f_B', 'g_AB'],
                'ffg_BB': ['f_B', 'f_B', 'g_BB'],
            },
            bondMapping="lambda i,j: 'ffg_AA' if i+j==0 else ('ffg_AB' if i+j==1 else 'ffg_BB')",
            cutoffs=[1.0, 3.0],
            numElements=2,
            bc_type='fixed',
        )

        self.rho = Rho(
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
            cutoffs=[1.0, 3.0],
            numElements=2,
            bc_type='fixed',
        )


    def test_sv_rho_dimers(self):
        params = {
            'rho_A': np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            'rho_B': np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
        }

        expected = {
            'aa': {'rho_A': 2.0, 'rho_B': 0.0},
            'ab': {'rho_A': 1.0, 'rho_B': 2.0},
            'bb': {'rho_A': 0.0, 'rho_B': 4.0},
        }

        for struct in ['aa', 'ab', 'bb']:
            atoms = _all_test_structs[struct]

            engSV, _ = self.rho.loop(atoms, evalType='vector')

            for bondType in ['rho_A', 'rho_B']:
                res = engSV[bondType].dot(params[bondType])

                self.assertEqual(sum(res), expected[struct][bondType])


    def test_sv_rho_trimers(self):
        params = {
            'rho_A': np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
            'rho_B': np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
        }

        expected = {
            'aaa': {'rho_A': 6.0, 'rho_B':  0.0},
            'bbb': {'rho_A': 0.0, 'rho_B': 12.0},
            'abb': {'rho_A': 2.0, 'rho_B':  8.0},
            'bab': {'rho_A': 2.0, 'rho_B':  8.0},
            'baa': {'rho_A': 4.0, 'rho_B':  4.0},
            'aba': {'rho_A': 4.0, 'rho_B':  4.0},
        }

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            atoms = _all_test_structs[struct]

            engSV, _ = self.rho.loop(atoms, evalType='vector')

            for bondType in ['rho_A', 'rho_B']:
                res = engSV[bondType].dot(params[bondType])

                self.assertEqual(sum(res), expected[struct][bondType])


    def test_sv_ffg_dimers(self):
        params = {
            'ffg_AA': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
            ]),
            'ffg_AB': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
            ]),
            'ffg_BB': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),
            ]),
        }

        params['ffg_AA'] = np.outer(
            np.outer(
                params['ffg_AA'][0], params['ffg_AA'][0]
            ).ravel(),
            params['ffg_AA'][1]
        ).ravel()

        params['ffg_AB'] = np.outer(
            np.outer(
                params['ffg_AB'][0], params['ffg_AB'][1]
            ).ravel(),
            params['ffg_AB'][2]
        ).ravel()

        params['ffg_BB'] = np.outer(
            np.outer(
                params['ffg_BB'][0], params['ffg_BB'][0]
            ).ravel(),
            params['ffg_BB'][1]
        ).ravel()

        expected = {
            'aa': {'ffg_AA': 0.0, 'ffg_AB': 0.0, 'ffg_BB': 0.0},
            'ab': {'ffg_AA': 0.0, 'ffg_AB': 0.0, 'ffg_BB': 0.0},
            'bb': {'ffg_AA': 0.0, 'ffg_AB': 0.0, 'ffg_BB': 0.0},
        }

        for struct in ['aa', 'ab', 'bb']:
            atoms = _all_test_structs[struct]

            engSV, _ = self.ffg.loop(atoms, evalType='vector')

            for bondType in ['ffg_AA', 'ffg_AB', 'ffg_BB']:
                res = engSV[bondType].dot(params[bondType])

                self.assertEqual(sum(res), expected[struct][bondType])


    def test_sv_ffg_trimers(self):
        params = {
            'ffg_AA': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([2, 2, 2, 2, 2, 2, 2, 0, 0]),
            ]),
            'ffg_AB': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([3, 3, 3, 3, 3, 3, 3, 0, 0]),
            ]),
            'ffg_BB': np.vstack([
                np.array([1, 1, 1, 1, 1, 1, 1, 0, 0]),
                np.array([4, 4, 4, 4, 4, 4, 4, 0, 0]),
            ]),
        }

        params['ffg_AA'] = np.outer(
            np.outer(
                params['ffg_AA'][0], params['ffg_AA'][0]
            ).ravel(),
            params['ffg_AA'][1]
        ).ravel()

        params['ffg_AB'] = np.outer(
            np.outer(
                params['ffg_AB'][0], params['ffg_AB'][1]
            ).ravel(),
            params['ffg_AB'][2]
        ).ravel()

        params['ffg_BB'] = np.outer(
            np.outer(
                params['ffg_BB'][0], params['ffg_BB'][0]
            ).ravel(),
            params['ffg_BB'][1]
        ).ravel()


        # 2 per ffg_AA
        # 3 per ffg_AB
        # 4 per ffg_BB

        expected = {
            'aaa': {'ffg_AA': 6.0, 'ffg_AB': 0.0, 'ffg_BB':  0.0},
            'bbb': {'ffg_AA': 0.0, 'ffg_AB': 0.0, 'ffg_BB': 12.0},
            'abb': {'ffg_AA': 0.0, 'ffg_AB': 6.0, 'ffg_BB':  4.0},
            'bab': {'ffg_AA': 0.0, 'ffg_AB': 6.0, 'ffg_BB':  4.0},
            'baa': {'ffg_AA': 2.0, 'ffg_AB': 6.0, 'ffg_BB':  0.0},
            'aba': {'ffg_AA': 2.0, 'ffg_AB': 6.0, 'ffg_BB':  0.0},
        }

        for struct in ['aaa', 'bbb', 'abb', 'bab', 'baa', 'aba']:
            atoms = _all_test_structs[struct]

            engSV, _ = self.ffg.loop(atoms, evalType='vector')

            for bondType in ['ffg_AA', 'ffg_AB', 'ffg_BB']:
                res = engSV[bondType].dot(params[bondType])

                np.testing.assert_almost_equal(
                    sum(res), expected[struct][bondType]
                )

