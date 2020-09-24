"""
An ASE Calculator object that implements generic tree evaluation.

TODO: make this object behave more like what is expected for general ASE
Calculators. Currently this class is only serving as a simple wrapper so that
it can replace a KIM calculator in a few OpenKIM tests.

Some expectations for ASE calculators:
    1) https://wiki.fysik.dtu.dk/ase/development/calculators.html
    2) https://wiki.fysik.dtu.dk/ase/development/proposals/calculators.html#aep1
"""

from ase.calculators.calculator import Calculator


class TreeCalculator(Calculator):
    def __init__(self, tree, y, *args, **kwargs):
        Calculator.__init__(self, *args, **kwargs)

        self.implemented_properties = ['energy', 'forces']
        self.discard_results_on_any_change = True

        self.y      = kwargs['y']
        self.tree   = kwargs['tree']


    def set_atoms(self, atoms):
        self.atoms = atoms


    def get_potential_energy(self):
        return self.tree.directEvaluation(self.y, self.atoms, 'energy')


    def get_forces(self):
        return self.tree.directEvaluation(self.y, self.atoms, 'forces')