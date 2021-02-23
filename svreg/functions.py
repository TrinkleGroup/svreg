import numpy as np

from gplearn.functions import _function_map as _gp_function_map
from gplearn.functions import _Function as _gp_Function

import dask

_protected_sqrt = _gp_function_map['sqrt'].function
_protected_log  = _gp_function_map['log'].function
_protected_inverse = _gp_function_map['inv'].function

"""
When using forward automatic differentiation, it is useful to work with tuples
of (value, derivative) pairs, then to define functions that operate on the
tuples. This allows for much simpler software implementation.

For each function, each input will be a tuple where the first entry of the tuple
is the per-atom energies of the shape (P, N) and the second entry of the tuple
is a force of the (P, N, N, 3) where forces[i, j] denotes the forces on j due to
atom i.

For computing derivatives:

Functions must be explicitly defined to account for the chain rule when
computing derivatives, and to apply the derivatives to the correct dimensions of
the force matrices.

In general, the process for computing the derivative is to compute U' (where U
is the embedding function), then to scale forces[i, :, :, :] by U'. Note that U'
should be the derivative of the embedding function evaluated at each per-atom
energy value.

"""

#@dask.delayed
def _derivative_add(a, b):
    """Derivative of a + b"""
    return a[1] + b[1]


# #@dask.delayed
# def _derivative_tan(x):
#     """Derivative of tan(x)"""
#     return _protected_inverse(np.cos(x[1])**2)


# #@dask.delayed
# def _derivative_sin(x):
#     """Derivative of sin(x)"""
#     return np.cos(x[1])


# #@dask.delayed
# def _derivative_cos(x):
#     """Derivative of cos(x)"""
#     return -1*np.sin(x[1])


# #@dask.delayed
# def _derivative_inv(x):
#     """Derivative of x**-1"""
#     return -1*(_protected_inverse(x[1])**2)


#@dask.delayed
def _derivative_log(x):
    """Chain rule on natural log = (1/x)*(dx/dr)"""
    return _protected_inverse(x[0])[:, :, np.newaxis, np.newaxis]*x[1]


#@dask.delayed
def _derivative_mul(a, b):
    """Chain rule on a*b = a*(db/dr) + (da/dr)*b"""
    return (
        a[0][:, :, np.newaxis, np.newaxis]*b[1]
        + a[1]*b[0][:, :, np.newaxis, np.newaxis]
    )


# #@dask.delayed
# def _derivative_sqrt(x):
#     """(1/2)x**(-1/2)"""
#     return 0.5*_protected_inverse(_protected_sqrt(x[1]))


#@dask.delayed
def _derivative_exp(x):
    """Chain rule on exp(x) = exp(x)*(dx/dr)"""
    return np.exp(x[0])[:, :, np.newaxis, np.newaxis]*x[1]


#@dask.delayed
def sigmoid(x):
    return 1/(1+np.exp(-x))


#@dask.delayed
def _derivative_sigmoid(x):
    """Chain rule on sig() = sig'()*(dx/dr), where sig'() = sig()*(1-sig())"""
    return (sigmoid(x[0])*(1-sigmoid(x[0])))[:, :, np.newaxis, np.newaxis]*x[1]


#@dask.delayed
def splus(x):
    # Truncated softplus function
    res = np.log(1+np.exp(x))

    badIndices = np.where(x > 1e6)
    res[badIndices] = x[badIndices]

    return res


#@dask.delayed
def _derivative_splus(x):
    """
    Chain rule on splus() = splus'()*(dx/dr), where splus'() = exp(x)/(1+exp(x))
    """
    res = np.exp(x[0])/(1+np.exp(x[0]))

    badIndices = np.where(x[0] > 1e6)
    res[badIndices] = 1

    return res[:, :, np.newaxis, np.newaxis]*x[1]


_derivative_map = {
    'add': _derivative_add,
    'mul': _derivative_mul,
    # 'sqrt': _derivative_sqrt,
    'log': _derivative_log,
    # 'inv': _derivative_inv,
    # 'cos': _derivative_cos,
    # 'sin': _derivative_sin,
    # 'tan': _derivative_tan,
    'exp': _derivative_exp,
    'sig': _derivative_sigmoid,
    'softplus': _derivative_splus,
}


class _Function(_gp_Function):
    """
    A wrapper to add derivatives to the _Function object and to let it take in
    tuple objects and only use the first values for function evaluation.
    
    Leverages the gplearn.functions module to handle function 'closure'
    (handling bad inputs) and function arity (number of inputs).

    gplearn.functions documentation:
    https://gplearn.readthedocs.io/en/stable/_modules/gplearn/functions.html
    """

    def __init__(self, function, name, arity):
        _gp_Function.__init__(self, function, name, arity)
        self.derivative = _derivative_map[name]

    def __call__(self, *args):
        args0 = [a[0] for a in args]
        return self.function(*args0)


# choose the allowed functions
add2  = _Function(function=np.add, name='add', arity=2)
mul2  = _Function(function=np.multiply, name='mul', arity=2)
# sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1  = _Function(function=_protected_log, name='log', arity=1)
# inv1  = _Function(function=_protected_inverse, name='inv', arity=1)
# sin1  = _Function(function=np.sin, name='sin', arity=1)
# cos1  = _Function(function=np.cos, name='cos', arity=1)
# tan1  = _Function(function=np.tan, name='tan', arity=1)
# arctan1  = _Function(function=np.arctan, name='arctan', arity=1)
exp   = _Function(function=np.exp, name='exp', arity=1)
sig   = _Function(function=sigmoid, name='sig', arity=1)
softplus = _Function(function=splus, name='softplus', arity=1)

_function_map = {
    'add': add2,
    # 'mul': mul2,
    # 'sqrt': sqrt1,
    # 'log': log1,
    # 'inv': inv1,
    # 'sin': sin1,
    # 'cos': cos1,
    # 'tan': tan1,
    # 'arctan': arctan1,
    # 'exp': exp,
    # 'sig': sig,
    # 'softplus': softplus
}

_arities = {
    1: [
        # 'sqrt',
        # 'log',
        # 'inv',
        # 'sin',
        # 'cos',
        # 'tan',
        # 'arctan',
        # 'exp',
        # 'sig',
        # 'softplus',
    ],
    2: [
        'add',
        # 'mul'
    ],
}

_latex = {
    'add': '{} + {}',
    'mul': '({})*({})',
    'sqrt': 'sqrt({})',
    'log': 'log({})',
    'inv': '1/{}',
    'sin': 'sin({})',
    'cos': 'cos({})',
    'tan': 'tan({})',
    'arctan': 'arctan({})',
    'exp': 'exp({})',
    'sig': 'sig({})',
    'softplus': 'softplus({})'
}
