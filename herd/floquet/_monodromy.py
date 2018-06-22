'''The core of the solver.  This is split out so it can be
re-implemented in Cython etc for speed.'''

from collections import deque

import numpy
from scipy.sparse._sparsetools import csr_matvecs


def _matvecs(A, B, C):
    '''Compute the matrix multiplication `C += A @ B`, where
    `A` is a `scipy.sparse.csr_matrix()`,
    `B` and `C` are `numpy.ndarray()`s.'''
    # Use the private function
    # `scipy.sparse._sparsetools.csr_matvecs()` so we can specify
    # the output array `C` to avoid the building of a new matrix
    # for the output.
    n_row, n_vecs = C.shape
    n_col, _ = B.shape
    csr_matvecs(n_row, n_col, n_vecs,
                A.indptr, A.indices, A.data,
                B.ravel(), C.ravel())


def _do_births(v_trapezoid, b, U):
    '''Calculate the birth integral
    B(t) = \int_0^{inf} b(t, a) U(t, a) da
    using the composite trapezoid rule.
    The result is stored in `U[0]`, the first row of `U`,
    i.e. age 0.'''
    # The simple version is
    # `U[0] = (v_trapezoid * b) @ U`
    # but avoid building new vectors.
    b *= v_trapezoid
    # This is slightly faster than `numpy.dot(b, U, out=U[0])`
    b.dot(U, out=U[0])


# Crank–Nicolson is 2nd order because the solution at t_n
# depends on the solution at t_{n - 1} and t_{n - 2}.
_order = 2


def solve(ages, t, M_crank_nicolson_2, M_crank_nicolson_1,
          M_implicit_euler, v_trapezoid, birth_rate):
    n_ages = ages.size
    # Set up solution.
    # `solution` is a length-3 sequence 3 with
    # `solution[0]` storing the solution at the current time step,
    # `solution[1]` storing the solution at the previous time step, and
    # `solution[2]` storing the solution 2 time steps ago.
    # Calling `solution.rotate()` in sync with iterating through
    # the elements of `t` rearranges the elements of
    # `solution` so that its elements stay in the above order
    # at each time step:
    # the old `solution[0]` becomes the new `solution[1]`;
    # the old `solution[1]` becomes the new `solution[2]`; and
    # the old `solution[2]` becomes the new `solution[0]`,
    # recycled and ready to be set to the value of the solution at the
    # new time step.
    # The fundamental solution is an `n_ages` x `n_ages` matrix.
    # One matrix for the current time step,
    # plus one for each order of the solver.
    # The fundamental solution is an `n_ages` x `n_ages` matrix.
    solution = deque(numpy.empty((n_ages, n_ages))
                     for _ in range(1 + _order))
    # To avoid repeated array creation, `b_n` will store the birth
    # rate at each time step.
    b_n = numpy.empty(n_ages)
    ################################################
    ## Begin iteratively solving over time steps. ##
    ################################################
    if t.size == 0:
        return None
    # `t.size > 0` is guaranteed below.
    ## n = 0 ##
    t_n = t[0]
    # The initial condition for the fundamental solution is the
    # identity matrix.
    solution[0][:] = 0
    numpy.fill_diagonal(solution[0], 1)
    if t.size == 1:
        return solution[0]
    # `t.size > 1` is guaranteed below.
    ## n = 1 ##
    t_n = t[1]
    solution.rotate()
    # The simple version is
    # `solution[0][:] = M_implicit_euler @ solution[1]`
    # but avoid building a new matrix.
    solution[0][:] = 0
    # solution[0] += M_implicit_euler @ solution[1]
    _matvecs(M_implicit_euler, solution[1], solution[0])
    # Birth.
    birth_rate(t_n, ages, out=b_n)
    _do_births(v_trapezoid, b_n, solution[0])
    ## n = 2, 3, ... ##
    for t_n in t[2 : ]:
        solution.rotate()
        # Aging & mortality.
        # The simple version is
        # `solution[0][:] = (M_crank_nicolson_2 @ solution[2]
        #                    + M_crank_nicolson_1 @ solution[1])`
        # but avoid building a new matrix.
        solution[0][:] = 0
        # solution[0] += M_crank_nicolson_2 @ solution[2]
        _matvecs(M_crank_nicolson_2, solution[2], solution[0])
        # solution[0] += M_crank_nicolson_1 @ solution[1]
        _matvecs(M_crank_nicolson_1, solution[1], solution[0])
        # Birth.
        birth_rate(t_n, ages, out=b_n)
        _do_births(v_trapezoid, b_n, solution[0])
    # Return the solution at the final time.
    return solution[0]
