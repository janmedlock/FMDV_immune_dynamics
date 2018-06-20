#cython: boundscheck=False, wraparound=False
from collections import deque

import numpy
cimport numpy
from scipy.sparse._sparsetools import csr_matvecs


cdef inline void matvecs(A,
                         double[:, ::1] B,
                         double[:, ::1] C,
                         Py_ssize_t n):
    '''Compute the matrix multiplication `C += A @ B`, where
    `A` is a `scipy.sparse.csr_matrix()`,
    `B` and `C` are `numpy.array()`s,
    and all 3 matrices are `n` x `n`.'''
    # Use the private function
    # `scipy.sparse._sparsetools.csr_matvecs()` so we can specify
    # the output array `C` to avoid the building of a new matrix
    # for the output.
    csr_matvecs(n, n,  # The shape of A.
                n,     # The number of columns in B & C.
                A.indptr, A.indices, A.data,
                numpy.ravel(B), numpy.ravel(C))


cdef inline void do_births(double[:] b,
                           double[:, ::1] U,
                           double[:] v_trapezoid) nogil:
    '''Calculate the birth integral
    B(t) = \int_0^{inf} b(t, a) U(t, a) da
    using the composite trapezoid rule.
    The result is stored in `U[0]`, the first row of `U`,
    i.e. age 0.'''
    # The simple version is
    # `U[0] += (v_trapezoid * b) @ U`
    # but avoid building new vectors.
    cdef Py_ssize_t i, j
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[0, j] += b[i] * v_trapezoid[i] * U[i, j]


# Crank–Nicolson is 2nd order because the solution at t_n
# depends on the solution at t_{n - 1} and t_{n - 2}.
_order = 2


def mssolve(double[:] ages,
            double[:] t,
            M_crank_nicolson_2,
            M_crank_nicolson_1,
            M_implicit_euler,
            double[:] v_trapezoid,
            birth_rate):
    '''The core of the monodromy solver.'''
    cdef Py_ssize_t n_ages = ages.size
    # Set up solution.
    # `solution`, a length-3 sequence 3 with
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
    solution = deque(numpy.empty((n_ages, n_ages))
                     for _ in range(1 + _order))
    # `b` will store the birth rate.
    cdef numpy.ndarray[numpy.float_t, ndim=1] b = numpy.empty(n_ages)
    cdef double t_n
    ################################################
    ## Begin iteratively solving over time steps. ##
    ################################################
    if t.size == 0:
        return None
    # `len(t) > 0` is guaranteed below.
    ## n = 0 ##
    t_n = t[0]
    # The initial condition for the fundamental solution is the
    # identity matrix.
    solution[0][:] = 0
    numpy.fill_diagonal(solution[0], 1)
    if t.size == 1:
        return solution[0]
    # `len(t) > 1` is guaranteed below.
    ## n = 1 ##
    t_n = t[1]
    solution.rotate()
    # The simple version is
    # `solution[0] = M_implicit_euler @ solution[1]`
    # but avoid building a new matrix.
    solution[0][:] = 0
    # solution[0] += M_implicit_euler @ solution[1]
    matvecs(M_implicit_euler, solution[1], solution[0], n_ages)
    # Birth.
    birth_rate(t_n, ages, out=b)
    do_births(b, solution[0], v_trapezoid)
    ## n = 2, 3, ... ##
    for t_n in t[2 : ]:
        solution.rotate()
        # Aging & mortality.
        # The simple version is
        # `solution[0] = (M_crank_nicolson_2 @ solution[2]
        #                 + M_crank_nicolson_1 @ solution[1])`
        # but avoid building a new matrix.
        solution[0][:] = 0
        # solution[0] += M_crank_nicolson_2 @ solution[2]
        matvecs(M_crank_nicolson_2, solution[2], solution[0], n_ages)
        # solution[0] += M_crank_nicolson_1 @ solution[1]
        matvecs(M_crank_nicolson_1, solution[1], solution[0], n_ages)
        # Birth.
        birth_rate(t_n, ages, out=b)
        do_births(b, solution[0], v_trapezoid)
    # Return the solution at the final time.
    return solution[0]
