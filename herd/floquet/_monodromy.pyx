#cython: boundscheck=False, wraparound=False

'''The core of the solver.'''

from collections import deque

import numpy
cimport numpy


cdef extern from 'cblas.h':
    cdef void cblas_daxpy(int n, double alpha, double *x, int incx,
                          double *y, int incy) nogil


cdef inline void csr_matvecs(const int[::1] A_indptr,
                             const int[::1] A_indices,
                             const double[::1] A_data,
                             const double[:, ::1] B,
                             double[:, ::1] C) nogil:
    '''Compute the matrix multiplication `C += A @ B`, where
    `A` is an `n_row` x `n_col` `scipy.sparse.csr_matrix()`,
    `B` is an `n_col` x `n_vecs` `numpy.ndarray()`
    and `C` is an `n_row` x `n_vecs` `numpy.ndarray()`.'''
    cdef Py_ssize_t n_row, n_vecs
    # `C.shape` has a bunch of trailing `0`s.
    n_row, n_vecs = C.shape[: 2]
    cdef Py_ssize_t i, jj
    for i in range(n_row):
        for jj in range(A_indptr[i], A_indptr[i + 1]):
            # C[i, :] += A_data[jj] * B[A_indices[jj], :].
            # `1`s are strides of `B[j, :]` and `C[i, :]`,
            # which are enforced by `double[:, ::1]`
            # in the function arguments.
            cblas_daxpy(n_vecs,
                        A_data[jj],
                        &B[A_indices[jj], 0], 1,
                        &C[i, 0], 1)


cdef inline void _matvecs(A,
                          const double[:, ::1] B,
                          double[:, ::1] C):
    '''Compute the matrix multiplication `C += A @ B`, where
    `A` is a `scipy.sparse.csr_matrix()`,
    `B` and `C` are `numpy.ndarray()`s.'''
    # Extract the required Python attributes of `A`, which requires
    # the GIL, and then call the pure-C helper function without the
    # GIL.
    csr_matvecs(A.indptr, A.indices, A.data, B, C)


cdef inline void _do_births(const double[::1] v_trapezoid,
                            const double[::1] b,
                            double[:, ::1] U) nogil:
    '''Calculate the birth integral
    B(t) = \int_0^{inf} b(t, a) U(t, a) da
    using the composite trapezoid rule.
    The result is stored in `U[0]`, the first row of `U`,
    i.e. age 0.'''
    # The simple version is
    # `U[0] += (v_trapezoid * b) @ U`
    # but avoid building new vectors.
    cdef Py_ssize_t n_ages, i
    n_ages = U.shape[1]
    for i in range(U.shape[0]):
        # U[0, :] += b[i] * v_trapezoid[i] * U[i, :].
        # `1`s are strides of `U[i, :]` and `U[0, :]`,
        # which is enforced by `double[:, ::1]`
        # in the function arguments.
        cblas_daxpy(n_ages,
                    b[i] * v_trapezoid[i],
                    &U[i, 0], 1,
                    &U[0, 0], 1)


# Crank–Nicolson is 2nd order because the solution at t_n
# depends on the solution at t_{n - 1} and t_{n - 2}.
_order = 2


def solve(const double[::1] ages,
          const double[::1] t,
          M_crank_nicolson_2,
          M_crank_nicolson_1,
          M_implicit_euler,
          const double[::1] v_trapezoid,
          birth_rate):
    '''The core of the monodromy solver.'''
    cdef Py_ssize_t n_ages = ages.size
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
    solution = deque(numpy.empty((n_ages, n_ages))
                     for _ in range(1 + _order))
    # To avoid repeated array creation, `b_n` will store the birth
    # rate at each time step.
    cdef numpy.ndarray[numpy.float_t, ndim=1] b_n = numpy.empty(n_ages)
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
    _matvecs(M_implicit_euler, solution[1], solution[0])
    # Birth.
    birth_rate(t_n, ages, out=b_n)
    _do_births(v_trapezoid, b_n, solution[0])
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
        _matvecs(M_crank_nicolson_2, solution[2], solution[0])
        # solution[0] += M_crank_nicolson_1 @ solution[1]
        _matvecs(M_crank_nicolson_1, solution[1], solution[0])
        # Birth.
        birth_rate(t_n, ages, out=b_n)
        _do_births(v_trapezoid, b_n, solution[0])
    # Return the solution at the final time.
    return solution[0]
