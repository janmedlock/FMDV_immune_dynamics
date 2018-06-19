#cython: boundscheck=False, wraparound=False
cimport cython

import numpy
cimport numpy

from scipy.sparse._sparsetools import csr_matvecs


cdef bint matvecs(A,
                  numpy.ndarray[numpy.float_t, ndim=2] B,
                  numpy.ndarray[numpy.float_t, ndim=2] C,
                  Py_ssize_t n) except False:
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
                B.ravel(), C.ravel())
    return True


cdef bint do_births(numpy.ndarray[numpy.float_t, ndim=1] b,
                    numpy.ndarray[numpy.float_t, ndim=2] U,
                    numpy.ndarray[numpy.float_t, ndim=1] v_trapezoid) \
                    except False:
    '''Calculate the birth integral
    B(t) = \int_0^{inf} b(t, a) U(t, a) da
    using the composite trapezoid rule.
    The result is stored in `U[0]`, the first row of `U`,
    i.e. age 0.'''
    # The simple version is
    # `U[0] = (v_trapezoid * b) @ U`
    # but avoid building new vectors.
    b *= v_trapezoid
    b.dot(U, out=U[0])
    return True


@cython.wraparound(True)
cdef inline bint update(list loc) except False:
    '''Shift the values to one to the right, wrapping the last value to
    the front.'''
    loc[:] = loc[-1: ] + loc[: -1]
    return True


def mssolve(numpy.ndarray[numpy.float_t, ndim=1] ages,
            numpy.ndarray[numpy.float_t, ndim=1] t,
            M_crank_nicolson_2,
            M_crank_nicolson_1,
            M_implicit_euler,
            numpy.ndarray[numpy.float_t, ndim=1] v_trapezoid,
            birth_rate):
    '''The core of the monodromy solver.'''
    cdef Py_ssize_t n_ages = len(ages)
    cdef list loc = list(range(3))
    # `solution` stores the solution at times
    # t_n, t_{n - 1}, and t_{n - 2} in
    # `solution[loc[0]]`, `solution[loc[1]]`, and `solution[loc[2]]`,
    # respectively.
    cdef numpy.ndarray[numpy.float_t, ndim=3] solution
    solution = numpy.empty((len(loc), n_ages, n_ages))
    cdef numpy.ndarray[numpy.float_t, ndim=1] b
    ###########
    ## n = 0 ##
    ###########
    # The initial condition for the fundamental solution is the
    # identity matrix.
    solution[loc[0]] = 0
    solution[loc[0]][numpy.diag_indices(n_ages)] = 1
    if len(t) <= 1:
        return solution[loc[0]]
    # `len(t) > 1` is guaranteed below.
    ###########
    ## n = 1 ##
    ###########
    update(loc)
    # The simple version is
    # `solution[loc[0]][:] = M_implicit_euler @ solution[loc[1]]`
    # but avoid building a new matrix.
    solution[loc[0]][:] = 0
    # solution[loc[0]] += M_implicit_euler @ solution[loc[1]]
    matvecs(M_implicit_euler, solution[loc[1]], solution[loc[0]], n_ages)
    # Birth.
    b = birth_rate(t[1], ages)
    do_births(b, solution[loc[0]], v_trapezoid)
    ###################
    ## n = 2, 3, ... ##
    ###################
    cdef double t_n
    for t_n in t[2 : ]:
        update(loc)
        # Aging & mortality.
        # The simple version is
        # `solution[loc[0]][:] = (M_crank_nicolson_2 @ solution[loc[2]]
        #                         + M_crank_nicolson_1 @ solution[loc[1]])`
        # but avoid building a new matrix.
        solution[loc[0]][:] = 0
        # solution[loc[0]] += M_crank_nicolson_2 @ solution[loc[2]]
        matvecs(M_crank_nicolson_2, solution[loc[2]], solution[loc[0]],
                n_ages)
        # solution[loc[0]] += M_crank_nicolson_1 @ solution[loc[1]]
        matvecs(M_crank_nicolson_1, solution[loc[1]], solution[loc[0]],
                n_ages)
        # Birth.
        b = birth_rate(t_n, ages)
        do_births(b, solution[loc[0]], v_trapezoid)
    # Return the solution at the final time.
    return solution[loc[0]]
