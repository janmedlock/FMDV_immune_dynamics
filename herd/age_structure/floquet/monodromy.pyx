#cython: language_level=3, boundscheck=False, wraparound=False

'''The McKendrick–von Foerster age-structured PDE model
for the density u(t, a) of buffalo of age a at time t is
(d/dt + d/da) u(t, a) = - d(a) u(t, a),
u(t, 0) = \int_0^{inf} b(t, a) u(t, a) da
u(0, a) = u_0(a).
The birth rate `b(t, a)`, which is the product of
`female_probability_at_birth` and `birth.gen.hazard(t, a)` in our
model, is periodic in `t` with period T = 1 year.

After discretizing in age, the monodromy matrix Phi(T) is the
fundamental solution of the PDEs after one period, starting from the
initial condition
Phi(0) = I,
the identity matrix.  The fundamental solution requires solving the
matrix-valued version of the orignal PDE.

Here, the matrix-valued PDE is solved using the Crank–Nicolson method
on characteristics, following the work of Fabio Milner.

So that the cache keys only depend on the relevant parts of
`herd.parameters.Parameters()` and so that the setup can be reused in
`_find_birth_scaling()`, the solver is called in 3 steps:
>>> solver_parameters = monodromy.Parameters(parameters)
>>> solver = monodromy.Solver(solver_parameters, agemax, agestep)
>>> PhiT = solver.solve(birth_scaling)
where `parameters` is a `herd.parameters.Parameters()` instance.'''

cimport cython
cimport numpy

import numpy
from scipy import sparse

from herd import birth, mortality, parameters, periods, utility


# Functions from BLAS.
cdef extern from '<cblas.h>':
    # y[:] += alpha * x[:] + y[:].
    cdef void cblas_daxpy(const int n,
                          const double alpha,
                          const double *x,
                          const int x_stride,
                          double *y,
                          const int y_stride) nogil


cdef class _CSR_Matrix:
    '''`scipy.sparse.csr_matrix()` but with attributes
    in C for speed.'''
    cdef:
        ssize_t[2] shape
        int[:] indptr
        int[:] indices
        double[:] data

    def __cinit__(_CSR_Matrix self,
                  object A):
        A = sparse.csr_matrix(A)
        self.shape = A.shape
        self.indptr = A.indptr
        self.indices = A.indices
        self.data = A.data

    cdef inline bint matvecs(_CSR_Matrix self,
                             const double[:, ::1] B,
                             double[:, ::1] C) nogil except False:
        '''Compute the matrix multiplication `C += A @ B`, where
        `A` is a `_CSR_Matrix()`, and
        `B` & `C` are `numpy.ndarray()`s.'''
        cdef:
            ssize_t n_vecs, i, jj
        n_vecs = B.shape[1]
        for i in range(self.shape[0]):
            # Loop over the non-zero entries in row[i],
            # A[i, A.indices[jj]] = A.data[jj].
            for jj in range(self.indptr[i], self.indptr[i + 1]):
                # C[i, :] += A.data[jj] * B[A.indices[jj], :].
                cblas_daxpy(n_vecs,
                            self.data[jj],
                            &B[self.indices[jj], 0], 1,
                            &C[i, 0], 1)
        return True


cdef class Parameters:
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''
    cdef:
        double birth_normalized_peak_time_of_year
        double birth_seasonal_coefficient_of_variation
        double female_probability_at_birth

    # Don't use `__cinit__()` because these need to be pickled.
    def __init__(Parameters self,
                 object params):
        # Normalize `params.birth_peak_time_of_year` by making
        # it relative to `params.start_time` and then modulo
        # `period` so that it is in [0, period).
        self.birth_normalized_peak_time_of_year = (
            (params.birth_peak_time_of_year - params.start_time)
            % periods.get_period())
        self.birth_seasonal_coefficient_of_variation = (
            params.birth_seasonal_coefficient_of_variation)
        self.female_probability_at_birth = (
            params.female_probability_at_birth)


# The fundamental solution at time t_n is 2 dimensional.
ctypedef double[:, ::1] _Solution_n
# For `_Solution()`, add one more dimension for time.
ctypedef double[:, :, ::1] _Solution_array


cdef class _Solution:
    '''A `_Solution()` is a sequence of length `1 + order`, with
    `solution[0]` storing the solution at the current time step,
    `solution[1]` storing the solution 1 time steps ago,
    ...
    `solution[order]` storing the solution `order` time steps ago.
    Calling `_Solution.update()` in sync with iterating through
    rearranges the elements of `solution` so that its elements stay in
    the above order at the next time step:
    the old `solution[0]` becomes the new `solution[1]`;
    ...
    the old `solution[order - 1]` becomes the new `solution[order]`;
    the old `solution[order]` becomes the new `solution[0]`,
    recycled and ready to be set to the value of the solution at the
    new time step.'''
    cdef:
        _Solution_array _array
        ssize_t _front

    def __cinit__(_Solution self,
                  ssize_t order,
                  tuple shape):
        self._array = numpy.empty((1 + order, ) + shape)
        # '_front' points to the `0` element in the current iteration.
        self._front = 0

    cdef inline _Solution_n get(_Solution self,
                                ssize_t i) nogil:
        '''Get the solution `i` time steps ago.'''
        # This is not `__getitem__()` so the GIL can be released.
        cdef:
            ssize_t j
        j = (i + self._front) % self._array.shape[0]
        return self._array[j]

    cdef inline bint update(_Solution self) nogil except False:
        '''Move the entries forward one position,
        wrapping the last entry to the front.'''
        # Decrement `_front` by 1 and wrap around if it's negative.
        self._front = (self._front - 1) % self._array.shape[0]
        return True


cdef class Solver:
    '''Solve the monodromy problem.'''
    # Crank–Nicolson is 1st order because the solution at t_n
    # depends on the solution at t_{n - 1}.
    _order = <ssize_t> 1

    cdef:
        public numpy.ndarray ages
        Parameters params
        double[:] _ages, _t, _v_trapezoid
        _CSR_Matrix _M_crank_nicolson

    def __cinit__(Solver self,
                  Parameters solver_params,
                  const double agemax,
                  const double agestep):
        cdef:
            double tstep
            object mortalityRV
        self.params = solver_params
        self.ages = utility.arange(0, agemax, agestep, endpoint=True)
        # The memoryview will be convenient...
        self._ages = self.ages
        tstep = agestep
        self._t = utility.arange(0, periods.get_period(), tstep, endpoint=True)
        mortalityRV = mortality.from_param_values()
        self._init_crank_nicolson(tstep, mortalityRV)
        self._init_births(agestep)

    @classmethod
    def from_parameters(type cls,
                        object params,
                        const double agemax,
                        const double agestep):
        '''Build a `Solver()` instance using `herd.parameters.Parameters()`
        directly.'''
        cdef:
            Parameters solver_params
        solver_params = Parameters(params)
        return cls(solver_params, agemax, agestep)

    cdef inline bint _set_initial_condition(Solver self,
                                            const double t_n,
                                            _Solution solution,
                                            object birth_rate,
                                            numpy.ndarray temp) \
                                           nogil except False:
        '''The initial condition for the fundamental solution is the
        identity matrix.'''
        cdef:
            _Solution_n solution0
            ssize_t i
        # Avoid creating a new matrix, like `numpy.eye()` does.
        solution0 = solution.get(0)
        solution0[:] = 0
        for i in range(self._ages.shape[0]):
            solution0[i, i] = 1
        return True

    @cython.wraparound(True)
    cdef inline bint _init_births(Solver self,
                                  const double agestep) nogil except False:
        '''The trapezoid rule for the birth integral for i = 0,
        u_0^n = \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) * da / 2.
        This can be written as
        u_0^n = (v * b^n) @ u^n,
        with
        v = da * [0.5, 1, 1, ..., 1, 1, 0.5].
        Put `female_probability_at_birth` in there, too, for
        simplicity & efficiency.'''
        with gil:
            # `numpy.empty()` requires the GIL.
            self._v_trapezoid = numpy.empty(self._ages.shape[0])
        self._v_trapezoid[:] = (agestep
                                * self.params.female_probability_at_birth)
        self._v_trapezoid[0] /= 2
        self._v_trapezoid[-1] /= 2
        return True

    cdef inline bint _step_births(Solver self,
                                  const double t_n,
                                  _Solution solution,
                                  object birth_rate,
                                  numpy.ndarray temp) nogil except False:
        '''Calculate the birth integral
        B(t) = \int_0^{inf} b(t, a) U(t, a) da
        using the composite trapezoid rule,
        where U = `solution[0]`.'''
        cdef:
            # The memoryview doesn't need the GIL.
            double[:] temp_view
            _Solution_n solution0
            ssize_t n_ages, n_col, i
        # The simple version is
        # `U[0] += (v_trapezoid * birth_rate) @ U`
        # but avoid building new vectors.
        with gil:
            # `birth_rate()` is a Python function, so it needs the GIL.
            # `out` must be a `numpy.ndarray()`.
            birth_rate(t_n, self.ages, out=temp)
            # Accessing the `numpy.ndarray()` also needs the GIL.
            temp_view = temp
        solution0 = solution.get(0)
        n_ages, n_col = solution0.shape[:2]
        for i in range(n_ages):
            # U[0] += (v_trapezoid[i] * b[i]) * U[i].
            cblas_daxpy(n_col,
                        self._v_trapezoid[i] * temp_view[i],
                        &solution0[i, 0], 1,
                        &solution0[0, 0], 1)
        return True

    @cython.wraparound(True)
    cdef inline bint _init_crank_nicolson(Solver self,
                                          const double tstep,
                                          object mortalityRV) except False:
        '''The Crank–Nicolson method is
        (u_i^n - u_{i - 1}^{n - 1}) / dt
        = - d_{i - 1 / 2} * (u_i^n + u_{i - 1}^{n - 1}) / 2,
        for i = 1, 2, ....
        This can be written as
        u^n = M @ u^{n - 1},
        with
        M[i, i - 1] = (1 - k_i) / (1 + k_i),
        k_i = d_{i - 1 / 2} * dt / 2,
        and, to prevent the last age group from aging out of the population,
        M[-1, -1] = (1 - k_{-1}) / (1 + k_{-1}),
        k_{-1} = d_{-1} * dt / 2.'''
        cdef:
            object M
            numpy.ndarray ages_mid, k
            double k_last
        # Everything needs the GIL.
        M = sparse.lil_matrix((self._ages.shape[0], self._ages.shape[0]))
        # Midpoints between adjacent ages.
        ages_mid = (self.ages[1:] + self.ages[:-1]) / 2
        k = mortalityRV.hazard(ages_mid) * tstep / 2
        # Set the first subdiagonal.
        M.setdiag((1 - k) / (1 + k), -1)
        # Keep the last age group from ageing out.
        k_last = mortalityRV.hazard(self.ages[-1]) * tstep / 2
        M[-1, -1] = (1 - k_last) / (1 + k_last)
        self._M_crank_nicolson = _CSR_Matrix(M)
        return True

    cdef inline bint _step_crank_nicolson(Solver self,
                                          const double t_n,
                                          _Solution solution,
                                          object birth_rate,
                                          numpy.ndarray temp) \
                                         nogil except False:
        '''Do a Crank–Nicolson step.'''
        cdef:
            _Solution_n solution0, solution1
        # The simple version is
        # `solution[0] = M @ solution[1]`
        # but avoid building a new matrix.
        solution0 = solution.get(0)
        solution1 = solution.get(1)
        solution0[:] = 0
        # solution[0] += M @ solution[1]
        self._M_crank_nicolson.matvecs(solution1, solution0)
        self._step_births(t_n, solution, birth_rate, temp)
        return True

    cdef inline _Solution_n _iterate(Solver self,
                                     _Solution solution,
                                     object birth_rate,
                                     numpy.ndarray temp) nogil:
        '''The core of the solver, iterating over `self._t`.'''
        cdef:
            ssize_t n
        if self._t.shape[0] == 0: return None
        ## n = 0 ##
        n = 0
        self._set_initial_condition(self._t[n], solution, birth_rate, temp)
        if self._t.shape[0] == 1: return solution.get(0)
        ## n = 1, 2, ... ##
        # Looping over `self._t[1:]` requires the GIL.
        for n in range(1, self._t.shape[0]):
            solution.update()
            self._step_crank_nicolson(self._t[n], solution, birth_rate, temp)
        # Return the solution at the final time.
        return solution.get(0)

    def solve(Solver self,
              double birth_scaling):
        '''Find the monodromy matrix Phi(T), where T is the period.'''
        cdef:
            _Solution solution
            object birthRV
            numpy.ndarray temp
        # The fundamental solution is an `n_ages` x `n_ages` matrix.
        # One matrix for the current time step, plus one for each
        # order of the solver.
        solution = _Solution(self._order,
                             (self._ages.shape[0], self._ages.shape[0]))
        # Set up birth rate.
        birthRV = birth.from_param_values(
            self.params.birth_normalized_peak_time_of_year,
            self.params.birth_seasonal_coefficient_of_variation,
            _scaling=birth_scaling)
        temp = numpy.empty(self._ages.shape[0])
        return numpy.asarray(self._iterate(solution, birthRV.hazard, temp))
