from collections import deque
import os.path

from joblib import Memory
import numpy
from scipy.integrate import trapz
from scipy.optimize import brentq
from scipy.sparse import lil_matrix
from scipy.sparse._sparsetools import csr_matvecs

from . import utility
from . import dominant_eigen
from . import birth
from . import mortality
from . import parameters


# Some of the functions below are very slow, so the values are cached to
# disk with `joblib.Memory()` so that they are only computed once.
_cachedir = os.path.join(os.path.dirname(__file__), '__cache__')
_cache = Memory(_cachedir, verbose=1)


class _MonodromySolver:
    '''Find the monodromy matrix Phi(period) by solving
    (d/dt + d/da) Phi = - d(a) Phi,
    Phi(t, 0) = \int_0^{inf} b(t, a) Phi(t, a) da
    Phi(0, a) = I.
    The PDE is solved using the Crank–Nicolson method on characteristics.
    So that the cache keys only depend on the relevant parts of
    `herd.parameters.Parameters()` and so that the setup can be reused in
    `_find_birth_scaling()`, the solver is called in 3 steps:
    >>> msparameters = _MonodromySolver.Parameters(parameters)
    >>> solver = _MonodromySolver(msparameters, agemax, agestep)
    >>> PhiT = solver.solve(birth_scaling)
    where `parameters` is a `herd.parameters.Parameters()` instance.'''
    # Crank–Nicolson is 2nd order because the solution at t_n
    # depends on the solution at t_{n - 1} and t_{n - 2}.
    order = 2
    # `herd.birth.gen.hazard` is the only time-dependent function.
    period = herd.birth._period

    class Parameters(parameters.Parameters):
        '''Build a `herd.parameters.Parameters()`-like object that
        only has the parameters needed by `_MonodromySolver()`
        so that it can be efficiently cached.'''
        def __init__(self, parameters):
            # Generally, the values of these parameters should be
            # floats, so explicitly convert them so the cache doesn't
            # get duplicated keys for the float and int representation
            # of the same number, e.g. `float(0)` and `int(0)`.
            # Normalize `parameters.birth_peak_time_of_year` by making
            # it relative to `parameters.start_time` and then modulo
            # `period` so that it is in [0, period).
            self.birth_normalized_peak_time_of_year = float(
                (parameters.birth_peak_time_of_year - parameters.start_time)
                % _MonodromySolver.period)
            self.birth_seasonal_coefficient_of_variation = float(
                parameters.birth_seasonal_coefficient_of_variation)
            self.female_probability_at_birth = float(
                parameters.female_probability_at_birth)

    def __init__(self, msparameters, agemax, agestep):
        self.parameters = msparameters
        self.ages = utility.arange(0, agemax, agestep, endpoint=True)
        n_ages = len(self.ages)
        tstep = agestep
        self.t = utility.arange(0, self.period, tstep, endpoint=True)
        # Set up mortality rate.
        mortalityRV = mortality.from_param_values()
        mortality_rate = mortalityRV.hazard
        # The Crank–Nicolson method is
        # (u_i^n - u_{i - 2}^{n - 2}) / 2 / dt
        # = - d_{i - 1} * (u_i^n + u_{i - 2}^{n - 2}) / 2,
        # for i = 2, 3, ...,
        # with implicit Euler for i = 1,
        # (u_1^n - u_0^{n - 1}) / dt = - d_1 * u_1^n.
        # This can be written as
        # u^n = M_2 @ u^{n - 2} + M_1 @ u^{n - 1},
        M_crank_nicolson_2 = lil_matrix((n_ages, n_ages))
        M_crank_nicolson_1 = lil_matrix((n_ages, n_ages))
        # with
        # M_2[i, i - 2] = (1 - dt * d_{i - 1}) / (1 + dt * d_{i - 1});
        diag2 = ((1 - tstep * mortality_rate(self.ages))
                 / (1 + tstep * mortality_rate(self.ages)))
        M_crank_nicolson_2.setdiag(diag2[1 : -1], -2)
        # to prevent the last age group from aging out of the population,
        # M_2[-1, -1] = (1 - dt * d_{-1}) / (1 + dt * d_{-1});
        M_crank_nicolson_2[-1, -1] = diag2[-1]
        # M_1[1, 0] = 1 / (1 + dt * d_i);
        diag1 = 1 / (1 + tstep * mortality_rate(self.ages))
        M_crank_nicolson_1[1, 0] = diag1[1]
        # and, to prevent the next to last age group from aging out,
        # M_1[-1, -2] = 1 / (1 + dt * d_{-1}).
        M_crank_nicolson_1[-1, -2] = diag1[-1]
        # Convert to CSR for fast left multiplication.
        self.M_crank_nicolson_2 = M_crank_nicolson_2.tocsr()
        self.M_crank_nicolson_1 = M_crank_nicolson_1.tocsr()
        # The first time step, n = 1, uses implicit Euler:
        # (u_i^n - u_{i - 1}^{n - 1}) / dt = - d_i * u_i^n.
        # This can be written as
        # u^1 = M @ u^0,
        M_implicit_euler = lil_matrix((n_ages, n_ages))
        # with
        # M[i, i - 1] = 1 / (1 + dt * d_i),
        M_implicit_euler.setdiag(diag1[1 : ], -1)
        # and, to prevent the last age group from aging out of the population,
        # M[-1, -1] = 1 / (1 + dt * d_{-1}).
        M_implicit_euler[-1, -1] = diag1[-1]
        # Convert to CSR for fast left multiplication.
        self.M_implicit_euler = M_implicit_euler.tocsr()
        # The trapezoid rule for the birth integral for i = 0,
        # u_0^n = \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) * da / 2.
        # This can be written as
        # u_0^n = (v * b^n) @ u^n,
        # with
        # v = da * [0.5, 1, 1, ..., 1, 1, 0.5].
        # Put `female_probability_at_birth` in there, too, for
        # simplicity & efficiency.
        self.v_trapezoid = (self.parameters.female_probability_at_birth
                            * agestep
                            * numpy.hstack((0.5, numpy.ones(n_ages - 2), 0.5)))

    @classmethod
    def SolutionCycle(cls, size):
        '''The generator returned yields sequences that store
        the solution at times t_n, t_{n - 1}, ..., t_{n - order}.
        The generator cycles so that the solution at
        the current time step is in `solution[0]`,
        the previous time step is in `solution[1]`, ...
        `order` time steps ago is in `solution[order]`.
        This is effectively just moving references around in a cycle,
        so no new arrays get built as the solver iterates in time.'''
        # One array for the current time step, plus one for each order
        # of the solver.
        solution = deque(numpy.empty(size, dtype=float)
                         for _ in range(1 + cls.order))
        while True:
            yield solution
            solution.rotate()

    @staticmethod
    def matvecs(A, B, C, n):
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

    @staticmethod
    def do_births(b, U, v_trapezoid):
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

    def solve(self, birth_scaling):
        '''Find the monodromy matrix Phi(T), where T is the period.'''
        # Set up birth rate.
        birthRV = birth.from_param_values(
            self.parameters.birth_normalized_peak_time_of_year,
            self.parameters.birth_seasonal_coefficient_of_variation,
            _scaling=birth_scaling)
        birth_rate = birthRV.hazard
        # Avoid lookups in the loop below.
        ages = self.ages
        n_ages = len(ages)
        t = self.t
        M_crank_nicolson_2 = self.M_crank_nicolson_2
        M_crank_nicolson_1 = self.M_crank_nicolson_1
        M_implicit_euler = self.M_implicit_euler
        v_trapezoid = self.v_trapezoid
        matvecs = self.matvecs
        do_births = self.do_births
        # Set up solution.
        # `solution_cycle()` returns a generator that yields
        # `solution`, a length-3 sequence 3 with
        # `solution[0]` storing the solution at the current time step,
        # `solution[1]` storing the solution at the previous time step, and
        # `solution[2]` storing the solution 2 time steps ago.
        # Iterating `solution_cycle()` in sync with iterating through
        # the elements of `t` rearranges the elements of the yielded
        # `solution` so that its elements stay in the above order
        # at each time step:
        # the old `solution[0]` becomes the new `solution[1]`;
        # the old `solution[1]` becomes the new `solution[2]`; and
        # the old `solution[2]` is recycled to `solution[0]`,
        # ready to be set to the value of the solution at the new time step.
        # The fundamental solution is a `n_ages` x `n_ages` matrix.
        solution_cycle = self.SolutionCycle((n_ages, n_ages))
        ###########
        ## n = 0 ##
        ###########
        (t_n, solution) = (t[0], next(solution_cycle))
        # The initial condition for the fundamental solution is the
        # identity matrix.
        solution[0][:] = 0
        numpy.fill_diagonal(solution[0], 1)
        if len(t) <= 1:
            return solution[0]
        # `len(t) > 1` is guaranteed below.
        ###########
        ## n = 1 ##
        ###########
        (t_n, solution) = (t[1], next(solution_cycle))
        # The simple version is
        # `solution[0][:] = M_implicit_euler @ solution[1]`
        # but avoid building a new matrix.
        solution[0][:] = 0
        # solution[0] += M_implicit_euler @ solution[1]
        matvecs(M_implicit_euler, solution[1], solution[0], n_ages)
        # Birth.
        do_births(birth_rate(t_n, ages), solution[0], v_trapezoid)
        ###################
        ## n = 2, 3, ... ##
        ###################
        for (t_n, solution) in zip(t[2 : ], solution_cycle):
            # Aging & mortality.
            # The simple version is
            # `solution[0][:] = (M_crank_nicolson_2 @ solution[2]
            #                    + M_crank_nicolson_1 @ solution[1])`
            # but avoid building a new matrix.
            solution[0][:] = 0
            # solution[0] += M_crank_nicolson_2 @ solution[2]
            matvecs(M_crank_nicolson_2, solution[2], solution[0], n_ages)
            # solution[0] += M_crank_nicolson_1 @ solution[1]
            matvecs(M_crank_nicolson_1, solution[1], solution[0], n_ages)
            # Birth.
            do_births(birth_rate(t_n, ages), solution[0], v_trapezoid)
        # Return the solution at the final time.
        return solution[0]


def _normalize_to_density(v, ages):
    '''Normalize `v` in place so that its integral over ages is 1.'''
    v /= trapz(v, ages)


# This function is very slow because it calls
# `_MonodromySolver.solve()`, which is very slow, and
# `dominant_eigen.find()`, which is somewhat slow, so it is cached.
# Caching also allows the eigenvector to be retrieved from the cache
# by `find_stable_birth_structure()` after the eigenvalue is computed
# in `find_birth_scaling()`.
@_cache.cache(ignore=['solver'], verbose=0)
def _find_dominant_eigen(birth_scaling, msparameters, agemax, agestep,
                         solver=None):
    '''Find the dominant Floquet exponent (the one with the largest real part)
    and its corresponding eigenvector.'''
    if solver is None:
        solver = _MonodromySolver(msparameters, agemax, agestep)
    PhiT = solver.solve(birth_scaling)
    # Finding the matrix B = log(Phi(T)) / T is very expensive,
    # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
    # and convert.
    rho0, v0 = dominant_eigen.find(PhiT, which='LM')
    # rho0 is the dominant (largest magnitude) Floquet multiplier.
    # mu0 is the dominant (largest real part) Floquet exponent.
    # They are related by rho0 = exp(mu0 * T).
    mu0 = numpy.log(rho0) / solver.period
    # v0 is the eigenvector for both rho0 and mu0.
    _normalize_to_density(v0, solver.ages)
    return (mu0, v0, solver.ages)


def _find_growth_rate(birth_scaling, msparameters, agemax, agestep, solver):
    '''Find the population growth rate.'''
    mu0, _, _ = _find_dominant_eigen(birth_scaling, msparameters,
                                     agemax, agestep, solver)
    return mu0


# This function is extremely slow because, through
# `_find_growth_rate()` and `scipy.optimize.brentq()`, it repeatedly
# calls `_find_dominant_eigen()`, which is very slow, so it is cached.
@_cache.cache
def _find_birth_scaling(msparameters, agemax, agestep):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Reuse the solver to avoid multiple setup/teardown.
    solver = _MonodromySolver(msparameters, agemax, agestep)
    args = (msparameters, agemax, agestep, solver)
    a = 0
    # We know that at the lower limit a = 0,
    # `_find_growth_rate(0, ...) < 0`,
    # so we need to find an upper limit `b`
    # with `_find_growth_rate(b, ...) >= 0`.
    b = 1
    while _find_growth_rate(b, *args) < 0:
        a = b
        b *= 2
    return brentq(_find_growth_rate, a, b, args=args)


_agemax_default = 35
_agestep_default = 0.01


def find_birth_scaling(parameters,
                       agemax=_agemax_default,
                       agestep=_agestep_default):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Call the cached version.
    msparameters = _MonodromySolver.Parameters(parameters)
    return _find_birth_scaling(msparameters, agemax, agestep)


def find_stable_age_structure(parameters,
                              agemax=_agemax_default,
                              agestep=_agestep_default):
    '''Find the stable age structure.'''
    msparameters = _MonodromySolver.Parameters(parameters)
    birth_scaling = _find_birth_scaling(msparameters, agemax, agestep)
    r, v, ages = _find_dominant_eigen(birth_scaling, msparameters,
                                      agemax, agestep)
    assert numpy.isclose(r, 0), 'Nonzero growth rate r={:g}.'.format(r)
    return (v, ages)
