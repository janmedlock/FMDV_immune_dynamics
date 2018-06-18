from collections import deque
import os.path

import joblib
import numpy
from scipy import integrate, optimize, sparse

from . import birth
from . import mortality
from . import parameters
from . import dominant_eigen


# Some of the functions below are very slow, so the values are cached to
# disk with `joblib.Memory()` so they are only computed once.
_cachedir = os.path.join(os.path.dirname(__file__), '__cache__')
_cache = joblib.Memory(_cachedir, verbose=1)


def _arange(start, stop, step, endpoint=False):
    '''`numpy.arange()` that can optionally include
    the right endpoint `stop`.'''
    val = numpy.arange(start, stop, step)
    if endpoint:
        if not numpy.isclose(val[-1], stop):
            val = numpy.hstack((val, stop))
    return val


def _normalize(v, ages):
    '''Normalize v so that it integrates to 1.'''
    return v / integrate.trapz(v, ages)


def solution_cycle(N, size):
    '''The generator returned yields sequences that store the solution at
    times t_n, t_{n - 1}, t_{n - 2}, ..., t_{n - N + 1}.  The
    generator cycles so that the solution at the current time step is
    in `solution[0]`, the previous time step is in `solution[1]`, 2
    time steps ago is in `solution[2]`, and so on, up to `N`.  Each
    `solution[n]` has size `size`.'''
    solution = deque(numpy.empty(size) for _ in range(N))
    while True:
        yield solution
        solution.rotate()


class _MonodromySolver:
    '''Find the monodromy matrix Phi(period) by solving
    (d/dt + d/da) Phi = - d(a) Phi,
    Phi(t, 0) = \int_0^{inf} b(t, a) Phi(t, a) da
    Phi(0, a) = I.
    The PDE is solved using the Crank–Nicolson method on characteristics.'''
    class Parameters(parameters.Parameters):
        '''Convert `herd.parameters.Parameters()` object `parameters` to
        the arguments needed by `_MonodromySolver()`.
        This two-step process, `_MonodromySolver.Parameters()`
        then `_MonodromySolver()`, sets the keys for caching.'''
        period = 1

        def __init__(self, parameters):
            # Relative to `parameters.start_time`.
            self.birth_peak_time_of_year = ((parameters.birth_peak_time_of_year
                                             - parameters.start_time)
                                            % self.period)
            self.birth_seasonal_coefficient_of_variation \
                = parameters.birth_seasonal_coefficient_of_variation
            self.female_probability_at_birth \
                = parameters.female_probability_at_birth

    def __init__(self, msparameters, agemax, agestep):
        self.parameters = msparameters
        self.ages = _arange(0, agemax, agestep, endpoint=True)
        tstep = agestep
        self.t = _arange(0, self.parameters.period, tstep, endpoint=True)
        mortalityRV = mortality._from_param_values()
        mortality_rate = mortalityRV.hazard
        # The Crank–Nicolson method is
        # (u_i^n - u_{i - 2}^{n - 2}) / 2 / dt
        # = - d_{i - 1} * (u_i^n + u_{i - 2}^{n - 2}) / 2,
        # for i = 2, 3, ...,
        # with implicit Euler for i = 1,
        # (u_1^n - u_0^{n - 1}) / dt = - d_1 * u_1^n.
        # This can be written as
        # u^n = M_2 @ u^{n - 2} + M_1 @ u^{n - 1},
        M_crank_nicolson_2 = sparse.lil_matrix((len(self.ages),
                                                len(self.ages)))
        M_crank_nicolson_1 = sparse.lil_matrix((len(self.ages),
                                                len(self.ages)))
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
        # Convert to CSR for fast multiply.
        self.M_crank_nicolson_2 = M_crank_nicolson_2.tocsr()
        self.M_crank_nicolson_1 = M_crank_nicolson_1.tocsr()
        # The first time step, n = 1, uses implicit Euler:
        # (u_i^n - u_{i - 1}^{n - 1}) / dt = - d_i * u_i^n.
        # This can be written as
        # u^1 = M @ u^0,
        M_implicit_euler = sparse.lil_matrix((len(self.ages),
                                              len(self.ages)))
        # with
        # M[i, i - 1] = 1 / (1 + dt * d_i),
        M_implicit_euler.setdiag(diag1[1 : ], -1)
        # and, to prevent the last age group from aging out of the population,
        # M[-1, -1] = 1 / (1 + dt * d_{-1}).
        M_implicit_euler[-1, -1] = diag1[-1]
        # Convert to CSR for fast multiply.
        self.M_implicit_euler = M_implicit_euler.tocsr()
        # The trapezoid rule for the birth integral for i = 0,
        # u_0^n = \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) / 2 / da.
        # This can be written as
        # u_0^n = (v * b^n) @ u^n,
        # with
        # v = [0.5, 1, 1, ..., 1, 1, 0.5] / da.
        self.v_trapezoid = (numpy.hstack((0.5,
                                          numpy.ones(len(self.ages) - 2),
                                          0.5))
                            / agestep)

    def solve(self, birth_scaling):
        # The Crank–Nicolson method needs the solution at the 3 times:
        # the current time step and the 2 previous time steps.
        # The fundamental solution is a len(ages) x len(ages) matrix.
        Phi_cycle = solution_cycle(3, (len(self.ages), len(self.ages)))
        # Set up birth rate.
        birthRV = birth._from_param_values(
            self.parameters.birth_peak_time_of_year,
            self.parameters.birth_seasonal_coefficient_of_variation,
            _scaling=birth_scaling)
        birth_rate = birthRV.hazard
        # Avoid lookups in the loop.
        ages = self.ages
        n_ages = len(ages)
        t = self.t
        female_probability_at_birth \
            = self.parameters.female_probability_at_birth
        M_crank_nicolson_2 = self.M_crank_nicolson_2
        M_crank_nicolson_1 = self.M_crank_nicolson_1
        M_implicit_euler = self.M_implicit_euler
        v_trapezoid = self.v_trapezoid
        for (n, (t_n, Phi)) in enumerate(zip(t, Phi_cycle)):
            # `Phi[0]`, `Phi[1]`, and `Phi[2]` are the solution
            # at t_n, t_{n - 1}, and t_{n - 2}, respectively.
            # `solution_cycle` keeps them in sync as `t_n` is incremented.
            if n == 0:
                # Initial condition is the identity matrix.
                # Phi[0][:] = numpy.eye(n_ages)
                # Avoid building a new matrix.
                Phi[0][:] = 0
                Phi[0][numpy.diag_indices_from(Phi[0])] = 1
            else:
                # Aging & mortality.
                if n == 1:
                    # Use implicit Euler for the first time step.
                    # Phi[0][:] = M_implicit_euler @ Phi[1]
                    # Avoid building a new matrix.
                    Phi[0][:] = 0
                    # Phi[0] += M_implicit_euler @ Phi[1]
                    sparse._sparsetools.csr_matvecs(n_ages, n_ages, n_ages,
                                                    M_implicit_euler.indptr,
                                                    M_implicit_euler.indices,
                                                    M_implicit_euler.data,
                                                    Phi[1].ravel(),
                                                    Phi[0].ravel())
                else:
                    # Crank–Nicolson with implicit Euler for i = 1, -1.
                    # Phi[0][:] = (M_crank_nicolson_2 @ Phi[2]
                    #              + M_crank_nicolson_1 @ Phi[1])
                    # Avoid building a new matrix.
                    Phi[0][:] = 0
                    # Phi[0] += M_crank_nicolson_2 @ Phi[2]
                    sparse._sparsetools.csr_matvecs(n_ages, n_ages, n_ages,
                                                    M_crank_nicolson_2.indptr,
                                                    M_crank_nicolson_2.indices,
                                                    M_crank_nicolson_2.data,
                                                    Phi[2].ravel(),
                                                    Phi[0].ravel())
                    # Phi[0] += M_crank_nicolson_1 @ Phi[1]
                    sparse._sparsetools.csr_matvecs(n_ages, n_ages, n_ages,
                                                    M_crank_nicolson_1.indptr,
                                                    M_crank_nicolson_1.indices,
                                                    M_crank_nicolson_1.data,
                                                    Phi[1].ravel(),
                                                    Phi[0].ravel())
                # Birth.
                # Composite trapezoid rule at t = t_n.
                # b = female_probability_at_birth * birth_rate(t_n, ages))
                # Avoid building a new vector.
                b = birth_rate(t_n, ages)
                b *= female_probability_at_birth
                # Top row, i.e. age 0, newborns.
                # Phi[0][0] = (v_trapezoid * b) @ Phi[0]
                # Avoid building a new vector.
                numpy.matmul((v_trapezoid * b), Phi[0], out=Phi[0][0])
        # Return the solution at the final time.
        return Phi[0]


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
    mu0 = numpy.log(rho0) / solver.parameters.period
    # v0 is the eigenvector for both rho0 and mu0.
    v0 = _normalize(v0, solver.ages)
    return (mu0, v0, solver.ages)


def _find_growth_rate(birth_scaling, msparameters, agemax, agestep, solver):
    '''Find the population growth rate.'''
    mu0, _, _ = _find_dominant_eigen(birth_scaling, msparameters,
                                     agemax, agestep, solver)
    return mu0


@_cache.cache
def _find_birth_scaling(msparameters, agemax, agestep):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Reuse the solver to avoid setup/teardown.
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
    return optimize.brentq(_find_growth_rate, a, b, args=args)


# Default values.
_agemax = 35
_agestep = 0.01


def find_birth_scaling(parameters, agemax=_agemax, agestep=_agestep):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Call the cached version.
    msparameters = _MonodromySolver.Parameters(parameters)
    return _find_birth_scaling(msparameters, agemax, agestep)


def find_stable_age_structure(parameters, agemax=_agemax, agestep=_agestep):
    '''Find the stable age structure.'''
    msparameters = _MonodromySolver.Parameters(parameters)
    birth_scaling = _find_birth_scaling(msparameters, agemax, agestep)
    r, v, ages = _find_dominant_eigen(birth_scaling, msparameters,
                                      agemax, agestep)
    assert numpy.isclose(r, 0), 'Nonzero growth rate r={:g}.'.format(r)
    return (v, ages)


def fill_cache(parameters, agemax=_agemax, agestep=_agestep):
    '''Fill the cache so that subsequent calls to `find_birth_scaling()`
    and `find_stable_age_structure()` just read from the cache.'''
    find_birth_scaling(parameters, agemax, agestep)
