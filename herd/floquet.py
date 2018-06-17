import os.path

from joblib import Memory
import numpy
from scipy.integrate import trapz
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

from . import birth
from . import mortality
from . import dominant_eigen


_agemax = 35
_agestep = 0.01
_period = 1


# Some of the functions below are very slow, so the values are cached to
# disk with `joblib.Memory()` so they are only computed once.
_cachedir = os.path.join(os.path.dirname(__file__), '__cache__')
_cache = Memory(_cachedir, verbose=1)


def _arange(start, stop, step, endpoint=False):
    '''`numpy.arange()` that can optionally include the right endpoint `stop`.'''
    val = numpy.arange(start, stop, step)
    if endpoint:
        if not numpy.isclose(val[-1], stop):
            val = numpy.hstack((val, stop))
    return val


def _normalize(v, ages):
    '''Normalize v so that it integrates to 1.'''
    return v / trapz(v, ages)


def _params_to_args(parameters):
    '''Convert `herd.parameters.Parameters()` object `parameters` to
    arguments needed by `_find_monodromy()`.'''
    return (parameters.birth_peak_time_of_year,
            parameters.birth_seasonal_coefficient_of_variation,
            parameters.female_probability_at_birth,
            parameters.start_time)


def _find_monodromy(birth_scaling,
                    birth_peak_time_of_year,
                    birth_seasonal_coefficient_of_variation,
                    female_probability_at_birth,
                    start_time, period,
                    agemax, agestep):
    '''Find the monodromy matrix Phi(period) by solving
    (d/dt + d/da) Phi = - d(a) Phi,
    Phi(t, 0) = \int_0^{inf} b(t, a) Phi(t, a) da
    Phi(0, a) = I.
    The PDE is solved using the Crank–Nicolson method on characteristics.
    '''
    ages = _arange(0, agemax, agestep, endpoint=True)
    tstep = agestep
    t = _arange(start_time, start_time + period, tstep, endpoint=True)
    birthRV = birth._from_param_values(birth_peak_time_of_year,
                                       birth_seasonal_coefficient_of_variation,
                                       _scaling=birth_scaling)
    birth_rate = birthRV.hazard
    mortalityRV = mortality._from_param_values()
    mortality_rate = mortalityRV.hazard
    # The Crank–Nicolson method is
    # (u_i^k - u_{i - 2}^{k - 2}) / 2 / dt
    # = - d_{i - 1} * (u_i^k + u_{i - 2}^{k - 2}) / 2,
    # for i = 2, 3, ...,
    # with implicit Euler for i = 1,
    # (u_1^k - u_0^{k - 1}) / dt = - d_1 * u_1^k,
    # Let
    # T_2[i, i - 2] = (1 - dt * d_{i - 1}) / (1 + dt * d_{i - 1}),
    # for i = 2, 3, ...,
    # and
    # T_1[1, 0] = 1 / (1 + dt * d_i).
    # To prevent the last age group from aging out of the population,
    # we set
    # T2[-1, -1] = (1 - dt * d_{-1}) / (1 + dt * d_{-1}).
    # To prevent the next to last age group from aging out,
    # T1[-1, -2] = 1 / (1 + dt * d_{-1}).
    # Then
    # u^k = T_2 @ u^{k - 2} + T_1 @ u^{k - 1}.
    T2 = lil_matrix((len(ages), len(ages)))
    diag2 = ((1 - tstep * mortality_rate(ages))
             / (1 + tstep * mortality_rate(ages)))
    T2.setdiag(diag2[1 : -1], -2)
    T2[-1, -1] = diag2[-1]
    T2 = T2.tocsr()
    T1 = lil_matrix((len(ages), len(ages)))
    diag1 = 1 / (1 + tstep * mortality_rate(ages))
    T1[1, 0] = diag1[1]
    T1[-1, -2] = diag1[-1]
    T1 = T1.tocsr()
    # Implicit Euler for the first time step.
    # The first time step, k = 1, uses implicit Euler:
    # (u_i^k - u_{i - 1}^{k - 1}) / dt = - d_i * u_i^k.
    # Let
    # T_euler[i, i - 1] = 1 / (1 + dt * d_i).
    # To prevent the last age group from aging out of the population,
    # we set
    # T_euler[-1, -1] = 1 / (1 + dt * d_{-1})
    # Then
    # u^1 = T_euler @ u^0.
    T_euler = lil_matrix((len(ages), len(ages)))
    T_euler.setdiag(diag1[1 : ], -1)
    T_euler[-1, -1] = diag1[-1]
    T_euler = T_euler.tocsr()
    # The trapezoid rule for the birth integral for i = 0,
    # u_0^k = \sum_j (b_j^k u_j^k + b_{j + 1}^k u_{j + 1}^k) / 2 / da.
    # Let
    # T_int = [0.5, 1, 1, ..., 1, 1, 0.5].
    # Then
    # u_0^k = (T_int * b^k) @ u^k.
    T_int = numpy.hstack((0.5, numpy.ones(len(ages) - 2), 0.5)) / agestep
    Phi = numpy.zeros((len(t), len(ages), len(ages)))
    Phi[0] = numpy.eye(len(ages))
    for (i, t_i) in enumerate(t[1 : ], 1):
        # Aging & mortality.
        if i == 1:
            # Use implicit Euler for the first time step.
            Phi[i] = T_euler @ Phi[i - 1]
        else:
            # Crank–Nicolson with implicit Euler for j = 1, -1.
            Phi[i] = (T2 @ Phi[i - 2] + T1 @ Phi[i - 1])
        # Birth.
        # Composite trapezoid rule at t = t_i.
        b = female_probability_at_birth * birth_rate(t_i, ages)
        Phi[i, 0] = (T_int * b) @ Phi[i]
    return (Phi[-1], ages, period)


@_cache.cache(verbose=0)
def _find_dominant_eigen(birth_scaling, *args):
    '''Find the dominant Floquet exponent
    (the one with the largest real part)
    and its corresponding eigenvector.'''
    PhiT, ages, period = _find_monodromy(birth_scaling, *args)
    # Finding the matrix B = log(Phi(T)) / T is very expensive,
    # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
    # and convert.
    rho0, v0 = dominant_eigen.find(PhiT, which='LM')
    # rho0 is the dominant (largest magnitude) Floquet multiplier.
    # mu0 is the dominant (largest real part) Floquet exponent.
    # They are related by rho0 = exp(mu0 * T).
    mu0 = numpy.log(rho0) / period
    # v0 is the eigenvector for both rho0 and mu0.
    v0 = _normalize(v0, ages)
    return (mu0, v0, ages)


def _find_growth_rate(birth_scaling, *args):
    '''Find the population growth rate.'''
    mu0, _, _ = _find_dominant_eigen(birth_scaling, *args)
    return mu0


@_cache.cache
def _find_birth_scaling(*args):
    '''Find the birth scaling that gives population growth rate r = 0.'''
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


def find_birth_scaling(parameters, agemax=_agemax, agestep=_agestep):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Call the cached version.
    return _find_birth_scaling(*_params_to_args(parameters),
                               _period, agemax, agestep)


def find_stable_age_structure(parameters, agemax=_agemax, agestep=_agestep):
    '''Find the stable age structure.'''
    birth_scaling = find_birth_scaling(parameters, agemax, agestep)
    r, v, ages = _find_dominant_eigen(birth_scaling,
                                      *_params_to_args(parameters),
                                      _period, agemax, agestep)
    assert numpy.isclose(r, 0), 'Nonzero growth rate r={:g}.'.format(r)
    return (v, ages)
