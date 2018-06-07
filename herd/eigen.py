import numpy
from scipy import optimize, sparse

from . import birth
from . import mortality
from . import dominant_eigen
from .shelved import Shelved


def _build_ages_and_matrices(parameters, agemax=25, agestep=0.01):
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    N = len(ages)
    # Birth
    # The first row, B_bar[0], is the cumulative birth hazards over a year
    # times the probability of female birth.
    birthRV = birth.gen(parameters, _scaling=1)
    # The cumulative hazard is -logsf.
    cumulative_hazard = -birthRV.logsf(1, parameters.start_time, ages)
    birth_rate = (parameters.female_probability_at_birth
                  * cumulative_hazard)
    B_bar = sparse.lil_matrix((N, N))
    B_bar[0] = birth_rate
    # Mortality and aging
    mortalityRV = mortality.gen(parameters)
    mortality_rate = mortalityRV.hazard(ages)
    # No aging out of the last age group.
    aging_rate = numpy.hstack((1 / numpy.diff(ages), 0))
    T = sparse.dia_matrix((N, N))
    T.setdiag(- mortality_rate - aging_rate, 0)
    T.setdiag(aging_rate[: -1], -1)
    # Convert to CSR for fast multiply.
    B_bar = B_bar.asformat('csr')
    T = T.asformat('csr')
    # Initial guess for eigenvector.
    v0 = mortalityRV.sf(ages)
    return (ages, (B_bar, T, v0))


def _build_G(birth_scaling, B_bar, T):
    return birth_scaling * B_bar + T


def _find_growth_rate(birth_scaling, B_bar, T, v0):
    G = _build_G(birth_scaling, B_bar, T)
    r, _ = dominant_eigen.find(G, v0=v0)
    return r


@Shelved('birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_birth_scaling(parameters, matrices=None, *args, **kwargs):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    if matrices is None:
        _, matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    B_bar, T, v0 = matrices
    a = 0
    # We know that at the lower limit a = 0,
    # `_find_growth_rate(0, ...) < 0`,
    # so we need to find an upper limit `b`
    # with `_find_growth_rate(b, ...) >= 0`.
    b = 1
    while _find_growth_rate(b, B_bar, T, v0) < 0:
        a = b
        b *= 2
    return optimize.brentq(_find_growth_rate, a, b, args=(B_bar, T, v0))


@Shelved('start_time',
         'birth_peak_time_of_year',
         'birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_stable_age_structure(parameters, *args, **kwargs):
    '''Find the stable age structure.'''
    ages, matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    B_bar, T, v0 = matrices
    birth_scaling = find_birth_scaling(parameters, matrices=(B_bar, T, v0))
    G = _build_G(birth_scaling, B_bar, T)
    r, v = dominant_eigen.find(G, v0=v0)
    assert numpy.isclose(r, 0), 'Nonzero growth rate.'
    return (ages, v)
