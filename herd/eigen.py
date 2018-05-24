import numpy
from scipy import optimize, sparse

from . import birth
from . import mortality
from . import shelved


def _build_ages_and_matrices(parameters, agemax=25, agestep=0.01):
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    N = len(ages)
    # Birth
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    # The mean integral is the cumulative hazard, which is -logsf.
    birthRV = birth.gen(parameters, _scaling=1)
    cumulative_hazard = -birthRV.logsf(1, parameters.start_time, ages)
    mean_birth_rate = parameters.female_probability_at_birth * cumulative_hazard
    B_bar = sparse.lil_matrix((N, N))
    B_bar[0] = mean_birth_rate
    # Mortality and aging
    mortalityRV = mortality.gen(parameters)
    mortality_rate = mortalityRV.hazard(ages)
    # Don't fall out of the last age group.
    aging_rate = numpy.hstack((1 / numpy.diff(ages), 0))
    T = sparse.dia_matrix((N, N))
    T.setdiag(- mortality_rate - aging_rate, 0)
    T.setdiag(aging_rate[: -1], -1)
    # Convert to CSR for fast multiply.
    # The root-finders in `scipy.optimize` need a `tuple`,
    # not a `list`.
    matrices = tuple([X.asformat('csr') for X in (B_bar, T)])
    return (ages, matrices)


def _build_G(birth_scaling, B_bar, T):
    return birth_scaling * B_bar + T


def find_dominant_eigenpair(A, which='LR'):
    '''Find the dominant eigenvalue & eigenvector of A using
    `scipy.sparse.linalg.eigs()`.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    assert numpy.isfinite(A[A.nonzero()]).all(), 'A has inf/NaN entries.'
    L, V = sparse.linalg.eigs(A, k=1, which=which, maxiter=100000)
    l0 = numpy.real_if_close(L[0])
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    v0 = V[:, 0]
    v0 = numpy.real_if_close(v0 / v0.sum())
    assert all(numpy.isreal(v0)), 'Complex dominant eigenvector: {}'.format(v0)
    assert all((numpy.real(v0) >= 0) | numpy.isclose(v0, 0)), \
        'Negative component in the dominant eigenvector: {}'.format(v0)
    return (l0, v0)


def _find_growth_rate(birth_scaling, *matrices):
    G = _build_G(birth_scaling, *matrices)
    r, _ = find_dominant_eigenpair(G)
    return r


# `start_time` doesn't matter since we're integrating a 1-year-periodic
# function over 1 year.
@shelved.Shelved('birth_seasonal_coefficient_of_variation',
                 'female_probability_at_birth')
def find_birth_scaling(parameters, matrices=None, *args, **kwargs):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    if matrices is None:
        _, matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    a, b = (0, 1)
    # We know that the growth rate is negative at `birth_scaling = 0`,
    # `_find_growth_rate(0) < 0`, so we need to find an upper limit `b`
    # with `_find_growth_rate(b) >= 0`.
    while _find_growth_rate(b, *matrices) < 0:
        b *= 2
    birth_scaling, info = optimize.brentq(_find_growth_rate, a, b,
                                          args=matrices, full_output=True)
    assert info.converged, info.flag
    return birth_scaling


@shelved.Shelved('birth_seasonal_coefficient_of_variation',
                 'female_probability_at_birth',
                 'start_time')
def find_stable_age_structure(parameters, *args, **kwargs):
    '''Find the stable age structure.'''
    ages, matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    birth_scaling = find_birth_scaling(parameters, matrices=matrices)
    G = _build_G(birth_scaling, *matrices)
    r, v = find_dominant_eigenpair(G)
    assert numpy.isclose(r, 0), 'Nonzero growth rate.'
    return (ages, v)
