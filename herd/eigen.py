import numpy
from scipy import integrate, optimize, sparse

from . import birth
from . import mortality
from . import dominant_eigen
from .shelved import Shelved


def _build_ages_and_monodromy_args(parameters, agemax=25, agestep=0.01):
    ages = numpy.arange(0, agemax, agestep)
    if not numpy.isclose(ages[-1], agemax):
        ages = numpy.hstack((ages, agemax))
    N = len(ages)
    # Birth
    # The first row, B[0], is the birth hazard
    # times the probability of female birth.
    birthRV = birth.gen(parameters, _scaling=1)
    def birth_rate(t):
        return parameters.female_probability_at_birth * birthRV.hazard(t, ages)
    B = sparse.lil_matrix((N, N))
    # Establish sparsity pattern.  This row will get updated at each time.
    B[0] = 1
    # Mortality and aging
    mortality_rate = mortality.gen(parameters).hazard(ages)
    # No aging out of the last age group.
    aging_rate = numpy.hstack((1 / numpy.diff(ages), 0))
    T = sparse.dia_matrix((N, N))
    T.setdiag(- mortality_rate - aging_rate, 0)
    T.setdiag(aging_rate[: -1], -1)
    # Convert to CSR for fast multiply.
    B = B.asformat('csr')
    T = T.asformat('csr')
    args = (parameters.start_time, N, birth_rate, B, T)
    return (ages, args)


def _monodromy_ODEs(phi, t, birth_scaling, N, birth_rate, B, T):
    Phi = phi.reshape((N, N))
    B[0] = birth_scaling * birth_rate(t)
    dPhi_dt = (B + T) @ Phi
    return dPhi_dt.reshape(-1)


def _find_monodromy_matrix(birth_scaling, start_time, N, birth_rate, B, T):
    Phi0 = numpy.eye(N)
    phi0 = Phi0.reshape(-1)
    # Solve over 1 year, the period of the birth rate.
    t = (start_time, start_time + 1)
    args = (birth_scaling, N, birth_rate, B, T)
    phi = integrate.odeint(_monodromy_ODEs, phi0, t, args=args,
                           mxstep=100000)[-1]
    Phi = phi.reshape((N, N))
    return Phi


def _find_growth_rate(birth_scaling, *args):
    Phi = _find_monodromy_matrix(birth_scaling, *args)
    R = dominant_eigen.find(Phi, which='LM', return_eigenvector=False)
    return numpy.log(R)


@Shelved('birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_birth_scaling(parameters, *args, **kwargs):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    (ages, args) = _build_ages_and_monodromy_args(parameters, *args, **kwargs)
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


@Shelved('start_time',
         'birth_peak_time_of_year',
         'birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_stable_age_structure(parameters, *args, **kwargs):
    '''Find the stable age structure.'''
    birth_scaling = find_birth_scaling(parameters, *args, **kwargs)
    (ages, args) = _build_ages_and_monodromy_args(parameters, *args, **kwargs)
    Phi = _find_monodromy_matrix(birth_scaling, *args)
    R, v = dominant_eigen.find(Phi, which='LM', return_eigenvector=True)
    assert numpy.isclose(numpy.log(R), 0, atol=1e-6), 'Nonzero growth rate.'
    v /= integrate.trapz(v, ages)
    return (ages, v)
