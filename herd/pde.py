#!/usr/bin/python3

import numpy
from scipy import sparse
from scipy import integrate

from . import utility
from . import birth
from . import transmission_rate


def rhs(Y, t, AM, B, force_of_infection, recovery_rate):
    (S, I, R) = numpy.hsplit(Y, 3)
    
    N = S + I + R

    lambda_ = force_of_infection(I)

    dS = (B(t).dot(N)
          + AM.dot(S)
          - lambda_ * S)
    dI = (AM.dot(I)
          + lambda_ * S
          - recovery_rate * I)
    dR = (AM.dot(R)
          + recovery_rate * I)

    dY = numpy.hstack((dS, dI, dR))

    return dY


def solve(tmax, agemax, agestep, parameters, Y0 = None):
    matrices = utility.build_matrices(parameters,
                                      agemax = agemax,
                                      agestep = agestep)
    if not hasattr(parameters, 'population_size'):
        parameters.population_size = 1.

    (ages, (B_bar, A, M)) = matrices

    AM = A - M

    birthRV = birth.gen(parameters)
    def B(t):
        Bval = sparse.lil_matrix((len(ages), ) * 2)
        Bval[0] = ((1 - parameters.male_probability_at_birth)
                   * birthRV.hazard(t, 0, ages - t))
        return Bval

    transmissibility = transmission_rate.gen(parameters)
    susceptibility = numpy.where(ages >= parameters.maternal_immunity_duration,
                                 1., 0.)
    def force_of_infection(I):
        return integrate.trapz(transmissibility * I, ages) * susceptibility

    recovery_rate = 1. / parameters.recovery_infection_duration

    if Y0 is None:
        eigenpair = utility.find_dominant_eigenpair(parameters,
                                                    _matrices = matrices)
        eigenvector = eigenpair[1][1]
        N0 = (eigenvector / integrate.trapz(eigenvector, ages)
              * parameters.population_size)

        # Everyone under 2 susceptible (except for maternal immunity).
        S0 = numpy.where(ages < 2., N0, 0.)
        # initial infections.
        I0 = 0.01 * S0
        S0 -= I0
        R0 = N0 - S0 - I0

        Y0 = numpy.hstack((S0, I0, R0))

    t = numpy.linspace(0., tmax, 1001)

    (Y, info) = integrate.odeint(rhs, Y0, t,
                                 args = (AM, B,
                                         force_of_infection, recovery_rate),
                                 mxstep = 10000,
                                 full_output = True)
    if numpy.any(numpy.diff(info['tcur']) < -1e-16):
        raise RuntimeError('ODE solver failed!')

    (S, I, R) = numpy.hsplit(Y, 3)

    # Split susceptibles into maternal immunity and not.
    mask = (ages < parameters.maternal_immunity_duration)
    M = numpy.where(mask[numpy.newaxis, :], S, 0.)
    S -= M

    return (t, ages, (M, S, I, R))


def get_period(t, ages, X, abserr = 1e-3, relerr = 1e-3,
               periodmax = 3):
    (M, S, I, R) = X
    Y = numpy.hstack((M + S, I, R))
    for period in range(1, periodmax + 1):
        i = numpy.argwhere(t <= t[-1] - period)[-1]
        if (numpy.linalg.norm(Y[i] - Y[-1])
            <= (abserr + relerr * numpy.linalg.norm(Y[-1]))):
            return period
    else:
        raise ValueError('period not found!')
    

def get_limit_cycle(parameters, agemax, agestep,
                    periodmax = 3, t_burnin = 100.):
    from scipy import special
    from scipy import optimize

    print('Running burn-in...')
    (t, ages, (M, S, I, R)) = solve(t_burnin, agemax, agestep, parameters)
    print('Burn-in finished.')

    Y0 = numpy.hstack((M[-1] + S[-1], I[-1], R[-1]))
    tmax = special.factorial(periodmax)
    def f(Y0):
        (t, ages, (M, S, I, R)) = solve(tmax, agemax, agestep, parameters,
                                        Y0 = Y0)
        return numpy.hstack((M[-1] + S[-1], I[-1], R[-1]))

    print('Running root solver...')
    sol = optimize.fixed_point(f, Y0, xtol = 1e-3, maxiter = 1000)
    print('Root solver finshed.')

    Y0 = sol.x
    (t, ages, Y) = solve(tmax, agemax, agestep, parameters, Y0 = Y0)

    period = get_period(t, ages, Y, periodmax = periodmax)

    ICs = []
    for j in range(period):
        k = numpy.argwhere(t <= t[-1] - j)[-1, 0]
        ICs.append([y[k] for y in Y])

    return ICs


@utility.shelved
def _get_endemic_equilibrium(parameters, tmax, agemax, agestep):
    (t, ages, Y) = solve(tmax, agemax, agestep, parameters)

    period = get_period(t, ages, Y)

    ICs = []
    for j in range(period):
        k = numpy.argwhere(t <= t[-1] - j)[-1, 0]
        ICs.append([y[k] for y in Y])

    return (ages, ICs)

    
def get_endemic_equilibrium(parameters, tmax = 200.,
                            agemax = 20., agestep = 0.01):
    # The PDE solutions simply scale multiplicatively with
    # population_size, so factor that out for more efficient caching.
    population_size = parameters.population_size
    del parameters.population_size
    (ages, ICs) = _get_endemic_equilibrium(parameters,
                                           tmax, agemax, agestep)
    parameters.population_size = population_size

    ICs_ = [[x * population_size for x in IC] for IC in ICs]

    return (ages, ICs_)
