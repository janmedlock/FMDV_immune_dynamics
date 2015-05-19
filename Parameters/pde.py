#!/usr/bin/python

import numpy
from scipy import sparse
from scipy import integrate

from . import utility
from . import birth
from . import transmission


def rhs(Y, t, AM, B, forceOfInfection, recoveryRate):
    (S, I, R) = numpy.hsplit(Y, 3)
    
    N = S + I + R

    lambda_ = forceOfInfection(I)

    dS = (B(t).dot(N)
          + AM.dot(S)
          - lambda_ * S)
    dI = (AM.dot(I)
          + lambda_ * S
          - recoveryRate * I)
    dR = (AM.dot(R)
          + recoveryRate * I)

    dY = numpy.hstack((dS, dI, dR))

    return dY


def solve(tMax, ageMax, ageStep, parameters, Y0 = None):
    matrices = utility.buildMatrices(parameters,
                                     ageMax = ageMax,
                                     ageStep = ageStep)
    (ages, (B_bar, A, M)) = matrices

    AM = A - M

    birthRV = birth.birth_gen(parameters)
    def B(t):
        Bval = sparse.lil_matrix((len(ages), ) * 2)
        Bval[0] = ((1 - parameters.probabilityOfMaleBirth)
                   * birthRV.hazard(t, 0, ages - t))
        return Bval

    transmissibility = transmission.transmissionRate_gen(parameters)
    susceptibility = numpy.where(ages >= parameters.maternalImmunityDuration,
                                 1., 0.)
    def forceOfInfection(I):
        return integrate.trapz(transmissibility * I, ages) * susceptibility

    recoveryRate = 1. / parameters.infectionDuration

    if Y0 is None:
        eigenpair = utility.findDominantEigenpair(parameters,
                                                  _matrices = matrices)
        eigenvector = eigenpair[1][1]
        N0 = (eigenvector / integrate.trapz(eigenvector, ages)
              * parameters.populationSize)

        # Everyone under 2 susceptible (except for maternal immunity).
        S0 = numpy.where(ages < 2., N0, 0.)
        # initial infections.
        I0 = 0.01 * S0
        S0 -= I0
        R0 = N0 - S0 - I0

        Y0 = numpy.hstack((S0, I0, R0))

    t = numpy.linspace(0., tMax, 1001)

    (Y, info) = integrate.odeint(rhs, Y0, t,
                                 args = (AM, B, forceOfInfection, recoveryRate),
                                 mxstep = 10000,
                                 full_output = True)
    if numpy.any(numpy.diff(info['tcur']) < -1e-16):
        raise RuntimeError('ODE solver failed!')

    (S, I, R) = numpy.hsplit(Y, 3)

    # Split susceptibles into maternal immunity and not.
    mask = (ages < parameters.maternalImmunityDuration)
    M = numpy.where(mask[numpy.newaxis, :], S, 0.)
    S -= M

    return (t, ages, (M, S, I, R))


def getPeriod(t, ages, (M, S, I, R), abserr = 1e-3, relerr = 1e-3,
              periodMax = 3):
    Y = numpy.hstack((M + S, I, R))
    for period in range(1, periodMax + 1):
        i = numpy.argwhere(t <= t[-1] - period)[-1]
        if (numpy.linalg.norm(Y[i] - Y[-1])
            <= (abserr + relerr * numpy.linalg.norm(Y[-1]))):
            return period
    else:
        raise ValueError('period not found!')
    

@utility.shelved
def _getEndemicEquilibrium(parameters, tMax, ageMax, ageStep):
    (t, ages, (M, S, I, R)) = solve(tMax, ageMax, ageStep, parameters)

    period = getPeriod(t, ages, (M, S, I, R))

    ICs = []
    for j in range(period):
        k = numpy.argwhere(t <= t[-1] - j)[-1, 0]
        ICs.append((M[k], S[k], I[k], R[k]))

    return (ages, ICs)

    
def getEndemicEquilibrium(parameters, tMax = 200.,
                          ageMax = 20., ageStep = 0.01):
    # The PDE solutions simply scale multiplicatively with
    # populationSize, so factor that out for more efficient caching.
    populationSize = parameters.populationSize
    parameters.populationSize = 1.
    (ages, ICs) = _getEndemicEquilibrium(parameters,
                                         tMax, ageMax, ageStep)
    parameters.populationSize = populationSize

    ICs_ = [[x * populationSize for x in IC] for IC in ICs]

    return (ages, ICs_)
    

if __name__ == '__main__':
    from matplotlib import pyplot
    import Parameters

    tMax = 10.

    ageMax = 20.
    ageStep = 0.01

    parameters = Parameters.Parameters()
    parameters.populationSize = 10000
    parameters.infectionDuration = 21. / 365.
    parameters.R0 = 10.
    parameters.birthSeasonalVariance = 1.

    (t, ages, (M, S, I, R)) = solve(tMax, ageMax, ageStep, parameters)

    i = integrate.trapz(I, ages, axis = 1)

    (fig, ax) = pyplot.subplots()
    ax.plot(t, i)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Infected buffaloes')

    pyplot.show()
