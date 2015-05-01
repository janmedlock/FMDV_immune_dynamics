#!/usr/bin/python

import numpy
from scipy import sparse
from scipy import integrate

import Parameters
from Parameters import utility


def B(t, ages, parameters, RVs):
    Bval = sparse.lil_matrix((len(ages), ) * 2)
    Bval[0] = ((1 - parameters.probabilityOfMaleBirth)
               * RVs.birth.hazard(t, 0, ages - t))
    return Bval


def rhs(Y, t, ages, parameters, RVs, AM):
    (M, S, I, R) = numpy.hsplit(Y, 4)
    
    N = M + S + I + R

    forceOfInfection = RVs.transmissionRate * integrate.trapz(I, ages)

    dM = (B(t, ages, parameters, RVs).dot(N)
          + AM.dot(M)
          - 1. / parameters.maternalImmunityDuration * M)
    dS = (AM.dot(S)
          + 1. / parameters.maternalImmunityDuration * M
          - forceOfInfection * S)
    dI = (AM.dot(I)
          + forceOfInfection * S
          - 1. / parameters.infectionDuration * I)
    dR = (AM.dot(R)
          + 1. / parameters.infectionDuration * I)

    dY = numpy.hstack((dM, dS, dI, dR))

    return dY


def solve(tMax, ageMax, ageStep, parameters = None, RVs = None):
    if parameters is None:
        parameters = Parameters.Parameters()
    if RVs is None:
        RVs = Parameters.RandomVariables(parameters)

    (ages, (B_bar, A, M)) = utility.buildMatrices(RVs.mortality,
                                                  RVs.birth,
                                                  RVs.male,
                                                  ageMax = ageMax,
                                                  ageStep = ageStep)
    AM = A - M

    evect = utility.findDominantEigenpair(B_bar + AM)[1]
    N0 = evect / integrate.trapz(evect, ages) * parameters.populationSize

    M0 = numpy.where(ages < parameters.maternalImmunityDuration, N0, 0.)
    S0 = numpy.where((ages >= parameters.maternalImmunityDuration)
                     & (ages < 2.),
                     N0, 0.)
    I0 = S0 * 2. / integrate.trapz(S0, ages)
    S0 -= I0
    R0 = N0 - M0 - S0 - I0

    Y0 = numpy.hstack((M0, S0, I0, R0))

    t = numpy.linspace(0., tMax, 1001)

    Y = integrate.odeint(rhs, Y0, t, args = (ages, parameters, RVs, AM))

    return (t, ages, numpy.hsplit(Y, 4))


def get_endemic_equilibrium(parameters, tMax = 20.,
                            ageMax = 20., ageStep = 0.01):
    (t, ages, (M, S, I, R)) = solve(tMax, ageMax, ageStep, parameters)
    
    return (M[-1], S[-1], I[-1], R[-1])

    

if __name__ == '__main__':
    from matplotlib import pyplot

    tMax = 10.

    ageMax = 20.
    ageStep = 0.1

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
