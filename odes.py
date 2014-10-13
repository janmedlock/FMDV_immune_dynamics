#!/usr/bin/python

import numpy
import scipy.integrate
import parameters


def rhs(Y, t):
    (M, S, I, R) = Y

    B = parameters.birth.hazard(t, 0., 4.)
    forceOfInfection = parameters.transmissionRate * I
    
    dM = B * (S + I + R) \
      - M / parameters.maternalImmunityDuration \
      + numpy.log(0.7) * M
    dS = M / parameters.maternalImmunityDuration \
      + numpy.log(0.95) * S \
      - forceOfInfection * S
    dI = forceOfInfection * S \
      + numpy.log(0.95) * I \
      - I / parameters.infectionDuration
    dR = I / parameters.infectionDuration \
      + numpy.log(0.95) * R

    return (dM, dS, dI, dR)


def solve(tMax):
    M0 = parameters.populationSize \
      * parameters.ageStructure.cdf(parameters.maternalImmunityDuration)
    I0 = parameters.initialInfections
    R0 = parameters.populationSize \
      * (1. - parameters.ageStructure.cdf(2.))
    S0 = parameters.populationSize - M0 - I0 - R0
    
    t = numpy.linspace(0., tMax, 1001)

    Y = scipy.integrate.odeint(rhs, (M0, S0, I0, R0), t)

    (M, S, I, R) = numpy.hsplit(Y, 4)

    return (t, M, S, I, R)


if __name__ == '__main__':
    import pylab
    
    tMax = 1.

    (t, M, S, I, R) = solve(tMax)

    pylab.plot(365. * t, I)
    pylab.xlabel('time (days)')
    pylab.ylabel('number infected')

    pylab.show()
