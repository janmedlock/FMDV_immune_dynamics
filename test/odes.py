#!/usr/bin/python3

import numpy
from scipy import integrate

import Parameters


def rhs(Y, t, parameters, RVs):
    (M, S, I, R) = Y

    B = parameters.probabilityOfMaleBirth * RVs.birth.scaling
    forceOfInfection = RVs.transmissionRate * I
    
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


def solve(tMax, parameters):
    RVs = Parameters.RandomVariables(parameters)

    (M0, I0, R0, S0) = (parameters.populationSize
                        * RVs.endemicEquilibrium.weights[0])
    
    t = numpy.linspace(0., tMax, 1001)

    Y = integrate.odeint(rhs, (M0, S0, I0, R0), t, args = (parameters, RVs))

    (M, S, I, R) = numpy.hsplit(Y, 4)

    return (t, M, S, I, R)


if __name__ == '__main__':
    import pylab
    
    tMax = 5.

    parameters = Parameters.Parameters()
    parameters.populationSize = 10000
    parameters.infectionDuration = 21. / 365.
    parameters.R0 = 10.

    (t, M, S, I, R) = solve(tMax, parameters)

    pylab.plot(t, I / (M + S + I + R) * parameters.populationSize)
    pylab.xlabel('time (days)')
    pylab.ylabel('number infected')

    pylab.show()
