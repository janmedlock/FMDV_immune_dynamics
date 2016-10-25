#!/usr/bin/python3

import numpy
from scipy import integrate

import sys
sys.path.append('..')

import herd


def rhs(Y, t, parameters, rvs):
    (M, S, I, R) = Y

    B = (1. - parameters.male_probability_at_birth) * rvs.birth.scaling
    force_of_infection = parameters.transmission_rate * I
    
    dM = (B * (S + I + R)
          - M / parameters.maternal_immunity_duration
          + numpy.log(0.7) * M)
    dS = (M / parameters.maternal_immunity_duration
          + numpy.log(0.95) * S
          - force_of_infection * S)
    dI = (force_of_infection * S
          + numpy.log(0.95) * I \
          - I / parameters.recovery_infection_duration)
    dR = (I / parameters.recovery_infection_duration
          + numpy.log(0.95) * R)

    return (dM, dS, dI, dR)


def solve(tmax, parameters):
    rvs = herd.RandomVariables(parameters)

    (M0, I0, R0, S0) = (parameters.population_size
                        * rvs.endemic_equilibrium.weights[0])
    
    t = numpy.linspace(0., tmax, 1001)

    Y = integrate.odeint(rhs, (M0, S0, I0, R0), t, args = (parameters, rvs))

    (M, S, I, R) = numpy.hsplit(Y, 4)

    return (t, M, S, I, R)


if __name__ == '__main__':
    import pylab
    
    tmax = 5.

    parameters = herd.Parameters()
    parameters.population_size = 10000
    parameters.recovery_infection_duration = 21 / 365

    (t, M, S, I, R) = solve(tmax, parameters)

    pylab.plot(t, I / (M + S + I + R) * parameters.population_size)
    pylab.xlabel('time (days)')
    pylab.ylabel('number infected')

    pylab.show()
