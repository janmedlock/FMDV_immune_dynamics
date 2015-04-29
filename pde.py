#!/usr/bin/python

import numpy
from scipy import sparse
from scipy import integrate
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import Parameters
from Parameters import utility


parameters = Parameters.Parameters()
parameters.populationSize = 10000
parameters.infectionDuration = 21. / 365.
parameters.R0 = 10.

RVs = Parameters.RandomVariables(parameters)


tmax = 10.
tstep = 0.01

ageMax = 20.
ageStep = 0.1

(ages, matrices) = utility.buildMatrices(RVs.mortality, RVs.birth, RVs.male,
                                         ageMax = ageMax, ageStep = ageStep)
(B_bar, A, M) = matrices

def B(t):
    Bval = sparse.lil_matrix((len(ages), ) * 2)
    Bval[0] = ((1 - parameters.probabilityOfMaleBirth)
               * RVs.birth.hazard(t, 0, ages - t))
    return Bval


def rhs(N, t):
    dN = (B(t) - M + A).dot(N)
    return dN


N0 = (utility.findDominantEigenpair(B_bar + A - M)[1]
      * parameters.populationSize)

t = numpy.arange(0., tmax + tstep, tstep)

N = integrate.odeint(rhs, N0, t)

n = integrate.trapz(N, ages, axis = 1)

(fig, ax) = pyplot.subplots(subplot_kw = {'projection': '3d'})
tvals = (t > t[-1] - 2)
(x, y) = numpy.meshgrid(ages, t[tvals])
z = N[tvals]
ax.plot_surface(x, y, z, linewidth = 0)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Time (years)')
ax.set_zlabel('Buffaloes')

(fig, ax) = pyplot.subplots()
ax.plot(t, n)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Total buffaloes')

(fig, ax) = pyplot.subplots()
m = numpy.interp(t[t >= 1] - 1, t, n)
r = (n[t >= 1] - m) / m
y = numpy.hstack((numpy.nan * numpy.ones_like(t[t < 1]), r))
ax.plot(t, y)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Growth rate (per year)')

pyplot.show()
