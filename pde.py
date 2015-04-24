#!/usr/bin/python

import numpy
from scipy import integrate
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import Parameters


parameters = Parameters.Parameters()
RVs = Parameters.RandomVariables(parameters)


tmax = 100.
tstep = 0.1

amax = 20.
astep = 0.1

a = numpy.arange(0., amax + astep, astep)

da = numpy.diff(ages)
A = - numpy.diag(numpy.hstack((da, 0.))) + numpy.diag(da, -1)

M = numpy.diag(RVs.mortality.hazard(a))

def B(t):
    Bval = numpy.zeros((len(a), ) * 2)
    Bval[0] = ((1 - parameters.probabilityOfMaleBirth)
               * RVs.birth.hazard(t, 0, a - t))
    return Bval


def rhs(N, t):
    dN = numpy.dot(B(t) - M + A, N)
    return dN


N0 = numpy.hstack((1, numpy.zeros(len(a) - 1)))

t = numpy.arange(0., tmax + tstep, tstep)

N = integrate.odeint(rhs, N0, t)

n = integrate.trapz(N, a, axis = 1)

(fig, ax) = pyplot.subplots(subplot_kw = {'projection': '3d'})
tvals = (t > t[-1] - 2)
(x, y) = numpy.meshgrid(a, t[tvals])
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
