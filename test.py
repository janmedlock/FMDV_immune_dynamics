#!/usr/bin/python

import numpy
import matplotlib.pyplot

import parameters
import herd

parameters.populationSize = 1000

parameters.infectionDuration = 21. / 365.
parameters.recovery = parameters.deterministic(
    scale = parameters.infectionDuration)

R0s = (5., 10., 20.)


tMax = numpy.inf
nRuns = 100
debug = False

def runSimulations(R0):
    parameters.R0 = R0
    parameters.setTransmissionRate()

    #numpy.random.seed(1)

    T = []
    I = []
    extinctionTimes = []

    for i in xrange(nRuns):
        h = herd.Herd(debug = debug)
        result = h.run(tMax)
        (t, i) = map(numpy.array, zip(*result))

        T.append(t)
        I.append(i)
        extinctionTimes.append(t[-1])

    # Compute average
    Tavg = numpy.unique(numpy.sort(numpy.hstack(T)))
    Iavg = numpy.empty((len(Tavg), nRuns))
    for r in xrange(nRuns):
        for j in xrange(len(T[r])):
            if j < len(T[r]) - 1:
                index = numpy.nonzero((Tavg >= T[r][j])
                                      & (Tavg < T[r][j + 1]))[0]
            else:
                index = numpy.nonzero(Tavg >= T[r][-1])[0]
            Iavg[index, r] = I[r][j]

    return (Tavg, Iavg, extinctionTimes)


Tavg = {}
Iavg = {}
extinctionTimes = {}
fig = matplotlib.pyplot.figure()
axes = fig.add_subplot(1, 1, 1)

for R0 in R0s:
    (Tavg[R0], Iavg[R0], extinctionTimes[R0]) = runSimulations(R0)

    axes.step(365. * Tavg[R0], numpy.mean(Iavg[R0], axis = 1),
              where = 'post', label = '$R_0 = {}$'.format(R0))


axes.set_xlabel('time (days)')
axes.set_ylabel('number infected')
axes.set_ylim(ymin = 0.)
axes.legend(loc = 'upper right')


matplotlib.pyplot.show()
