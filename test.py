#!/usr/bin/python

import numpy
import matplotlib.pyplot

import herd
import Parameters


populationSize = 10000

infectionDuration = 21. / 365.

birthSeasonalVariance = 0.

# R0s = (5., 10., 20.)
R0s = (10., )


# tMax = numpy.inf
tMax = 10
# nRuns = 10
nRuns = 1
# debug = False
debug = True

def runSimulations(R0):
    # numpy.random.seed(1)

    T = []
    I = []
    extinctionTimes = []

    p = Parameters.Parameters()
    p.populationSize = populationSize
    p.infectionDuration = infectionDuration
    p.birthSeasonalVariance = birthSeasonalVariance
    p.R0 = R0

    for i in xrange(nRuns):
        print 'R0 = {:g}, run #{}'.format(R0, i + 1)

        h = herd.Herd(p,
                      debug = debug)
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
