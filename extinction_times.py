#!/usr/bin/python3

import numpy

import herd


def findExtinctionTimes(nRuns,
                        parameters,
                        tMax,
                        *args,
                        **kwds):
    data = herd.multirun(nRuns, parameters, tMax, *args, **kwds)

    (T, X) = zip(*(zip(*y) for (runNumber, y) in data))

    extinctionTimes = [t[-1] if (x[-1][2] == 0) else None
                       for (t, x) in zip(T, X)]

    return extinctionTimes


def ppf(D, q, a = 0):
    Daug = numpy.asarray(sorted(D) + [a])
    indices = numpy.ceil(numpy.asarray(q) * len(D) - 1).astype(int)
    return Daug[indices]

def proportion_ge_x(D, x):
    return float(len(numpy.compress(numpy.asarray(D) >= x, D))) / float(len(D))

def findStats(extinctionTimes):
    mystats = {}
    
    mystats['median'] = numpy.median(extinctionTimes)
    mystats['mean'] = numpy.mean(extinctionTimes)
        
    mystats['q_90'] = ppf(extinctionTimes, 0.9)
    mystats['q_95'] = ppf(extinctionTimes, 0.95)
    mystats['q_99'] = ppf(extinctionTimes, 0.99)

    mystats['proportion >= 1'] = proportion_ge_x(extinctionTimes, 1.)
    mystats['proportion >= 10'] = proportion_ge_x(extinctionTimes, 10.)
    
    return mystats
    
def showStats(mystats):
    print('stats: {'
          + ',\n        '.join(['{} = {}'.format(k, v)
                                for (k, v) in mystats.items()])
          + '}')


if __name__ == '__main__':
    import Parameters

    p = Parameters.Parameters()

    p.populationSize = 100
    p.infectionDuration = 21. / 365.
    p.R0 = 10.
    p.birthSeasonalVariance = 1.

    nRuns = 10000
    tMax = 5.
    debug = False
    
    eT = findExtinctionTimes(nRuns, p, tMax, debug = debug)

    mystats = findStats(eT)
    showStats(mystats)
