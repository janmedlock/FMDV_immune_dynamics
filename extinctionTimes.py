#!/usr/bin/python

import numpy
import multiprocessing
import collections

import herd


def setRandomSeed():
    numpy.random.seed()

def doOne(parameters, tMax, *args, **kwds):
    h = herd.Herd(parameters, *args, **kwds)
    return h.findExtinctionTime(tMax)

def showResult(extinctionTime):
    print 'Extinct after {} days'.format(365. * extinctionTime)
   
def findExtinctionTimes(parameters,
                        tMax,
                        nRuns,
                        *args,
                        **kwds):
    pool = multiprocessing.Pool(initializer = setRandomSeed)
    
    results = [pool.apply_async(doOne,
                                (parameters, tMax, ) + args,
                                kwds,
                                callback = showResult)
               for n in xrange(nRuns)]

    pool.close()

    extinctionTimes = [r.get() for r in results]

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
    
    # Convert everything from years to days
    for (k, v) in mystats.iteritems():
        mystats[k] = 365. * v

    return mystats
    
def showStats(mystats):
    print ('stats: {'
           + ',\n        '.join(['{} = {}'.format(k, v)
                                 for (k, v) in mystats.iteritems()])
           + '}')


Key = collections.namedtuple('parameters', ('populationSize',
                                            'infectionDuration',
                                            'R0'))

def searchParameters(parameters,
                     tMax,
                     nRuns,
                     callback = None,
                     *args, **kwds):
    numpy.random.seed(1)

    eT = findExtinctionTimes(parameters,
                             tMax,
                             nRuns,
                             *args,
                             **kwds)

    if callable(callback):
        callback(parameters, eT)

    return eT


if __name__ == '__main__':
    import Parameters

    parameters = Parameters.Parameters()

    nRuns = 10
    tMax = 10.
    debug = False
    
    numpy.random.seed(1)

    eT = findExtinctionTimes(parameters, tMax, nRuns, debug = debug)

    mystats = findStats(eT)
    showStats(mystats)
