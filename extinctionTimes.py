#!/usr/bin/python

import numpy
import scipy.stats
import multiprocessing
import collections
import parameters
import herd


def setRandomSeed():
    numpy.random.seed()

def doOne(tMax, *args, **kwds):
    h = herd.Herd(*args, **kwds)
    return h.findExtinctionTime(tMax)

def showResult(extinctionTime):
    print 'Extinct after {} days'.format(365. * extinctionTime)
   
def findExtinctionTimes(tMax,
                        nRuns,
                        populationSize = None,
                        infectionDuration = None,
                        R0 = None,
                        birthSeasonalAmplitude = None,
                        *args,
                        **kwds):

    runFindTransmissionRate = False

    if populationSize is not None:
        parameters.populationSize = populationSize
        runFindTransmissionRate = True

    if infectionDuration is not None:
        parameters.infectionDuration = infectionDuration
        parameters.recovery \
          = parameters.deterministic(scale = infectionDuration)
        runFindTransmissionRate = True

    if R0 is not None:
        parameters.R0 = R0
        runFindTransmissionRate = True

    if runFindTransmissionRate:
        parameters.findTransmissionRate(parameters.R0,
                                        parameters.recovery,
                                        parameters.populationSize)

    if birthSeasonalAmplitude is not None:
        parameters.birthSeasonalAmplitude = birthSeasonalAmplitude
        parameters.birth.findBirthScaling()

    pool = multiprocessing.Pool(initializer = setRandomSeed)
    
    results = [pool.apply_async(doOne,
                                (tMax, ) + args,
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
    stats = {}
    
    stats['median'] = numpy.median(extinctionTimes)
    stats['mean'] = numpy.mean(extinctionTimes)
        
    stats['q_90'] = ppf(extinctionTimes, 0.9)
    stats['q_95'] = ppf(extinctionTimes, 0.95)
    stats['q_99'] = ppf(extinctionTimes, 0.99)

    stats['proportion >= 1'] = proportion_ge_x(extinctionTimes, 1.)
    stats['proportion >= 10'] = proportion_ge_x(extinctionTimes, 10.)
    
    # Convert everything from years to days
    for (k, v) in stats.iteritems():
        stats[k] = 365. * v

    return stats
    
def showStats(stats):
    print ('stats: {'
           + ',\n        '.join(['{} = {}'.format(k, v)
                                 for (k, v) in stats.iteritems()])
           + '}')


Key = collections.namedtuple('parameters', ('populationSize',
                                            'infectionDuration',
                                            'R0'))

def searchParameters(tMax,
                     nRuns,
                     populationSize = parameters.populationSize,
                     infectionDuration = parameters.infectionDuration,
                     R0 = parameters.R0,
                     callback = None,
                     *args, **kwds):
    numpy.random.seed(1)

    eT = findExtinctionTimes(tMax,
                             nRuns,
                             populationSize,
                             infectionDuration,
                             R0,
                             *args,
                             **kwds)

    if callable(callback):
        callback(populationSize, infectionDuration, R0, eT)

    return eT


if __name__ == '__main__':
    nRuns = 100
    tMax = 10.
    debug = False
    
    numpy.random.seed(1)

    eT = findExtinctionTimes(tMax, nRuns, debug = debug)

    stats = findStats(eT)
    showStats(stats)
