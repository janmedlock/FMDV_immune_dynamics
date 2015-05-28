#!/usr/bin/python

import sys
import os.path
import csv

import Parameters
from Parameters import birth
import extinctionTimes


def searchParameter(parameterName, values, nRuns, parameters, tMax,
                    *args, **kwargs):
    assert hasattr(parameters, parameterName)

    (basename, ext) = os.path.splitext(os.path.basename(sys.argv[0]))
    filename = basename + '.csv'

    w = csv.writer(open(filename, 'a', 0)) # 0 for unbuffered.

    paramkeys = sorted(parameters.__dict__.keys())

    # Write header.
    # w.writerow(paramkeys + ['extinctionTimes (years)'])

    for v in values:
        setattr(parameters, parameterName, v)
        print '{} = {}'.format(parameterName, v)
        eT = extinctionTimes.findExtinctionTimes(nRuns, parameters, tMax,
                                                 *args, **kwargs)

        w.writerow([getattr(parameters, k) for k in paramkeys]
                   + eT)
    

if __name__ == '__main__':
    import numpy

    parameters = Parameters.Parameters()

    parameters.infectionDuration = 21. / 365.
    parameters.R0 = 10.

    # populationSizes = (100, 150, 200, 250, 300, 350, 400, 500, 600, 700,
    #                    800, 900, 1000, 2000, 3000, 5000, 7500, 10000)
    parameters.populationSize = 10000

    # birthSeasonalVariance calculated from gapSizes
    gapSizes = [None] + range(12) # In months.  None is aseasonal.
    birthSeasonalVariances = map(birth.getSeasonalVarianceFromGapSize,
                                 gapSizes)

    nRuns = 100
    # tMax = numpy.inf
    tMax = 5.
    debug = False

    searchParameter('birthSeasonalVariance',
                    birthSeasonalVariances,
                    nRuns,
                    parameters,
                    tMax,
                    debug = debug)
