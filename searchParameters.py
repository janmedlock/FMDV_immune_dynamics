#!/usr/bin/python

import sys
import os.path
import csv
import numpy
import parameters
import extinctionTimes


parameters.infectionDuration = 21. / 365.
parameters.R0 = 10.


# populationSizes = (100, 150, 200, 250, 300, 350, 400, 500, 600, 700,
#                    800, 900, 1000, 2000, 3000, 5000, 7500, 10000)
populationSizes = (100, )

birthSeasonalVariances = (1., )


(basename, ext) = os.path.splitext(os.path.basename(sys.argv[0]))
filename = basename + '.csv'


def searchPopulationSize(tMax, nRuns,
                         callback = None, debug = False):
    for populationSize in populationSizes:
        print 'populationSize = {}'.format(populationSize)
        extinctionTimes.searchParameters(
            tMax,
            nRuns,
            populationSize = populationSize,
            callback = callback,
            debug = debug)


def searchBirthSeasonalVariance(tMax, nRuns,
                         callback = None, debug = False):
    for birthSeasonalVariance in birthSeasonalVariances:
        print 'birthSeasonalVariance = {}'.format(birthSeasonalVariance)
        extinctionTimes.searchParameters(
            tMax,
            nRuns,
            birthSeasonalVariance = birthSeasonalVariance,
            callback = callback,
            debug = debug)


def SaveOut(csvWriter):
    def _SaveOut(populationSize, birthSeasonalVariance, data):
        csvWriter.writerow([populationSize, birthSeasonalVariance]
                           + data)
    return _SaveOut
    

if __name__ == '__main__':
    nRuns = 100
    tMax = numpy.inf
    debug = False

    w = csv.writer(open(filename, 'w'))
    saveOut = SaveOut(w)
    saveOut('populationSize',
            ['extinctionTimes (years)'])

    searchPopulationSize(tMax, nRuns,
                         callback = saveOut, debug = debug)
