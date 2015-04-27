#!/usr/bin/python

import Parameters
import extinctionTimes


def searchPopulationSize(populationSizes, parameters, tMax, nRuns,
                         callback = None, debug = False):
    for x in populationSizes:
        parameters.populationSize = x
        print 'populationSize = {}'.format(parameters.populationSize)
        extinctionTimes.searchParameters(
            parameters,
            tMax,
            nRuns,
            callback = callback,
            debug = debug)


def searchBirthSeasonalVariance(birthSeasonalVariances, parameters, tMax, nRuns,
                                callback = None, debug = False):
    for x in birthSeasonalVariances:
        parameters.birthSeasonalVariance = x

        print 'birthSeasonalVariance = {}'.format(
            parameters.birthSeasonalVariance)

        extinctionTimes.searchParameters(
            parameters,
            tMax,
            nRuns,
            callback = callback,
            debug = debug)


def SaveOut(csvWriter):
    # Write header.
    csvWriter.writerow(['populationSize', 'birthSeasonalVariance',
                        'extinctionTimes (years)'])
    def _SaveOut(parameters, data):
        csvWriter.writerow([parameters.populationSize,
                            parameters.birthSeasonalVariance]
                           + data)
    return _SaveOut
    

if __name__ == '__main__':
    import sys
    import os.path
    import csv
    import numpy


    parameters = Parameters.Parameters()

    parameters.infectionDuration = 21. / 365.
    parameters.R0 = 10.

    # populationSizes = (100, 150, 200, 250, 300, 350, 400, 500, 600, 700,
    #                    800, 900, 1000, 2000, 3000, 5000, 7500, 10000)
    # populationSizes = (100, 150, 200, 250, 300, 350, 400, 500, 600, 700,
    #                    800, 900, 1000)
    parameters.populationSize = 10000

    birthSeasonalVariances = numpy.linspace(0., 1., 7)

    nRuns = 100
    # tMax = numpy.inf
    tMax = 10.
    debug = False

    (basename, ext) = os.path.splitext(os.path.basename(sys.argv[0]))
    filename = basename + '.csv'

    w = csv.writer(open(filename, 'w', 0)) # 0 for unbuffered.
    saveOut = SaveOut(w)

    searchBirthSeasonalVariance(birthSeasonalVariances, parameters, tMax, nRuns,
                                callback = saveOut, debug = debug)
