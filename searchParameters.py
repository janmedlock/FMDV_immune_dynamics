#!/usr/bin/python

import sys
import os.path
import csv
import numpy
import extinctionTimes


populationSizes = (100, 150, 200, 250, 300, 350, 400, 500, 600, 700,
                   800, 900, 1000, 2000, 3000, 5000, 7500, 10000)

infectionDurations = (1.6, 2., 2.5, 3., 5., 7.5, 10., 12.5,
                      15., 17.5, 20., 21.)
# Convert from days to years
infectionDurations = tuple(iD / 365. for iD in infectionDurations)

R0s = (1.2, 1.5, 1.8, 2., 2.5, 3., 3.5, 4., 5., 6.,
       7., 8., 9., 10., 12., 15., 18., 20., 25., 30.)


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


def searchInfectionDuration(tMax, nRuns,
                            callback = None, debug = False):
    for infectionDuration in infectionDurations:
        print 'infectionDuration = {} days'.format(infectionDuration * 365.)
        extinctionTimes.searchParameters(
            tMax,
            nRuns,
            infectionDuration = infectionDuration,
            callback = callback,
            debug = debug)


def searchR0(tMax, nRuns,
             callback = None, debug = False):
    for R0 in R0s:
        print 'R0 = {}'.format(R0)
        extinctionTimes.searchParameters(
            tMax,
            nRuns,
            R0 = R0,
            callback = callback,
            debug = debug)


def searchAll(tMax, nRuns,
             callback = None, debug = False):
    for populationSize in populationSizes:
        for infectionDuration in infectionDurations:
            for R0 in R0s:
                print 'populationSize = {}, ' \
                  + 'infectionDuration = {} days, ' \
                  + 'R0 = {}'.format(populationSize,
                                     infectionDuration * 365.,
                                     R0)
                extinctionTimes.searchParameters(
                    tMax,
                    nRuns,
                    populationSize = populationSize,
                    infectionDuration = infectionDuration,
                    R0 = R0,
                    callback = callback,
                    debug = debug)


def SaveOut(csvWriter):
    def _SaveOut(populationSize, infectionDuration, R0, data):
        csvWriter.writerow([populationSize, infectionDuration, R0]
                           + data)
    return _SaveOut
    

if __name__ == '__main__':
    nRuns = 100
    tMax = numpy.inf
    debug = False

    w = csv.writer(open(filename, 'w'))
    saveOut = SaveOut(w)
    saveOut('populationSize', 'infectionDuration (years)', 'R0',
            ['extinctionTimes (years)'])

    searchAll(tMax, nRuns,
              callback = saveOut, debug = debug)
