#!/usr/bin/python3

import sys
import os.path
import csv

import herd
from herd import birth
import extinction_times


def search_parameter(parameter_name, values, nruns, parameters, tmax,
                    *args, **kwargs):
    assert hasattr(parameters, parameter_name)

    (basename, ext) = os.path.splitext(os.path.basename(sys.argv[0]))
    filename = basename + '.csv'

    new = not os.path.exists(filename)
    w = csv.writer(open(filename, 'a'))

    paramkeys = sorted(parameters.__dict__.keys())

    if new:
        # Write header.
        w.writerow(paramkeys + ['extinction_times (years)'])

    for v in values:
        setattr(parameters, parameter_name, v)
        print('{} = {}'.format(parameter_name, v))
        ets = extinction_times.find_extinction_times(nruns, parameters, tmax,
                                                     *args, **kwargs)
        w.writerow([getattr(parameters, k) for k in paramkeys]
                   + ets)
    

if __name__ == '__main__':
    import numpy

    population_sizes = (100, 200, 500, 1000)
    birth_seasonal_coefficients_of_variation = (
        0.61 * numpy.array([0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4]))

    nruns = 100
    tmax = numpy.inf
    debug = False

    for SAT in (1, 2, 3):
        parameters = herd.Parameters(SAT = SAT)
        for ps in population_sizes:
            parameters.population_size = ps
            search_parameter('birth_seasonal_coefficient_of_variation',
                             birth_seasonal_coefficients_of_variation,
                             nruns,
                             parameters,
                             tmax,
                             debug = debug)
