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
    fd = open(filename, 'a')
    w = csv.writer(fd)

    paramkeys = sorted(parameters.__dict__.keys())

    if new:
        # Write header.
        w.writerow(paramkeys + ['extinction_times (years)'])
        fd.flush()

    for v in values:
        setattr(parameters, parameter_name, v)
        print('{} = {}'.format(parameter_name, v))
        ets = extinction_times.find_extinction_times(nruns, parameters, tmax,
                                                     *args, **kwargs)
        w.writerow([getattr(parameters, k) for k in paramkeys]
                   + ets)
        fd.flush()

    fd.close()


if __name__ == '__main__':
    import numpy

    population_sizes = (100, 200, 500, 1000)
    birth_seasonal_coefficients_of_variation = (
        0.61 * numpy.array([1, 0.75, 0.5, 2, 3, 0.25, 4, 0.1, 0]))

    nruns = 1000
    tmax = 10
    debug = False

    for bscov in birth_seasonal_coefficients_of_variation:
        for SAT in (1, 2, 3):
            parameters = herd.Parameters(SAT = SAT)
            parameters.birth_seasonal_coefficient_of_variation = bscov
            search_parameter('population_size',
                             population_sizes,
                             nruns,
                             parameters,
                             tmax,
                             debug = debug)
