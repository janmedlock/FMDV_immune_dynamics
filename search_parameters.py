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

    parameters = herd.Parameters()

    population_sizes = (100, 200, 500, 1000)
    maternal_immunity_durations = (3 / 12, 6 / 12, 9 / 12)

    nruns = 100
    tmax = numpy.inf
    debug = False

    for ps in population_sizes:
        parameters.population_size = ps
        search_parameter('maternal_immunity_duration',
                         maternal_immunity_durations,
                         nruns,
                         parameters,
                         tmax,
                         debug = debug)
