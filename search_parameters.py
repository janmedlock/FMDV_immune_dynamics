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

    w = csv.writer(open(filename, 'a'))

    paramkeys = sorted(parameters.__dict__.keys())

    # Write header.
    # w.writerow(paramkeys + ['extinction_times (years)'])

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

    parameters.recovery_infection_duration = 21 / 365
    parameters.R0 = 10

    population_sizes = (1000, 2000, 5000, 10000)

    nruns = 100
    tmax = numpy.inf
    debug = False

    for ps in population_sizes:
        parameters.population_size = ps

        # birth_seasonal_coefficient_of_variation calculated from gap_sizes
        gap_sizes = [None] + list(range(12)) # In months.  None is aseasonal.
        birth_seasonal_coefficient_of_variations = map(
            birth.get_seasonalcoefficient_of_variation_from_gap_size,
            gap_sizes)

        search_parameter('birth_seasonal_coefficient_of_variation',
                         birth_seasonal_coefficient_of_variations,
                         nruns,
                         parameters,
                         tmax,
                         debug = debug)
