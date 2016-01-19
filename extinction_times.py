#!/usr/bin/python3

import numpy

import run_many


def find_extinction_times(nruns,
                          parameters,
                          tmax,
                          *args,
                          **kwds):
    data = run_many.run_many(nruns, parameters, tmax, *args, **kwds)

    (T, X) = zip(*(zip(*d) for d in data))

    extinction_times = [t[-1] if (x[-1][2] == 0) else None
                       for (t, x) in zip(T, X)]

    return extinctionTimes


def ppf(D, q, a = 0):
    Daug = numpy.asarray(sorted(D) + [a])
    indices = numpy.ceil(numpy.asarray(q) * len(D) - 1).astype(int)
    return Daug[indices]

def proportion_ge_x(D, x):
    return float(len(numpy.compress(numpy.asarray(D) >= x, D))) / float(len(D))

def find_stats(extinction_times):
    mystats = {}
    
    mystats['median'] = numpy.median(extinction_times)
    mystats['mean'] = numpy.mean(extinction_times)
        
    mystats['q_90'] = ppf(extinction_times, 0.9)
    mystats['q_95'] = ppf(extinction_times, 0.95)
    mystats['q_99'] = ppf(extinction_times, 0.99)

    mystats['proportion >= 1'] = proportion_ge_x(extinction_times, 1)
    mystats['proportion >= 10'] = proportion_ge_x(extinction_times, 10)
    
    return mystats
    
def show_stats(mystats):
    print('stats: {'
          + ',\n        '.join(['{} = {}'.format(k, v)
                                for (k, v) in mystats.items()])
          + '}')


if __name__ == '__main__':
    import herd

    p = herd.Parameters()

    p.population_size = 100
    p.infection_duration = 21 / 365
    p.R0 = 10
    p.birth_seasonal_coefficient_of_variation = 1

    nruns = 10000
    tmax = 5
    debug = False
    
    extinction_times = find_extinction_times(nruns, p, tmax, debug = debug)

    mystats = find_stats(extinction_times)
    show_stats(mystats)
