#!/usr/bin/python3

import numpy
import pandas

import h5
import run_common


def _get_index_levels_unique(store, levels, *args, key=None, **kwargs):
    to_drop = None  # Build this on the first iteration.
    unique = pandas.MultiIndex.from_tuples([], names=levels)
    for chunk in store.get_index(*args, key=key, iterator=True, **kwargs):
        if to_drop is None:
            to_drop = [n for n in chunk.names if n not in levels]
        unique = unique.union(chunk.droplevel(to_drop)
                                   .drop_duplicates()
                                   .reorder_levels(levels))
    return unique


def _get_extinction(infected, tmax=10):
    t = infected.index.get_level_values('time (y)')
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    assert observed or (time == tmax)
    return dict(time=time, observed=observed)


def load_extinction_times(filename, by):
    columns = ['exposed', 'infectious', 'chronic']
    extinction = {}
    with h5.HDFStore(filename, mode='r') as store:
        index = _get_index_levels_unique(store, by)
        for ix in index:
            where = ' & '.join(f'{b}={i}' for (b, i) in zip(by, ix))
            print(where.replace(' & ', ', '))
            df = store.select(where=where, columns=columns)
            infected = df.sum(axis='columns')
            extinction[ix] = _get_extinction(infected)
    extinction = pandas.DataFrame.from_dict(extinction,
                                            orient='index')
    extinction.index.names = by
    extinction.sort_index(inplace=True)
    return extinction


def number_infected(x):
    M, S, E, I, R = x
    return (E + I)


def find_extinction_times(parameters, tmax, nruns, *args, **kwds):
    data = run_common.run_many(parameters, tmax, nruns, *args, **kwds)
    (T, X) = zip(*(zip(*d) for d in data))
    extinction_times = [t[-1] if (number_infected(x[-1]) == 0) else None
                        for (t, x) in zip(T, X)]
    return extinction_times


def ppf(D, q, a=0):
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

    SAT = 1
    model = 'chronic'

    p = herd.Parameters(model=model, SAT=SAT)

    p.population_size = 100
    p.infection_duration = 21 / 365
    p.birth_seasonal_coefficient_of_variation = 1

    nruns = 10000
    tmax = 10
    debug = False

    extinction_times = find_extinction_times(p, tmax, nruns, debug = debug)

    mystats = find_stats(extinction_times)
    show_stats(mystats)
