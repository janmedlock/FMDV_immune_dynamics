#!/usr/bin/python3

import functools
import multiprocessing

import numpy
import pandas

import herd
import run_many

export_data = False

def _build_ix(SAT, rep, t):
    return pandas.MultiIndex.from_arrays(([SAT] * len(t), [rep] * len(t), t),
                                         names = ('SAT', 'rep', 'time'))
    

def run_SATs(nruns = 100, tmax = numpy.inf, debug = False):
    cols = ['M', 'S', 'E', 'I', 'C', 'R', 'Total']

    sheets = []
    for SAT in (1, 2, 3):
        p = herd.Parameters(SAT = SAT)

        print('Running SAT {}.'.format(SAT))
        t0 = time.time()
        data = run_many.run_many(nruns, p, tmax, debug = debug)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    
        (T, X) = zip(*(zip(*d) for d in data))
        for (i, tx) in enumerate(zip(T, X)):
            t, x = map(numpy.array, tx)
            # Add column for total
            x = numpy.column_stack((x, x.sum(-1)))
            ix = _build_ix(SAT, i, t)
            sheets.append(pandas.DataFrame(x,
                                           index = ix,
                                           columns = cols))
        # Make a dataheet with the mean values of all the interations
        (t, x) = run_many.get_mean(T, X)
        x = numpy.column_stack((x, x.sum(-1)))
        ix = _build_ix(SAT, 'mean', t)
        sheets.append(pandas.DataFrame(x,
                                       index = ix,
                                       columns = cols))

    # Append them together in long format and save
    df = pandas.concat(sheets)
    return df


if __name__ == '__main__':
    import time

    numpy.random.seed(1)

    data = run_SATs()

    if export_data:
        data.to_csv("run_SATs.csv", sep=',')
