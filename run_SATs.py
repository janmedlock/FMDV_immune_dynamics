#!/usr/bin/python3

import functools
import itertools
import multiprocessing

import numpy
import pandas

import herd
import run_many

export_data = True

def make_datasheet(data):
    # Make datasheets (manyruns_data.csv) for each iteration
    (T, X) = zip(*(zip(*d) for d in data))
    appended_data = []
    for (i, tx) in enumerate(zip(T, X)):
        t, x = map(numpy.array, tx)
        # Add column for total
        x = numpy.column_stack((x, x.sum(-1)))
        ix = pandas.MultiIndex.from_tuples(([i] * len(t), t),
                                           names = ('rep', 'time'))
        data = pandas.DataFrame(
            data = x,
            index = ix,
            columns = ['M', 'S', 'E', 'I', 'R', 'Total'])
        appended_data.append(data)  
    # Make a dataheet with the mean values of all the interations
    (T_mean, X_mean) = get_mean(T, X)
    X_mean = numpy.column_stack((X_mean, X_mean.sum(-1)))
    ix = pandas.MultiIndex.from_tuples((['mean'] * len(t), t),
                                       names = ('rep', 'time'))
    mean_data = pandas.DataFrame(data = X_mean,
                                 index = ix,
                                 columns = data.columns)
    appended_data.append(mean_data)
    # Append them together in long format and save
    final_data = pandas.concat(appended_data)          
    final_data.to_csv("manyruns_data.csv", sep=',')


if __name__ == '__main__':
    import time

    numpy.random.seed(1)

    tmax = 1
    nruns = 1
    debug = False
    
    data = []
    for SAT in (1, 2, 3):
        print('Running SAT {}.'.format(SAT))
        p = herd.Parameters(SAT = SAT)

        t0 = time.time()
        data.append(run_many.run_many(nruns, p, tmax, debug = debug))
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    
    if export_data:
        # make_datasheet(data)
        pass
