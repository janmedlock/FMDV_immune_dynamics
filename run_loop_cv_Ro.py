#!/usr/bin/python3

import numpy
import functools
import multiprocessing
import pandas

import herd

export_data = True
test= True

def run_one(parameters, tmax, *args, **kwds):
    '''
    Run one simulation, among multiple running in parallel.
    run_number is at the end of *args to allow easy use of
    multiprocessing.Pool().map() with functools.partial().
    '''
    run_number = args[-1]
    args_other = args[ : -1]
    return herd.Herd(parameters, *args_other,
                     run_number = run_number, **kwds).run(tmax)


def run_many(nruns, parameters, tmax, *args, **kwds):
    'Run many simulations in parallel.'

    # Build the RVs once to make sure the caches are seeded.
    rvs = herd.RandomVariables(parameters)

    # Set the random seed on each worker to a different seed.
    # numpy.random.seed() with no argument uses /dev/urandom to seed.
    with multiprocessing.Pool(initializer = numpy.random.seed) as pool:
        # f is _doOne with all the args and kwargs set, except runNumber,
        # which will be appended to args.
        f = functools.partial(run_one, parameters, tmax, *args, **kwds)

        # f(0), f(1), f(2), ..., f(nruns - 1),
        # computed in parallel.
        return pool.map(f, range(nruns))


def get_mean(T, X):
    T_mean = numpy.unique(numpy.hstack(T))
    X_mean = numpy.zeros((len(T_mean), len(X[0][0])))
    n = numpy.zeros_like(T_mean)
    for (Tk, Xk) in zip(T, X):
        Tk = numpy.array(Tk)
        Xk = numpy.array(Xk)

        # Only go to the end of this simulation.
        T_ = T_mean.compress(T_mean <= Tk[-1])

        # Find the indicies i[j] of the largest Tk with Tk[i[j]] <= T_[j]
        indices = [(Tk <= t).nonzero()[0][-1] for t in T_]

        X_mean[ : len(T_)] += Xk[indices]
        n[ : len(T_)] += 1
    X_mean /= n[:, numpy.newaxis]

    return (T_mean, X_mean)


def make_datasheet(data, b, n):
	# Saves one datasheet per parameter combination
	# Each datasheet is has multiple simulations in long format
    # Make datasheets (manyruns_data.csv) for each iteration
    (T, X) = zip(*(zip(*d) for d in data))
    index = 0
    appended_data = []
    bval = b
    nval = n
    for (t, x) in zip(T, X):
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total and index
        n = x.sum(-1)
        rep = numpy.array([index]* len(t))
        b = numpy.array([bval]*len(t))
        # doesn't like this one
        x = numpy.column_stack((x, n, rep, b))
        data = pandas.DataFrame(data=x, index=t, columns=['M', 'S', 'I', 'R', 'Total', 'Rep', 'b'])
        appended_data.append(data)  
        index += 1      
    # Make a datasheet with the mean values of all the interations
    (T_mean, X_mean) = get_mean(T, X)
    N_mean = X_mean.sum(-1)
    rep2 = numpy.array(["mean"]* len(T_mean))
    b2 = numpy.array([bval]*len(T_mean))
    X_mean = numpy.column_stack((X_mean, N_mean, rep2, b2))
    mean_data = pandas.DataFrame(data=X_mean, index=T_mean, columns=['M', 'S', 'I', 'R', 'Total', 'Rep', 'b'])
    appended_data.append(mean_data)
    # Append them together in long format and save
    final_data = pandas.concat(appended_data)          
    final_data.to_csv("manyruns_data_cv_Ro_"+str(bval)+ "_" + str(nval) + ".csv", sep=',')




if __name__ == '__main__':
    import time

    numpy.random.seed(1)
    if test:
        birth_coefficient_values= numpy.arange(0.5, 0.61, 0.1)
        pop_size = numpy.arange(1050, 2000.1, 25)
        
    else:
        birth_coefficient_values = numpy.arange(0.2, 1.6, 0.05)
        Ro = numpy.arange(0.5, 20.2, 0.1)

	# test should run 4 loops
    for b in  birth_coefficient_values:
        p = herd.Parameters()
        p.birth_seasonal_coefficient_of_variation = b
        tmax = 1
        nruns = 1000
        debug = False
        print("Seasonal coefficient: ", p.birth_seasonal_coefficient_of_variation, sep="")
        for r in Ro:
            t0 = time.time()        
            p.R0 = r
            if test:
                print("Ro = ", p.R0)
            data = run_many(nruns, p, tmax, debug = debug)
            t1 = time.time()
            print('Run time: {} seconds.'.format(t1 - t0))
            if export_data:
                make_datasheet(data, b, n)
