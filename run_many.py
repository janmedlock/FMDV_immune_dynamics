#!/usr/bin/python3

import numpy
import functools
import multiprocessing
import pandas

import herd

export_data = True

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


def make_plots(data, show = True):
    from matplotlib import pyplot
    import seaborn
    import itertools
    from scipy import integrate

    from herd import pde

    (fig, ax) = pyplot.subplots(6, sharex = True)
    #(fig, ax) = pyplot.subplots(5, sharex = True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))

    (T, X) = zip(*(zip(*d) for d in data))
    for (t, x) in zip(T, X):
        c = next(colors)
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total.
        # n = x.sum(-1)
        # x = numpy.column_stack((x, n))
        for j in range(x.shape[-1]):
            ax[j].step(365 * t, x[:, j], where = 'post',
                       color = c, alpha = 0.5)

    (T_mean, X_mean) = get_mean(T, X)
    # Add column for total.
    # N_mean = X_mean.sum(-1)
    # X_mean = numpy.column_stack((X_mean, N_mean))
    for j in range(X_mean.shape[-1]):
        ax[j].step(365 * T_mean, X_mean[:, j], where = 'post',
                   color = 'black', alpha = 0.6)

    # (t_, a_, X_) = pde.solve(20, 20, 0.01, p)
    # x_ = numpy.zeros((len(X_), len(t_)))
    # for j in range(len(X_)):
    #     x_[j] = integrate.trapz(X_[j], a_, axis = 1)
    # # Add column for total.
    # n_ = x_.sum(0)
    # x_ = numpy.row_stack((x_, n_))
    # Tmax_ = numpy.ceil(numpy.max(numpy.hstack(T)))
    # dt_ = t_[1] - t_[0]
    # j_ = int(- Tmax_ / dt_ - 1)
    # t_ -= t_[j_]
    # for k in range(len(x_)):
    #     ax[k].plot(365 * t_[j_ : ], x_[k, j_ : ] / n_[j_] * p.population_size,
    #                linestyle = ':', color = 'black')

    ax[0].set_ylabel('maternal\nimmunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('exposed')
    ax[3].set_ylabel('infected')
    ax[4].set_ylabel('chronic')
    ax[5].set_ylabel('recovered')
    # ax[6].set_ylabel('total')

    ax[-1].set_xlabel('time (days)')

    for ax_ in ax:
        yl = ax_.get_ylim()
        if yl[0] < 0:
            ax_.set_ylim(ymin = 0)

    if show:
        pyplot.show()


def make_datasheet(data):
    # Make datasheets (manyruns_data.csv) for each iteration
    (T, X) = zip(*(zip(*d) for d in data))
    index = 0
    appended_data = []
    for (t, x) in zip(T, X):
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total and index
        n = x.sum(-1)
        rep = numpy.array([index]* len(t))
        x = numpy.column_stack((x, n, rep))
        data = pandas.DataFrame(data=x, index=t, columns=['M', 'S', 'E', 'I', 'C', 'R', 'Total', 'Rep'])
        appended_data.append(data)  
        index += 1      
    # Make a dataheet with the mean values of all the interations
    (T_mean, X_mean) = get_mean(T, X)
    N_mean = X_mean.sum(-1)
    rep = numpy.array(["mean"]* len(T_mean))
    X_mean = numpy.column_stack((X_mean, N_mean, rep))
    mean_data = pandas.DataFrame(data=X_mean, index=T_mean, columns=['M', 'S', 'E', 'I', 'C', 'R', 'Total', 'Rep'])
    appended_data.append(mean_data)
    # Append them together in long format and save
    final_data = pandas.concat(appended_data)          
    final_data.to_csv("manyruns_data.csv", sep=',')

def get_I_terminal(data):
    tX_terminal = (d[-1] for d in data)
    X_terminal = (tX[-1] for tX in tX_terminal)
    I_terminal = (X[2] for X in X_terminal)
    return numpy.fromiter(I_terminal, numpy.int, len(data))


if __name__ == '__main__':
    import time

    numpy.random.seed(1)

    p = herd.Parameters(SAT = 1)

    tmax = 1
    nruns = 100
    debug = False
    
    t0 = time.time()
    data = run_many(nruns, p, tmax, debug = debug)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))
    
    make_plots(data)
    if export_data:
        make_datasheet(data)
