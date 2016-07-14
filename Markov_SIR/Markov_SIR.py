#!/usr/bin/python

import numpy
import multiprocessing
from matplotlib import pyplot

import Markov


class SIR(object):
    populationsize = 10000
    turnoverrate = 1. / 15.
    recoveryrate = 365. / 21.
    R0 = 10.

    def __init__(self, seed = None):
        self.transmissionrate = (self.R0
                                 * (self.recoveryrate + self.turnoverrate)
                                 / self.populationsize)
        
        # Initial population
        # Epidemic
        # I0 = int(round(self.populationsize * 0.01)) # 1% infected
        # S0 = self.populationsize - I0

        # Endemic
        S0 = self.populationsize / self.R0
        I0 = (self.turnoverrate / self.transmissionrate
              * (self.populationsize / S0 - 1.))
        R0 = self.recoveryrate / self.turnoverrate * I0

        # Loop to get positive number of infections.
        while True:
            self.population = numpy.random.multinomial(
                self.populationsize,
                numpy.array([S0, I0, R0]) / self.populationsize)
            if self.population[1] > 0:
                break

        if seed is not None:
            numpy.random.seed(seed)

        (rates, rules) = zip(*self.transitions())
        self.rules = numpy.array(rules)

        # Order of highest-order reaction for species i.
        self.g = numpy.array([2, 2, 1])

    def transitions(self, state = None):
        'Return a tuple of pairs (rate, transition rule).'

        if state is None:
            state = self.population

        (S, I, R) = state

        N = S + I + R

        return (
            (self.turnoverrate * N,         [ 1,  0,  0]), # an S birth
            (self.turnoverrate * S,         [-1,  0,  0]), # an S death
            (self.transmissionrate * S * I, [-1,  1,  0]), # an infection
            (self.turnoverrate * I,         [ 0, -1,  0]), # an I death
            (self.recoveryrate * I,         [ 0, -1,  1]), # a recovery
            (self.turnoverrate * R,         [ 0,  0, -1])  # an R death
        )

    def rates(self, state = None):
        (rates, rules) = zip(*self.transitions(state))
        return numpy.array(rates)

# runSIR allows call of any of the methods
def runSIR(func, *args, **kwds):
    sim = SIR()
    X = sim.population
    v = sim.rules
    rates = sim.rates
    g = sim.g
    return func(X, v, rates, g, *args, **kwds)

def multirunSIR(nruns, func, *args, **kwds):
    pool = multiprocessing.Pool(initializer = numpy.random.seed)

    args_ = (func, ) + args
    results = [pool.apply_async(runSIR, args_, kwds)
               for i in xrange(nruns)]
    pool.close()
    return [r.get() for r in results]

def plotIt(Y, name):
    for (T, X) in Y:
        pyplot.step(T, X[:, 1], where = 'post', alpha = 0.5)

    # (t, X) = Markov.getmean(Y)
    # pyplot.step(t, X[:, 1],
    #             where = 'post')

    pyplot.ylabel(r'{} $I$'.format(name))


def plotIend(Y, name):
    Iend = numpy.sort([X[-1, 1] for (T, X) in Y])
    p = numpy.cumsum(1. / len(Iend) * numpy.ones_like(Iend))

    pyplot.step(numpy.hstack((0., Iend)), numpy.hstack((0., p)),
                where = 'post', label = name)
    

if __name__ == '__main__':
    import time

    t_end = 2.

    # nruns = multiprocessing.cpu_count()
    nruns = 100
    
    time0 = time.time()
    Y_SSA = multirunSIR(nruns, Markov.SSA.run, 0, t_end)
    print 'SSA time = {} seconds.'.format(
        time.time() - time0)

    time0 = time.time()
    Y_tauexp = multirunSIR(nruns, Markov.tauexplicit.run, 0, t_end)
    print 'explicit tau-leaping time = {} seconds.'.format(
        time.time() - time0)

    time0 = time.time()
    Y_tauimp = multirunSIR(nruns, Markov.tauimplicit.run, 0, t_end)
    print 'implicit tau-leaping time = {} seconds.'.format(
        time.time() - time0)

    time0 = time.time()
    Y_tauada = multirunSIR(nruns, Markov.tauadaptive.run, 0, t_end)
    print 'adaptive tau-leaping time = {} seconds.'.format(time.time() - time0)


    pyplot.subplot(4, 1, 1)
    plotIt(Y_SSA, 'SSA')
    pyplot.subplot(4, 1, 2)
    plotIt(Y_tauexp, r'explicit $\tau$ leaping')
    pyplot.subplot(4, 1, 3)
    plotIt(Y_tauimp, r'implicit $\tau$ leaping')
    pyplot.subplot(4, 1, 4)
    plotIt(Y_tauada, r'adaptive $\tau$ leaping')
    pyplot.xlabel(r'$t$')

    pyplot.figure()
    plotIend(Y_SSA, 'SSA')
    plotIend(Y_tauexp, r'explicit $\tau$ leaping')
    plotIend(Y_tauimp, r'implicit $\tau$ leaping')
    plotIend(Y_tauada, r'adaptive $\tau$ leaping')
    pyplot.xlabel(r'$I({:g})$'.format(t_end))
    pyplot.ylabel(r'cumulative density')
    pyplot.legend(loc = 'lower right')
    
    pyplot.show()
