#!/usr/bin/python

import numpy
from scipy import stats
import multiprocessing

import Parameters


class Event:
    def __init__(self, time, func, label):
        self.time = time
        self.func = func
        self.label = label

    def __cmp__(self, other):
        return cmp(self.time, other.time)

    def __call__(self):
        return self.func()


class Buffalo:
    def __init__(self, herd, immuneStatus = 'maternal immunity', age = 0.,
                 identifier = None):
        self.herd = herd
        self.immuneStatus = immuneStatus

        # All members of the herd have the same parameters.
        self.RVs = self.herd.RVs

        self.birthDate = self.herd.time - age
        self.identifier = identifier
        self.sex = 'male' if (self.RVs.male.rvs() == 1) \
          else 'female'

        self.events = {}

        if self.immuneStatus == 'maternal immunity':
            eventTime = self.birthDate + self.RVs.maternalImmunityWaning.rvs()
            assert eventTime >= 0.
            self.events['maternalImmunityWaning'] = Event(
                    eventTime,
                    self.maternalImmunityWaning,
                    'maternal immunity waning for #{}'.format(self.identifier))
        elif self.immuneStatus == 'susceptible':
            pass
        elif self.immuneStatus == 'infectious':
            self.events['recovery'] \
                = Event(self.herd.time + self.RVs.recovery.rvs(),
                        self.recovery,
                        'recovery for #{}'.format(self.identifier))
        elif self.immuneStatus == 'recovered':
            pass
        else:
            raise ValueError('Unknown immuneStatus = {}!'.format(
                self.immuneStatus))

        # Use resampling to get a death age > current age.
        while True:
            deathAge = self.RVs.mortality.rvs()
            if deathAge > age:
                break
        self.events['mortality'] = Event(self.birthDate + deathAge,
                                         self.mortality,
                                         'mortality for #{}'.format(
                                             self.identifier))

        if self.sex == 'female':
            self.events['giveBirth'] \
              = Event(self.herd.time
                      + self.RVs.birth.rvs(self.herd.time, age),
                      self.giveBirth,
                      'give birth for #{}'.format(self.identifier))

    def age(self):
        return self.herd.time - self.birthDate

    def mortality(self):
        self.herd.mortality(self)

    def giveBirth(self):
        if self.immuneStatus == 'recovered':
            calfStatus = 'maternal immunity'
        else:
            calfStatus = 'susceptible'

        self.herd.birth(immuneStatus = calfStatus)
        self.events['giveBirth'] \
          = Event(
              self.herd.time
              + self.RVs.birth.rvs(self.herd.time, self.age()),
              self.giveBirth,
              'give birth for #{}'.format(self.identifier))

    def maternalImmunityWaning(self):
        assert self.immuneStatus == 'maternal immunity'
        self.immuneStatus = 'susceptible'
        try:
            del self.events['maternalImmunityWaning']
        except KeyError:
            pass

    def infection(self):
        assert self.isSusceptible()
        self.immuneStatus = 'infectious'
        try:
            del self.events['infection']
        except KeyError:
            pass
        
        self.events['recovery'] \
          = Event(self.herd.time
                  + self.RVs.recovery.rvs(),
                  self.recovery,
                  'recovery for #{}'.format(self.identifier))
    
    def recovery(self):
        assert self.isInfectious()
        self.immuneStatus = 'recovered'
        try:
            del self.events['recovery']
        except KeyError:
            pass
    
    def getNextEvent(self):
        return min(self.events.itervalues())

    def isSusceptible(self):
        return self.immuneStatus == 'susceptible'

    def isInfectious(self):
        return self.immuneStatus == 'infectious'

    ## Fix me! ##
    def updateInfectionTime(self, forceOfInfection):
        if self.isSusceptible():
            if (forceOfInfection > 0.):
                infectionTime = stats.expon.rvs(scale = 1. / forceOfInfection)
            
                self.events['infection'] = Event(
                    self.herd.time + infectionTime,
                    self.infection,
                    'infection for #{}'.format(self.identifier))
            else:
                try:
                    del self.events['infection']
                except KeyError:
                    pass


class Herd(list):
    def __init__(self, parameters, debug = False, runNumber = None):
        self.parameters = parameters

        self.debug = debug
        self.runNumber = runNumber

        self.RVs = Parameters.RandomVariables(self.parameters)

        self.time = 0.
        self.identifier = 0

        # Loop until we get a non-zero number of initial infections.
        while True:
            status_ages = self.RVs.endemicEquilibrium.rvs(
                self.parameters.populationSize)
            if len(status_ages['infectious']) > 0:
                break
            else:
                print 'Initial infections = 0!  Re-sampling initial conditions.'

        for (immuneStatus, ages) in status_ages.items():
            for age in ages:
                self.birth(immuneStatus, age)

    def mortality(self, buffalo):
        self.remove(buffalo)

    def birth(self, immuneStatus = 'maternal immunity', age = 0.):
        if self.debug:
            if age > 0:
                print 't = {}: arrival of #{} at age {} with status {}'.format(
                    self.time,
                    self.identifier,
                    age,
                    immuneStatus)
            else:
                print 't = {}: birth of #{} with status {}'.format(
                    self.time,
                    self.identifier,
                    immuneStatus)

        self.append(Buffalo(self, immuneStatus, age,
                            identifier = self.identifier))
        self.identifier += 1

    def updateInfectionTimes(self):
        self.numberInfectious = sum(buffalo.isInfectious() for buffalo in self)
        self.forceOfInfection \
          = self.RVs.transmissionRate * self.numberInfectious
        for buffalo in self:
            buffalo.updateInfectionTime(self.forceOfInfection)

    def getStats(self):
        counts = []
        for status in ('maternal immunity', 'susceptible',
                       'infectious', 'recovered'):
            counts.append(sum(1 for buffalo in self
                              if buffalo.immuneStatus == status))
        return [self.time, counts]

    def getNextEvent(self):
        if len(self) > 0:
            return min([b.getNextEvent() for b in self])
        else:
            return None

    def stop(self):
        try:
            return (self.numberInfectious == 0)
        except AttributeError:
            # If self.numberInfectious isn't defined yet.
            return False

    def step(self, tMax = numpy.inf):
        self.updateInfectionTimes()
        event = self.getNextEvent()

        if (event is not None) and (event.time < tMax):
            if self.debug:
                print 't = {}: {}'.format(event.time, event.label)
            self.time = event.time
            event()
        else:
            self.time = tMax

    def run(self, tMax):
        result = [self.getStats()]

        while (self.time < tMax) and (not self.stop()):
            self.step(tMax)
            result.append(self.getStats())

        if self.runNumber is not None:
            return (self.runNumber, result)
        else:
            return result

    def findExtinctionTime(self, tMax):
        result = self.run(tMax)
        return result[-1][0]



def doOne(parameters, tMax, *args, **kwds):
    return Herd(parameters, *args, **kwds).run(tMax)

def showResult(y):
    (runNumber, x) = y
    state_last = x[-1]
    t_last = state_last[0]

    print 'Simulation #{} ended at {:g} days.'.format(runNumber,
                                                      365. * t_last)

def getkwds(kwds, i):
    res = kwds.copy()
    res['runNumber'] = i
    return res

def multirun(nRuns, parameters, *args, **kwds):
    # Build the RVs once to make sure the caches are seeded.
    RVs = Parameters.RandomVariables(parameters)

    pool = multiprocessing.Pool(initializer = numpy.random.seed)
    
    results = [pool.apply_async(doOne,
                                (parameters, ) + args,
                                getkwds(kwds, i),
                                showResult)
               for i in xrange(nRuns)]

    pool.close()

    return [r.get() for r in results]


def getMean(T, X):
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
        n[ : len(T_)] += 1.
    X_mean /= n[:, numpy.newaxis]

    return (T_mean, X_mean)


def makePlots(data, show = True):
    from matplotlib import pyplot
    import seaborn
    import itertools
    from scipy import integrate

    from Parameters import pde

    (fig, ax) = pyplot.subplots(5, sharex = True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))

    (T, X) = zip(*(zip(*y) for (runNumber, y) in data))
    for (t, x) in zip(T, X):
        c = colors.next()
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total.
        n = x.sum(-1)
        x = numpy.column_stack((x, n))
        for j in range(x.shape[-1]):
            ax[j].step(365. * t, x[:, j], where = 'post',
                       color = c, alpha = 0.5)

    (T_mean, X_mean) = getMean(T, X)
    # Add column for total.
    N_mean = X_mean.sum(-1)
    X_mean = numpy.column_stack((X_mean, N_mean))
    for j in range(X_mean.shape[-1]):
        ax[j].step(365. * T_mean, X_mean[:, j], where = 'post',
                   color = 'black')

    (t_, a_, X_) = pde.solve(20, 20, 0.01, p)
    x_ = numpy.zeros((len(X_), len(t_)))
    for j in range(len(X_)):
        x_[j] = integrate.trapz(X_[j], a_, axis = 1)
    # Add column for total.
    n_ = x_.sum(0)
    x_ = numpy.row_stack((x_, n_))
    Tmax_ = numpy.ceil(numpy.max(numpy.hstack(T)))
    dt_ = t_[1] - t_[0]
    j_ = int(- Tmax_ / dt_ - 1)
    t_ -= t_[j_]
    for k in range(len(x_)):
        ax[k].plot(365. * t_[j_ : ], x_[k, j_ : ] / n_[j_] * p.populationSize,
                   linestyle = ':', color = 'black')

    ax[0].set_ylabel('maternal immunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('infected')
    ax[3].set_ylabel('recovered')
    ax[4].set_ylabel('total')

    ax[4].set_xlabel('time (days)')

    for ax_ in ax:
        yl = ax_.get_ylim()
        if yl[0] < 0.:
            ax_.set_ylim(ymin = 0.)

    if show:
        pyplot.show()


def getIterminal(data):
    (T, X) = zip(*(zip(*y) for (runNumber, y) in data))
    Iterminal = [x[-1][2] for x in X]
    return numpy.asarray(Iterminal)


if __name__ == '__main__':
    import time

    numpy.random.seed(1)

    p = Parameters.Parameters()
    p.populationSize = 1000
    p.birthSeasonalVariance = 1.

    tMax = 1.
    # nRuns = multiprocessing.cpu_count()
    nRuns = 100
    debug = False
    
    t0 = time.time()
    data = multirun(nRuns, p, tMax, debug = debug)
    t1 = time.time()
    print 'Run time: {} seconds.'.format(t1 - t0)
    
    makePlots(data)
