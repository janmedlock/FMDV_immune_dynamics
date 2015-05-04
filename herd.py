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
                infectionTime \
                  = stats.expon.rvs(scale = 1. / forceOfInfection)
            
                self.events['infection'] \
                  = Event(self.herd.time + infectionTime,
                          self.infection,
                          'infection for #{}'.format(self.identifier))
            else:
                try:
                    del self.events['infection']
                except KeyError:
                    pass


class Herd(list):
    def __init__(self, parameters, debug = False):
        self.parameters = parameters

        self.debug = debug

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
        return [self.time] + counts

    def getNextEvent(self):
        if len(self) > 0:
            return min([b.getNextEvent() for b in self])
        else:
            return None

    def stop(self):
        return False
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

        return result

    def findExtinctionTime(self, tMax):
        result = self.run(tMax)
        return result[-1][0]



def doOne(parameters, tMax, *args, **kwds):
    return  Herd(parameters, *args, **kwds).run(tMax)

def showResult(x):
    print 'Simulation ended at {} days.'.format(365. * x[-1][0])

def multirun(nRuns, *args, **kwds):
    pool = multiprocessing.Pool(initializer = numpy.random.seed)
    
    results = [pool.apply_async(doOne,
                                args,
                                kwds,
                                showResult)
               for i in xrange(nRuns)]

    pool.close()

    return [r.get() for r in results]


def getMean(data):
    (T, M, S, I, R) = zip(*(zip(*x) for x in data))

    t_mean = numpy.unique(numpy.hstack(T))

    m_mean = numpy.zeros_like(t_mean)
    s_mean = numpy.zeros_like(t_mean)
    i_mean = numpy.zeros_like(t_mean)
    r_mean = numpy.zeros_like(t_mean)
    for (j, tj) in enumerate(t_mean):
        n = 0.
        for (Tk, Mk, Sk, Ik, Rk) in zip(T, M, S, I, R):
            # If we're not past the end of this simulation.
            if tj <= Tk[-1]:
                # Find the last time point <= tj.
                jk = numpy.argwhere(numpy.asarray(Tk) <= tj)[-1]
                n += 1.
                m_mean[j] += Mk[jk]
                s_mean[j] += Sk[jk]
                i_mean[j] += Ik[jk]
                r_mean[j] += Rk[jk]
        m_mean[j] /= n
        s_mean[j] /= n
        i_mean[j] /= n
        r_mean[j] /= n

    return (t_mean, m_mean, s_mean, i_mean, r_mean)


if __name__ == '__main__':
    from matplotlib import pyplot
    import seaborn
    import itertools
    from scipy import integrate

    from Parameters import pde

    p = Parameters.Parameters()

    p.populationSize = 1000
    p.infectionDuration = 21. / 365.
    p.R0 = 10.
    p.birthSeasonalVariation = 1.

    tMax = 1.
    nRuns = multiprocessing.cpu_count()
    debug = False
    
    colors = itertools.cycle(seaborn.color_palette('husl', 8))
    data = multirun(nRuns, p, tMax, debug = debug)

    (fig, ax) = pyplot.subplots(4, sharex = True)
    (T, M, S, I, R) = zip(*(zip(*x) for x in data))
    for (t, m, s, i, r) in zip(T, M, S, I, R):
        ax[0].step(365. * numpy.asarray(t), m, where = 'post',
                   color = colors.next(), alpha = 0.5)
        ax[1].step(365. * numpy.asarray(t), s, where = 'post',
                   color = colors.next(), alpha = 0.5)
        ax[2].step(365. * numpy.asarray(t), i, where = 'post',
                   color = colors.next(), alpha = 0.5)
        ax[3].step(365. * numpy.asarray(t), r, where = 'post',
                   color = colors.next(), alpha = 0.5)

    (t_mean, M_mean, S_mean, I_mean, R_mean) = getMean(data)
    ax[0].step(365. * t_mean, M_mean, where = 'post',
               color = 'black')
    ax[1].step(365. * t_mean, S_mean, where = 'post',
               color = 'black')
    ax[2].step(365. * t_mean, I_mean, where = 'post',
               color = 'black')
    ax[3].step(365. * t_mean, R_mean, where = 'post',
               color = 'black')

    (t_, a_, (M_, S_, I_, R_)) = pde.solve(20, 20, 0.1, p)
    m_ = integrate.trapz(M_, a_, axis = 1)
    s_ = integrate.trapz(S_, a_, axis = 1)
    i_ = integrate.trapz(I_, a_, axis = 1)
    r_ = integrate.trapz(R_, a_, axis = 1)
    n_ = m_ + s_ + i_ + r_
    Tmax_ = numpy.ceil(numpy.max(numpy.hstack(T)))
    dt_ = t_[1] - t_[0]
    j_ = int(- Tmax_ / dt_ - 1)
    t_ -= t_[j_]
    ax[0].plot(365. * t_[j_ : ], m_[j_ : ] / n_[j_] * p.populationSize,
               linestyle = ':', color = 'black')
    ax[1].plot(365. * t_[j_ : ], s_[j_ : ] / n_[j_] * p.populationSize,
               linestyle = ':', color = 'black')
    ax[2].plot(365. * t_[j_ : ], i_[j_ : ] / n_[j_] * p.populationSize,
               linestyle = ':', color = 'black')
    ax[3].plot(365. * t_[j_ : ], r_[j_ : ] / n_[j_] * p.populationSize,
               linestyle = ':', color = 'black')

    for ax_ in ax:
        ax_.set_ylim(ymin = 0.)
    ax[3].set_xlabel('time (days)')
    ax[0].set_ylabel('maternal immunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('infected')
    ax[3].set_ylabel('recovered')

    pyplot.show()
