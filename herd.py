#!/usr/bin/python

import numpy
from scipy import stats

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

        flags_ages = self.RVs.endemicEquilibrium.rvs(
            self.parameters.populationSize)

        for (flag, ages) in enumerate(flags_ages):
            if flag == 0:
                immuneStatus = 'maternal immunity'
            elif flag == 1:
                immuneStatus = 'susceptible'
            elif flag == 2:
                immuneStatus = 'infectious'
            elif flag == 3:
                immuneStatus = 'recovered'
            else:
                raise ValueError('Unknown immuneStatus flag = {}!'.format(flag))
                
            for age in ages:
                self.birth(immuneStatus, age)

        # self.addInfections(self.parameters.initialInfections)

    def addInfections(self, numberOfInfections):
        i = 0
        for b in self:
            if b.isSusceptible():
                b.infection()
                i += 1
            if i >= numberOfInfections:
                break
        if i != numberOfInfections:
            raise RuntimeError('Could only make {} infections!'.format(i))

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

    def getNextEvent(self):
        if len(self) > 0:
            return min([b.getNextEvent() for b in self])
        else:
            return None

    def stop(self):
        return (self.numberInfectious == 0)

    def step(self, tMax = numpy.inf):
        event = self.getNextEvent()

        if (event is not None) and (event.time < tMax):
            if self.debug:
                print 't = {}: {}'.format(event.time, event.label)
            self.time = event.time
            event()
        else:
            self.time = tMax

        self.updateInfectionTimes()

    def run(self, tMax):
        self.updateInfectionTimes()
        result = [(self.time, self.numberInfectious)]

        while (self.time < tMax) and (not self.stop()):
            self.step(tMax)
            result.append((self.time, self.numberInfectious))

        return result

    def findExtinctionTime(self, tMax):
        result = self.run(tMax)
        return result[-1][0]


def getMean(data):
    (T, I) = zip(*data)

    t_mean = numpy.unique(numpy.hstack(T))

    i_mean = numpy.zeros_like(t_mean)
    for (j, tj) in enumerate(t_mean):
        for (Tk, Ik) in zip(T, I):
            i_mean[j] += numpy.compress(numpy.asarray(Tk) <= tj, Ik)[-1]
    i_mean /= len(data)

    return (t_mean, i_mean)


if __name__ == '__main__':
    import pylab
    import seaborn
    import itertools
    from scipy import integrate

    from Parameters import pde

    p = Parameters.Parameters()

    p.populationSize = 10000
    p.infectionDuration = 21. / 365.
    p.R0 = 10.
    p.birthSeasonalVariation = 1.

    tMax = 2.
    nRuns = 5
    debug = False
    
    numpy.random.seed(1)

    colors = itertools.cycle(seaborn.color_palette('husl', 8))
    data = []
    extinctionTimes = []
    for r in xrange(nRuns):
        h = Herd(p, debug = debug)
        result = h.run(tMax)
        (t, I) = zip(*result)
        data.append((t, I))

        eT = t[-1]
        extinctionTimes.append(eT)
        print 'Run #{}: Extinct after {} days'.format(r + 1, 365. * eT)

        pylab.step(365. * numpy.asarray(t), I, where = 'post',
                   color = colors.next(), alpha = 0.5)

    (t_mean, I_mean) = getMean(data)
    pylab.step(365. * t_mean, I_mean, where = 'post',
               color = 'black')

    (t, a, (M, S, I, R)) = pde.solve(20, 20, 0.1, p)
    i = integrate.trapz(I, a, axis = 1)
    n = numpy.ceil(max(extinctionTimes))
    dt = t[1] - t[0]
    j0 = int(- n / dt - 1)
    t -= t[j0]
    pylab.plot(365. * t[j0 : ], i[j0 : ],
               linestyle = ':', color = 'black')

    pylab.xlabel('time (days)')
    pylab.ylabel('number infected')
    pylab.ylim(ymin = 0.)

    pylab.show()
