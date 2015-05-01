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
    def __init__(self, herd, age = 0., identifier = None):
        self.herd = herd

        # All members of the herd have the same parameters.
        self.RVs = self.herd.RVs

        self.birthDate = self.herd.time - age
        self.identifier = identifier
        self.sex = 'male' if (self.RVs.male.rvs() == 1) \
          else 'female'

        self.events = {}

        maternalImmunityWaningAge \
          = self.RVs.maternalImmunityWaning.rvs()
        if age < maternalImmunityWaningAge:
            self.immuneStatus = 'maternal immunity'

            self.events['maternalImmunityWaning'] \
            = Event(self.birthDate + maternalImmunityWaningAge,
                    self.maternalImmunityWaning,
                    'maternal immunity waning for #{}'.format(self.identifier))
        else:
            if age < 2.:
                self.immuneStatus = 'susceptible'
            else:
                self.immuneStatus = 'recovered'

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
        self.herd.birth()
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

        ages = self.RVs.ageStructure.rvs(size = self.parameters.populationSize)
        for a in ages:
            self.birth(a)

        self.addInfections(self.parameters.initialInfections)

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

    def birth(self, age = 0.):
        if self.debug:
            if age > 0:
                print 't = {}: arrival of #{} at age {}'.format(self.time,
                                                                self.identifier,
                                                                age)
            else:
                print 't = {}: birth of #{}'.format(self.time,
                                                    self.identifier)

        self.append(Buffalo(self, age, identifier = self.identifier))
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


if __name__ == '__main__':
    import pylab

    import odes

    p = Parameters.Parameters()

    p.populationSize = 10000
    p.infectionDuration = 21. / 365.
    p.R0 = 10.
    p.birthSeasonalVariation = 1.

    tMax = 10.
    nRuns = 10
    debug = False
    
    numpy.random.seed(1)

    extinctionTimes = []
    for r in xrange(nRuns):
        h = Herd(p, debug = debug)
        result = h.run(tMax)
        (t, I) = zip(*result)

        eT = t[-1]
        extinctionTimes.append(eT)
        print 'Run #{}: Extinct after {} days'.format(r + 1, 365. * eT)

        pylab.step(365. * numpy.asarray(t), I, where = 'post')

    (to, Mo, So, Io, Ro) = odes.solve(max(extinctionTimes), p)
    pylab.plot(365. * to, Io, linestyle = ':')

    pylab.xlabel('time (days)')
    pylab.ylabel('number infected')
    pylab.ylim(ymin = 0.)

    pylab.show()
