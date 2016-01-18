from scipy import stats

from event import Event


class Buffalo:
    'A single buffalo and the actions that can occur to it.'

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
        return min(self.events.values())

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
