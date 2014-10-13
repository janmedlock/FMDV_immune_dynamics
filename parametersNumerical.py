import numpy
import scipy.integrate
import scipy.stats
import utility


class mortality_gen(scipy.stats.rv_continuous):
    def annualSurvival(self, age):
        return numpy.where(
            age < 1., 0.7, numpy.where(
                age < 12., 0.95,
                0.5))

    def hazard(self, age):
        return - numpy.log(self.annualSurvival(age))

    def _cdf(self, age):
        result = scipy.integrate.quad(self.hazard, 0., age,
                                      limit = 100, full_output = 1)
        I = result[0]
        return 1. - numpy.exp(- I)

mortality = mortality_gen(name = 'mortality', a = 0., xa = 0., xb = 50.)


growthRate = 0.

class ageStructure_gen(scipy.stats.rv_continuous):
    def _argcheck(self, r):
        return True
        
    def _pdf(self, age, mortality, r):
        return numpy.exp(- r * age) * (1. - mortality.cdf(age))

ageStructure = ageStructure_gen(name = 'ageStructure',
                                a = 0., shapes = 'mortality, r')(
                                    mortality, growthRate)


probabilityOfMaleBirth = 0.5

male = scipy.stats.bernoulli(probabilityOfMaleBirth)


_birthScaling = 1.

class birth_gen(scipy.stats.rv_continuous):
    def _argcheck(self, time0, age0):
        return (age0 >= 0.)
        
    def hazard(self, time, time0, age0):
        return _birthScaling \
          * numpy.where(age0 + time < 4., 0.,
                        (1. + numpy.cos(2 * numpy.pi * (time + time0))) / 2.)

    def _cdf_single(self, time, time0, age0):
        result = scipy.integrate.quad(self.hazard, 0, time,
                                      args = (time0, age0),
                                      limit = 100, full_output = 1)
        I = result[0]
        return 1. - numpy.exp(- I)

    def _cdf(self, time, time0, age0):
        return numpy.vectorize(self._cdf_single)(time, time0, age0)

birth = birth_gen(name = 'birth', a = 0., shapes = 'time0, age0',
                  xa = 0., xb = 50.)


def findBirthScaling(mortality, birth, male, growthRate,
                     ageStep = 0.1, ageMax = 20.):
    ages = numpy.arange(0., ageMax + ageStep, ageStep)

    A = 1. / ageStep * (
        - numpy.diag(numpy.append(numpy.ones(len(ages) - 1), [0.])) \
        + numpy.diag(numpy.ones(len(ages) -1), -1))

    M = numpy.diag(mortality.hazard(ages))

    F = numpy.zeros((len(ages), len(ages)))
    # The first row, F[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in xrange(len(ages)):
        bj = lambda t: (1. - male.mean()) * birth.hazard(t, 0., ages[j])
        result = scipy.integrate.quad(bj, 0., 1., limit = 100)
        F[0, j] = result[0]

    def objective(z):
        G = z * F + A - M
        return utility.findDominantEigenvalue(G) - growthRate

    return numpy.asscalar(scipy.optimize.fsolve(objective, 1.))

_birthScaling = findBirthScaling(mortality, birth, male, growthRate)


class deterministic_gen(scipy.stats.rv_continuous):
    def _cdf(self, age):
        return numpy.where(age < 1., 0., 1.)

    def _ppf(self, age):
        return numpy.ones_like(age)
    
    def _rvs(self):
        return numpy.ones(self._size)

deterministic = deterministic_gen(name = 'deterministic', a = 1., b = 1.)


maternalImmunityDuration = 0.5

maternalImmunityWaning = deterministic(scale = maternalImmunityDuration)


infectionDuration = 1.6 / 365

recovery = deterministic(scale = infectionDuration)


populationSize = 100


R0 = 10.

transmissionRate = R0 / populationSize / recovery.mean()
