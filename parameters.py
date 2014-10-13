import numpy
from scipy import stats, integrate, optimize
import utility


class mortality_gen(stats.rv_continuous):
    def annualSurvival(self, age):
        return numpy.where(
            age < 1., 0.7, numpy.where(
                age < 12., 0.95,
                0.5))

    def hazard(self, age):
        return - numpy.log(self.annualSurvival(age))

    def _cdf(self, age):
        return numpy.where(
            age < 1., 1. - 0.7**age, numpy.where(
                age < 12., 1. - 0.7 * 0.95 ** (age - 1.),
                1. - 0.7 * 0.95 ** 11. * 0.5 ** (age - 12.)))
        
    def _ppf(self, q):
        return numpy.where(
            q < 1. - 0.7,
            numpy.log(1. - q) / numpy.log(0.7),
            numpy.where(
                q < 1. - 0.7 * 0.95 ** 11,
                1. + (numpy.log(1. - q) - numpy.log(0.7)) / numpy.log(0.95),
                12. + (numpy.log(1. - q) - numpy.log(0.7)
                       - 11. * numpy.log(0.95)) / numpy.log(0.5)))

mortality = mortality_gen(name = 'mortality', a = 0.)


growthRate = 0.

class ageStructure_gen(stats.rv_continuous):
    def _argcheck(self, r):
        return True
        
    def _integrals(self, r):
        # Unscaled CDF at a = 1
        I1 = (1. - 0.7 * numpy.exp(- r)) / (r - numpy.log(0.7))

        # Unscaled CDF at a = 12
        I2 = I1 \
          + 0.7 * numpy.exp(- r) * (1. - 0.95 ** 11. * numpy.exp(- 11. * r)) \
          / (r - numpy.log(0.95))

        # Unscaled CDF at a = infty
        I3 = I2 \
          + 0.7 * 0.95 ** 11. * numpy.exp(- 12. * r) \
          / (r - numpy.log(0.5))

        return (I1, I2, I3)

    def _pdf(self, age, r):
        (I1, I2, I3) = self._integrals(r)

        return numpy.exp(- r * age) / I3 \
          * numpy.where(
              age < 1., 0.7 ** age,
              numpy.where(
                  age < 12., 0.7 * 0.95 ** (age - 1.),
                  0.7 * 0.95 ** 11 * 0.5 ** (age - 12.)))

    def _cdf(self, age, r):
        (I1, I2, I3) = self._integrals(r)

        return 1. / I3 \
          * numpy.where(
              age < 1.,
              (1. - 0.7 ** age * numpy.exp(- r * age)) / (r - numpy.log(0.7)),
              numpy.where(
                  age < 12.,
                  I1
                  + 0.7 * numpy.exp(- r)
                  * (1. - 0.95 ** (age - 1.) * numpy.exp(- r * (age - 1.)))
                  / (r - numpy.log(0.95)),
                  I2
                  + 0.7 * 0.95 ** 11. * numpy.exp(- 12. * r)
                  * (1. - 0.5 ** (age - 12.) * numpy.exp(-r * (age - 12.)))
                  / (r - numpy.log(0.5))))

    def _ppf(self, q, r):
        (I1, I2, I3) = self._integrals(r)

        y = q * I3
        # Censor y to avoid warnings about log of negative numbers
        y1 = numpy.where(y < I1, y, I1)
        y2 = numpy.where(y < I2, y, I2)

        return numpy.where(
            y < I1,
            numpy.log(1. + (numpy.log(0.7) - r) * y1) / (numpy.log(0.7) - r),
            numpy.where(
                y < I2,
                1. + numpy.log(1.
                               + (numpy.log(0.95) - r) / 0.7 * numpy.exp(r)
                               * (y2 - I1))
                / (numpy.log(0.95) - r),
                12. + numpy.log(1.
                                + (numpy.log(0.5) - r) / 0.7 / 0.95 ** 11.
                                * numpy.exp(12. * r) * (y - I2))
                / (numpy.log(0.5) - r)))

ageStructure = ageStructure_gen(name = 'ageStructure', a = 0., shapes = 'r')(
    growthRate)


probabilityOfMaleBirth = 0.5

male = stats.bernoulli(probabilityOfMaleBirth)


_birthScaling = 1.

class birth_gen(stats.rv_continuous):
    def _argcheck(self, time0, age0):
        return (age0 >= 0.)
        
    def hazard(self, time, time0, age0):
        return _birthScaling \
          * numpy.where(age0 + time < 4., 0.,
                        (1. + numpy.cos(2 * numpy.pi * (time + time0))) / 2.)

    def _cdf(self, time, time0, age0):
        lb = numpy.max(numpy.hstack((0., 4 - age0)))
        I = numpy.where(
            time < 4 - age0,
            0.,
            _birthScaling / 2. \
            * ((time - lb)
               + 1. / 2. / numpy.pi
               * (numpy.sin(2. * numpy.pi * (time - time0))
                  - numpy.sin(2. * numpy.pi * (lb - time0)))))
        
        return 1. - numpy.exp(- I)

    def _ppf(self, q, *args, **kwds):
        'Trap errors for _ppf'
        try:
            result = stats.rv_continuous._ppf(self, q, *args, **kwds)
        except ValueError:
            # Assume the error is near q = 1,
            # so return the right-hand endpoint
            # of the support of the distribution
            # (which is +inf by default).
            result = self.b
        return result

birth = birth_gen(name = 'birth', a = 0., shapes = 'time0, age0')


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
        result = integrate.quad(bj, 0., 1., limit = 100)
        F[0, j] = result[0]

    def objective(z):
        G = z * F + A - M
        return utility.findDominantEigenvalue(G) - growthRate

    return numpy.asscalar(optimize.fsolve(objective, 1.))

_birthScaling = findBirthScaling(mortality, birth, male, growthRate)


class deterministic_gen(stats.rv_continuous):
    def _cdf(self, age):
        return numpy.where(age < 1., 0., 1.)

    def _ppf(self, age):
        return numpy.ones_like(age)
    
    def _rvs(self):
        return numpy.ones(self._size)

deterministic = deterministic_gen(name = 'deterministic', a = 1., b = 1.)


maternalImmunityDuration = 0.5

maternalImmunityWaning = deterministic(scale = maternalImmunityDuration)


infectionDuration = 1.6 / 365.

recovery = deterministic(scale = infectionDuration)


populationSize = 100


R0 = 5.

def setTransmissionRate():
    global transmissionRate
    transmissionRate = R0 * (1. / recovery.mean() + 1. / mortality.mean()) \
      / populationSize

setTransmissionRate()


initialInfections = 2
