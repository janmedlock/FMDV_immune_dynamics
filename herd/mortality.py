import numpy
from scipy import integrate, stats

from . import rv


class gen(rv.RV, stats.rv_continuous):
    def __init__(self, parameters, *args, **kwargs):
        stats.rv_continuous.__init__(self, name = 'mortality',
                                     a = 0., *args, **kwargs)

    def annualSurvival(self, age):
        return numpy.where(
            age < 1, 0.7, numpy.where(
                age < 12, 0.95,
                0.5))

    def hazard(self, age):
        return - numpy.log(self.annualSurvival(age))

    # def _cdf(self, age):
    #     result = scipy.integrate.quad(self.hazard, 0, age,
    #                                   limit = 100, full_output = 1)
    #     I = result[0]
    #     return 1. - numpy.exp(- I)

    def _cdf(self, age):
        return numpy.where(
            age < 1, 1 - 0.7**age, numpy.where(
                age < 12, 1 - 0.7 * 0.95 ** (age - 1),
                1 - 0.7 * 0.95 ** 11. * 0.5 ** (age - 12)))
        
    def _ppf(self, q):
        return numpy.where(
            q < 1 - 0.7,
            numpy.log(1 - q) / numpy.log(0.7),
            numpy.where(
                q < 1 - 0.7 * 0.95 ** 11,
                1 + (numpy.log(1 - q) - numpy.log(0.7)) / numpy.log(0.95),
                12 + (numpy.log(1 - q) - numpy.log(0.7)
                      - 11 * numpy.log(0.95)) / numpy.log(0.5)))
