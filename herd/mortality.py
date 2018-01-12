import numpy
from scipy import integrate, stats

from . import rv


class gen(rv.RV, stats.rv_continuous):
    def __init__(self, parameters, *args, **kwargs):
        stats.rv_continuous.__init__(self, name = 'mortality',
                                     a = 0., *args, **kwargs)

    def annualSurvival(self, age):
        return numpy.where(
            age < 1, 0.66, numpy.where(
                age < 3, 0.79, numpy.where(
                    age < 12, 0.88,
                    0.66)))

    def hazard(self, age):
        return - numpy.log(self.annualSurvival(age))

    def _cdf(self, age):
        cdf_under_1 = 1 - 0.66 ** age
        cdf_1_to_3 = 1 - 0.66 * 0.79 ** (age - 1)
        cdf_3_to_12 = 1 - 0.66 * 0.79 * 0.79 * 0.88 ** (age - 3)
        cdf_12_and_up = 1 - 0.66 * 0.79 ** 2 * 0.88 ** 9 * 0.66 ** (age - 12)
        return numpy.where(age < 1, cdf_under_1,
                           numpy.where(age < 3, cdf_1_to_3,
                                       numpy.where(age < 12, cdf_3_to_12,
                                                   cdf_12_and_up)))

    # Inverse of CDF.
    def _ppf(self, q):
        q_1 = 1 - 0.66
        ppf_under_1 = numpy.log(1 - q) / numpy.log(0.66)
        q_3 = 1 - 0.66 * 0.79 ** 2
        ppf_1_to_3 = 1 + (numpy.log(1 - q) - numpy.log(0.66)) / numpy.log(0.79)
        q_12 = 1 - 0.66 * 0.79 * 0.79 * 0.88 ** 9
        ppf_3_to_12 = (3 + (numpy.log(1 - q) - numpy.log(0.66)
                            - 2 * numpy.log(0.79)) / numpy.log(0.88))
        ppf_12_and_up = (12 + (numpy.log(1 - q) - numpy.log(0.66)
                               - 2 * numpy.log(0.79) - 9 * numpy.log(0.66))
                         / numpy.log(0.66))

        return numpy.where(q < q_1, ppf_under_1,
                           numpy.where(q < q_3, ppf_1_to_3,
                                       numpy.where(q < q_12, ppf_3_to_12,
                                                   ppf_12_and_up)))
