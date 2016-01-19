import numpy

from . import birth_super


def fracpart(x):
    return numpy.mod(x, 1)


class gen(birth_super.gen):
    def _getparams(self):
        alpha = 1 + self.seasonal_coefficient_of_variation ** 2
        beta = 1 / (1 + self.seasonal_coefficient_of_variation ** 2)
        return (alpha, beta)

    def hazard(self, time, time0, age0):
        (alpha, beta) = self._getparams()
        
        tau = fracpart(time + time0 + beta / 2)

        # 0 if current age (age0 + time) < 4
        # else: alpha if (time + time0 - beta / 2) mod 1 <= beta
        #       else: 0
        return self.scaling * numpy.where(age0 + time < 4, 0,
                                          numpy.where(tau <= beta, alpha, 0))

    def _cdf(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        c = numpy.clip(4 - age0, 0, numpy.inf) + time0 + beta / 2
        d = time + time0 + beta / 2
        
        I = self.scaling * numpy.where(
            time < 4 - age0, 0,
            alpha * (beta * (numpy.floor(d) - numpy.floor(c))
                     + numpy.clip(fracpart(d), -numpy.inf, beta)
                     - numpy.clip(fracpart(c), -numpy.inf, beta)))

        return 1 - numpy.exp(- I)
