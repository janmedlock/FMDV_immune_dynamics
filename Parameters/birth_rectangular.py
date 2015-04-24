import numpy

from . import birth


class birth_gen(birth.birth_gen):
    def _getparams(self):
        alpha = 1 + self.seasonalVariance
        beta = 1 / (1 + self.seasonalVariance)
        return (alpha, beta)

    def hazard(self, time, time0, age0):
        (alpha, beta) = self._getparams()
        
        tau = numpy.mod(time + time0 + beta / 2., 1.)

        # 0 if current age (age0 + time) < 4
        # else: alpha if (time + time0 - beta / 2) mod 1 <= beta
        #       else: 0
        return self.scaling * numpy.where(age0 + time < 4., 0.,
                                          numpy.where(tau <= beta, alpha, 0.))

    def _cdf(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        c = numpy.clip(4 - age0, 0., numpy.inf) + time0 + beta / 2.
        d = time + time0 + beta / 2.
        
        I = self.scaling * numpy.where(
            time < 4 - age0, 0.,
            alpha * (beta * (numpy.floor(d) - numpy.floor(c))
                     + numpy.clip(numpy.mod(d, 1.), -numpy.inf, beta)
                     - numpy.clip(numpy.mod(c, 1.), -numpy.inf, beta)))

        return 1. - numpy.exp(- I)
