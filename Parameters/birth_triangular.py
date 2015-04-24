import numpy

from . import birth


class birth_gen(birth.birth_gen):
    def _getparams(self):
        if self.seasonalVariance < 1. / 3.:
            alpha = 1. + numpy.sqrt(3. * self.seasonalVariance)
            beta = (2. * numpy.sqrt(3. * self.seasonalVariance)
                    / (1. + numpy.sqrt(3. * self.seasonalVariance)))

        else:
            alpha = 3. * (1. + self.seasonalVariance) / 2.
            beta = 3. * (1. + self.seasonalVariance) / 4.
            
        return (alpha, beta)

    def hazard(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        tau = numpy.mod(time + time0, 1.)

        fdown = alpha * numpy.clip(1. - 2. * beta * tau, 0., numpy.inf)
        fup = alpha * numpy.clip(1. - 2. * beta * (1 - tau), 0., numpy.inf)
        # 0 if current age (age0 + time) < 4
        # else: alpha if (time + time0 - beta / 2) mod 1 <= beta
        #       else: 0
        return self.scaling * numpy.where(age0 + time < 4., 0.,
                                          numpy.where(tau <= 0.5, fdown, fup))

    def _cdf(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        c = numpy.clip(4 - age0, 0., numpy.inf) + time0
        d = time + time0
        
        if beta < 1:
            H0 = (1 - beta / 2.) * (numpy.floor(d) - numpy.floor(c) - 1.)
            H1 = numpy.where(numpy.mod(c, 1.) < 1. / 2.,
                             1. / 2. * (1. - beta / 2.) + 
                             (1. / 2. - numpy.mod(c, 1.))
                             * (1. - beta / 2. - beta * numpy.mod(c, 1.)),
                             (1. - numpy.mod(c, 1.))
                             * (1 - beta * numpy.mod(c, 1.)))
            H2 = numpy.where(numpy.mod(d, 1.) < 1. / 2.,
                             numpy.mod(d, 1.) * (1. - beta * numpy.mod(d, 1.)),
                             1. / 2. * (1. - beta / 2.)
                             + (numpy.mod(d, 1.) - 1. / 2.)
                             * (1. - 3. / 2. * beta + beta * numpy.mod(d, 1.)))

        else:
            H0 = 1 / 2. / beta * (numpy.floor(d) - numpy.floor(c) - 1.)
            H1 = numpy.where(numpy.mod(c, 1.) < 1. / 2. / beta,
                             1. / 4. / beta
                             + beta * (1. / 2. / beta - numpy.mod(c, 1.)) ** 2,
                             numpy.where(numpy.mod(c, 1.) < 1. - 1. / 2. / beta,
                                         1. / 4. / beta,
                                         (1. - numpy.mod(c, 1.))
                                         * (1. - beta
                                            * (1. - numpy.mod(c, 1.)))))
            H2 = numpy.where(numpy.mod(d, 1.) < 1. / 2. / beta,
                             numpy.mod(d, 1.) * (1. - beta * numpy.mod(d, 1.)),
                             numpy.where(numpy.mod(d, 1.) < 1. - 1. / 2. / beta,
                                         1. / 4. / beta,
                                         1. / 4. / beta
                                         + beta * (numpy.mod(d, 1.) - 1
                                                   + 1. / 2. / beta) ** 2))
                                     

        I = self.scaling * numpy.where(time < 4 - age0,
                                       0.,
                                       alpha * (H0 + H1 + H2))

        return 1. - numpy.exp(- I)
