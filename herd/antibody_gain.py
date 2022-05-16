import numpy

from herd.rv import RV


class gen(RV):
    '''Waiting time to gain of antibodies.
    Linear hazard with
    hazard(time_min) = alpha,
    hazard(time_max) = beta,
    then clipped to ensure it is nonnegative.'''
    def __init__(self, parameters, *args, **kwargs):
        self.alpha = parameters.antibody_gain_hazard_alpha
        self.beta = parameters.antibody_gain_hazard_beta
        self.time_max = parameters.antibody_gain_hazard_time_max
        self.time_min = parameters.antibody_gain_hazard_time_min
        self._slope = ((self.beta - self.alpha)
                       / (self.time_max - self.time_min))

    def __repr__(self):
        return super().__repr__(('alpha', 'beta', 'time_max', 'time_min'))

    def hazard(self, time):
        h = self._slope * (time - self.time_min) + self.alpha
        return numpy.clip(h, 0, None)

    def _support(self):
        '''Find the interval where the hazard is positive.'''
        if self.alpha == self.beta:
            # The support is (-inf, inf) unless
            # alpha == beta == 0, i.e. the hazard is 0 everywhere,
            # but the methods that use _support() all continue to give
            # correct results in that case with any return value here.
            return (numpy.NINF, numpy.PINF)
        else:
            # Where the linear part of the hazard would cross 0.
            time_intercept = self.time_min - self.alpha / self._slope
            if self.alpha > self.beta:
                return (numpy.NINF, time_intercept)
            else:
                return (time_intercept, numpy.PINF)

    def logsf(self, time, time0):
        '''Logarithm of the survival function.'''
        support = self._support()
        u_0 = numpy.clip(time0, *support)
        u = numpy.clip(time0 + time, *support)
        u_mid = (u + u_0) / 2
        return numpy.where(u >= u_0,
                           - self.hazard(u_mid) * (u - u_0),
                           0)

    def sf(self, time, time0):
        '''Survival function.'''
        return numpy.exp(self.logsf(time, time0))

    def isf(self, q, time0):
        '''Inverse survival function.'''
        support = self._support()
        if support[0] <= time0 <= support[1]:
            # log q = - \int_{time0}^{time0 + time}
            #                slope * (u - time_min) + alpha du.
            # Let t0 = time0 - time_min,
            #     t = time0 + time - time_min.
            # Then
            # log q = - \int_{t0}^t slope * v + alpha dv,
            # or
            # 0 = slope / 2 * t ** 2
            #     + alpha * t
            #     + log q - slope / 2 * t0 ** 2 - alpha * t0
            # Coefficients of the quadratic polynomial we need to solve.
            t0 = time0 - self.time_min
            a = self._slope / 2
            b = self.alpha
            c = (numpy.ma.log(q)
                 - self._slope / 2 * t0 ** 2
                 - self.alpha * t0)
            # sqrt_D is imaginary when the q is very small.  (When
            # alpha > beta, this is because there is some chance that
            # an animal survives until the hazard hits 0. In general,
            # it can also happen because log(q) is large and
            # negative.) We will say that the survival time is
            # infinity in this case.
            sqrt_D = numpy.ma.sqrt(b ** 2 - 4 * a * c)
            t_vals = numpy.ma.stack(((- b + sqrt_D) / 2 / a,
                                     (- b - sqrt_D) / 2 / a))
            t_vals = numpy.ma.filled(t_vals, numpy.PINF)
            if self.alpha >= self.beta:  # self._slope <= 0
                t = t_vals.min(axis=0)
            else:
                t = t_vals.max(axis=0)
            # t -> time_min + t
            # undoes the change of variables in the integral
            # u -> u - time_min.
            # time -> time - time0
            # gives a result that is time since time0,
            # as the other RVs do.
            time = self.time_min + t - time0
        elif time0 > support[1]:
            time = numpy.PINF * numpy.ones_like(q)
        else:  # time0 < support[0]
            time = self.isf(q, support[0]) + (support[0] - time0)
        assert numpy.all((time > 0) | numpy.isclose(time, 0))
        return time

    def cdf(self, time, time0):
        '''Cumulative distribution function.'''
        return 1 - self.sf(time, time0)

    def ppf(self, q, time0):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q, time0)

    def logpdf(self, time, time0):
        '''Logarithm of the probability density function.'''
        return (numpy.log(self.hazard(time0 + time))
                + self.logsf(time, time0))

    def pdf(self, time, time0):
        '''Probability density function.'''
        return numpy.exp(self.logpdf(time, time0))

    def rvs(self, time0, size=None):
        U = numpy.random.random_sample(size=size)
        return self.ppf(U, time0)
