import numpy

from herd.rv import RV


class gen(RV):
    '''Waiting time to loss of antibodies.
    Linear hazard with
    hazard(time_min) = alpha,
    hazard(time_max) = beta,
    then clipped to ensure it is nonnegative.'''
    def __init__(self, parameters, *args, **kwargs):
        self.alpha = parameters.antibody_loss_hazard_alpha
        self.beta = parameters.antibody_loss_hazard_beta
        self.time_max = parameters.antibody_loss_hazard_time_max
        self.time_min = parameters.antibody_loss_hazard_time_min
        self._slope = ((self.alpha - self.beta)
                       / (self.time_max - self.time_min))

    def __repr__(self):
        return super().__repr__(('alpha', 'beta', 'time_max', 'time_min'))

    def hazard(self, time):
        h = self._slope * (time - self.time_min) + self.alpha
        return numpy.clip(h, 0, None)

    def _support(self):
        '''Find the interval where the hazard is positive.'''
        if self.alpha == self.beta:
            # If alpha == beta == 0, the hazard is 0 everywhere.
            # The methods that use _support() all continue to give
            # correct results with any return value here.
            return (-numpy.inf, numpy.inf)
        else:
            # Where the linear part of the hazard would cross 0.
            time_intercept = self.time_min - self.alpha / self._slope
            if self.alpha > self.beta:
                return (time_intercept, numpy.inf)
            else:
                return (-numpy.inf, time_intercept)

    def logsf(self, time, time0):
        '''Logarithm of the survival function.'''
        support = self._support()
        time = numpy.clip(time, *support)
        time0 = numpy.clip(time0, *support)
        time_mid = (time + time0) / 2
        return numpy.where(time >= time0,
                           - self.hazard(time_mid) * (time - time0),
                           0)

    def sf(self, time, time0):
        '''Survival function.'''
        return numpy.exp(self.logsf(time, time0))

    def isf(self, q, time0):
        '''Inverse survival function.'''
        # log q = - \int_{time_0}^{time} slope * (u - time_min + alpha) du
        # Let t   = time   - time_min,
        #     t_0 = time_0 - time_min.
        # Then
        # 0 = slope / 2 * t ** 2
        #     + alpha * t
        #     + log q - slope / 2 * t_0 ** 2 - alpha * t_0
        # Coefficients of the quadratic polynomial we need to solve.
        t0 = time0 - self.time_min
        a = self._slope / 2
        b = self.alpha
        c = (numpy.ma.log(q)
             - self._slope / 2 * t0 ** 2
             - self.alpha * t0)
        t = numpy.ma.filled((- b + numpy.sqrt(b ** 2 - 4 * a * c)) / 2 / a,
                            numpy.inf)
        return t + self.time_min

    def cdf(self, time, time0):
        '''Cumulative distribution function.'''
        return 1 - self.sf(time, time0)

    def ppf(self, q, time0):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q, time0)

    def logpdf(self, time, time0):
        '''Logarithm of the probability density function.'''
        return numpy.log(self.hazard(time)) + self.logsf(time, time0)

    def pdf(self, time, time0):
        '''Probability density function.'''
        return numpy.exp(self.logpdf(time, time0))

    def rvs(self, time0, size=None):
        U = numpy.random.random_sample(size=size)
        return self.ppf(U, time0)
