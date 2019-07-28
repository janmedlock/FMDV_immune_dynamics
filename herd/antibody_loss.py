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

    def __repr__(self):
        return super().__repr__(('alpha', 'beta', 'time_max', 'time_min'))

    def hazard(self, time):
        h = ((self.alpha - self.beta)
             * (time - self.time_min) / (self.time_max - self.time_min)
             + self.alpha)
        return numpy.clip(h, 0, None)

    def logsf(self, time, time0):
        '''Logarithm of the survival function.'''
        assert time0 <= time
        # Avoid division by 0.
        if self.alpha != self.beta:
            time_intercept = (self.time_min
                              - self.alpha / (self.alpha - self.beta)
                              * (self.time_max - self.time_min))
            if self.alpha > self.beta:
                # _/
                time0 = max(time0, time_intercept)
            else:
                # \_
                time = min(time, time_intercept)
        time_mid = (time + time0) / 2
        return - self.hazard(time_mid) * (time - time0)

    def sf(self, time, time0):
        '''Survival function.'''
        return numpy.exp(self.logsf(time, time0))

    def isf(self, q, time0):
        '''Inverse survival function.'''
        # FIXME
        raise NotImplementedError

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

    def rvs(self, size=None):
        U = numpy.random.random_sample(size=size)
        return self.ppf(U)
