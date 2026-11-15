import numpy
import scipy

from .. import _event, _floquet, _period, _rv, immune_status

# The code below is not tested at all for `period != 1`!
period = 1  # Units: year
_period.periods.append(period)


def fracpart(x, out=None):
    return numpy.ma.mod(x, period, out=out)


def get_seasonal_coefficient_of_variation_from_gap_size(g):
    '''g in months'''
    if g is None:
        return 0
    else:
        return numpy.sqrt(4 / 3 / (1 - g / 12) - 1)


def get_gap_size_from_seasonal_coefficient_of_variation(c_v):
    '''in months'''
    if c_v == 0:
        return None
    else:
        return 12 * (1 - 4 / 3 / (1 + c_v ** 2))


class WaitingTime(_rv.MemorylessWaitingTimeMixin,
                  _rv.RV):
    '''Waiting time to the next birth.'''
    def __init__(self, parameters, seed=None, _scaling=None, **kwds):
        super().__init__(parameters, seed=seed)
        self.birth_peak_time = parameters.birth_peak_time
        self.birth_seasonal_coefficient_of_variation = (
            parameters.birth_seasonal_coefficient_of_variation)
        self.menarche = parameters.birth_menarche
        if self.birth_seasonal_coefficient_of_variation < 1 / numpy.sqrt(3):
            self._alpha = (1 + numpy.sqrt(3)
                           * self.birth_seasonal_coefficient_of_variation)
            self._beta = (2 * numpy.sqrt(3)
                          * self.birth_seasonal_coefficient_of_variation
                          / (1 + numpy.sqrt(3)
                             * self.birth_seasonal_coefficient_of_variation))
        else:
            self._alpha = (
                3 / 2
                * (1 + self.birth_seasonal_coefficient_of_variation ** 2))
            self._beta = (
                3 / 4
                * (1 + self.birth_seasonal_coefficient_of_variation ** 2))
        if _scaling is None:
            _scaling = _floquet.find_birth_scaling(parameters, **kwds)
        self._scaling = _scaling
        self.isf = numpy.vectorize(self._isf_single, otypes='d')

    def hazard(self, time, age, out=None):
        # age is the age at t = time.
        time, age = numpy.broadcast_arrays(time, age)
        if out is None:
            out = numpy.empty(time.shape)
        # Without building intermediate arrays, compute
        # `tau = fracpart(time - self.birth_peak_time)`.
        out[()] = time - self.birth_peak_time
        fracpart(out, out=out)
        # Now out = fracpart(time - self.birth_peak_time)
        #         = tau.
        # Without building intermediate arrays, compute
        # `out = (self._alpha
        #         * (1 - self._beta * (1 - numpy.abs(1 - 2 * tau))))`.
        out *= 2
        out -= 1
        numpy.abs(out, out=out)
        # Now out = abs(2 * tau - 1)
        #         = abs(1 - 2 * tau).
        out -= 1
        out *= self._beta
        out += 1
        # Now out = 1 + beta * (abs(2 * tau - 1) - 1)
        #         = 1 - beta * (1 - abs(2 * tau - 1)).
        out *= self._alpha
        # Now out = alpha * (1 - beta * (1 - abs(2 * tau - 1)))
        # In one step:
        # * If the hazard is negative, set it to 0.
        # * If age < self.menarche, set the hazard to 0.
        out[(out < 0) | (age < self.menarche)] = 0
        # Scale the hazard.
        out *= self._scaling
        return out

    def logsf(self, time, time0, age0):
        # age0 is the age at time = 0, i.e. t = time0.
        c = (time0
             + numpy.clip(self.menarche - age0, 0, numpy.inf)
             - self.birth_peak_time)
        d = time0 + time - self.birth_peak_time
        H0 = numpy.floor(d) - numpy.floor(c) - 1
        if self._beta < 1:
            H1 = numpy.where(
                fracpart(c) < 1 / 2,
                1 / 2 + (self._alpha * (1 / 2 - fracpart(c))
                         * (1 - self._beta
                            + self._beta * (1 / 2 - fracpart(c)))),
                self._alpha * (1 - fracpart(c))
                * (1 - self._beta + self._beta * (1 - fracpart(c))))
            H2 = numpy.where(
                fracpart(d) < 1 / 2,
                self._alpha * fracpart(d) * (1 - self._beta * fracpart(d)),
                1 / 2 + (self._alpha * (fracpart(d) - 1 / 2)
                         * (1 - self._beta
                            + self._beta * (fracpart(d) - 1 / 2))))
            H = H0 + H1 + H2
        else:
            H3 = numpy.where(
                fracpart(c) < 1 / 2 / self._beta,
                1 / 2 + self._alpha * self._beta * (1 / 2 / self._beta
                                                    - fracpart(c)) ** 2,
                numpy.where(
                    fracpart(c) < 1 - 1 / 2 / self._beta,
                    1 / 2,
                    self._alpha * (1 - fracpart(c))
                    * (1 - self._beta * (1 - fracpart(c)))))
            H4 = numpy.where(
                fracpart(d) < 1 / 2 / self._beta,
                self._alpha * fracpart(d) * (1 - self._beta * fracpart(d)),
                numpy.where(
                    fracpart(d) < 1 - 1 / 2 / self._beta,
                    1 / 2,
                    1 / 2 + (self._alpha * self._beta
                             * (fracpart(d) - (1 - 1 / 2 / self._beta)) ** 2)))
            H = H0 + H3 + H4
        H = numpy.where(time < 4 - age0, 0, H)
        return -(self._scaling * H)

    def sf(self, time, time0, age0):
        '''Survival function.'''
        return numpy.exp(self.logsf(time, time0, age0))

    def cdf(self, time, time0, age0):
        '''Cumulative distribution function.'''
        return -numpy.expm1(self.logsf(time, time0, age0))

    def logcdf(self, age):
        '''The logarithm of the cumulative distribution function.'''
        return numpy.log1p(-self.sf(age))

    def logpdf(self, time, time0, age0):
        '''Logarithm of the probability density function.'''
        return (numpy.log(self.hazard(time0 + time, age0 + time))
                + self.logsf(time, time0, age0))

    def pdf(self, time, time0, age0):
        '''The probability density function.'''
        return numpy.exp(self.logpdf(time, time0, age0))

    def _isf_to_solve(self, time, time0, age0, logq):
        return self.logsf(time, time0, age0) - logq

    def _isf_single(self, q, time0, age0):
        '''Inverse survival function.'''
        assert 0 <= q <= 1
        if q == 1:
            return 0
        if q == 0:
            return numpy.inf
        # `scipy.optimize.elementwise.find_root()` is much slower here
        args = (time0, age0, numpy.log(q))
        FACTOR = 10
        left = 0
        # sf(left) > q
        right = FACTOR
        while self._isf_to_solve(right, *args) > 0:
            left = right
            right *= FACTOR
        # sf(right) < q
        result = scipy.optimize.root_scalar(self._isf_to_solve,
                                            bracket=(left, right),
                                            args=args)
        assert result.converged, result
        return result.root

    def ppf(self, q, time0, age0):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q, time0, age0)

    def rvs(self, time0, age0, size=None):
        return self.isf(self.rng.random(size=size), time0, age0)


class BuffaloMixin:
    '''Mixin for `..buffalo.Buffalo()` to add the method needed by
    `Birth()`.'''
    def give_birth(self):
        '''Give birth to a new offspring.'''
        assert self.sex == 'female'
        offspring_immune_status = immune_status.birth_immune_status[
            self.immune_status]
        offspring = type(self)(self.herd, offspring_immune_status)
        self.herd.add(offspring)


class Birth(_event.ForSexMixin,
            _event.Event):
    '''A buffalo giving birth.'''
    for_sex = 'female'
    waiting_time = WaitingTime

    @property
    def waiting_time_rvs_args(self):
        '''Arguments to pass to `.waiting_time.rvs()`.'''
        return (self.buffalo.herd.time,
                self.buffalo.age)

    def do(self):
        self.buffalo.give_birth()
        # Add next birth.
        self.time = self.sample_time()
        self.buffalo.events.add(self)


class _ParametersNoSeasonality:
    '''Dummy parameters for the birth random variable without seasonal
    variation.'''
    def __init__(self, birth_menarche, female_probability):
        self.birth_menarche = birth_menarche
        self.birth_seasonal_coefficient_of_variation = 0
        # The value of `.birth_peak_time` doesn't matter when the
        # seasonal variation is 0, but it must be set.
        self.birth_peak_time = 0


def hazard_no_seasonality(birth_menarche, female_probability, age,
                          *args, out=None, **kwargs):
    '''The hazard without seasonal variation.'''
    params = _ParametersNoSeasonality(birth_menarche, female_probability)
    waiting_time = WaitingTime(params, *args, **kwargs)
    # The value of `time` doesn't matter when the seasonal variation
    # is 0.
    time = 0
    return waiting_time.hazard(time, age, out=out)
