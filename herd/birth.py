import numpy
from scipy import optimize

from herd.rv import RV
# To avoid an import loop caused by `_period`,
# the import is inside `gen()` below.
# from herd.floquet import find_birth_scaling


# Annual period.
# WARNING: Most of the code below has not been checked to see if it
#          works correctly with _period != 1.
_period = 1


def fracpart(x, out=None):
    return numpy.ma.mod(x, _period, out=out)


def get_seasonal_coefficient_of_variation_from_gap_size(g):
    'g in months'
    if g is None:
        return 0
    else:
        return numpy.sqrt(4 / 3 / (1 - g / 12) - 1)


def get_gap_size_from_seasonal_coefficient_of_variation(c_v):
    'in months'
    if c_v == 0:
        return None
    else:
        return 12 * (1 - 4 / 3 / (1 + c_v ** 2))


class gen(RV):
    def __init__(self, parameters, _scaling=None, *args, **kwargs):
        self.peak_time_of_year = parameters.birth_peak_time_of_year
        self.seasonal_coefficient_of_variation \
            = parameters.birth_seasonal_coefficient_of_variation
        if self.seasonal_coefficient_of_variation < 1 / numpy.sqrt(3):
            self._alpha = (1 + numpy.sqrt(3)
                           * self.seasonal_coefficient_of_variation)
            self._beta = (2 * numpy.sqrt(3)
                          * self.seasonal_coefficient_of_variation
                          / (1 + numpy.sqrt(3)
                             * self.seasonal_coefficient_of_variation))
        else:
            self._alpha = (3 / 2
                           * (1 + self.seasonal_coefficient_of_variation ** 2))
            self._beta = (3 / 4
                          * (1 + self.seasonal_coefficient_of_variation ** 2))
        if _scaling is None:
            # To avoid an import loop caused by `_period`, the import is here.
            from herd.floquet import find_birth_scaling
            _scaling = find_birth_scaling(parameters, *args, **kwargs)
        self._scaling = _scaling
        self.ppf = numpy.vectorize(self._ppf_single, otypes='d')

    def __repr__(self):
        return super().__repr__(('peak_time_of_year',
                                 'seasonal_coefficient_of_variation'))

    def hazard(self, time, age, out=None):
        # age is the age at t = time.
        time, age = numpy.broadcast_arrays(time, age)
        if out is None:
            out = numpy.empty(time.shape)
        # Without building intermediate arrays, compute
        # `tau = fracpart(time - self.peak_time_of_year)`.
        out[:] = time - self.peak_time_of_year
        fracpart(out, out=out)
        # Now out = fracpart(time - self.peak_time_of_year)
        #         = tau.
        # Without building intermediate arrays, compute
        # `out = self._alpha * (1 - self._beta * (1 - numpy.abs(1 - 2 * tau)))`.
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
        # * If age < 4, set the hazard to 0.
        out[(out < 0) | (age < 4)] = 0
        # Scale the hazard.
        out *= self._scaling
        return out

    def logsf(self, time, time0, age0):
        # age0 is the age at t = time0.
        c = time0 + numpy.clip(4 - age0, 0, numpy.inf) - self.peak_time_of_year
        d = time0 + time - self.peak_time_of_year
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
        return numpy.exp(self.logsf(time, time0, age0))

    def cdf(self, time, time0, age0):
        return 1 - self.sf(time, time0, age0)

    def pdf(self, time, time0, age0):
        return (self.hazard(time0 + time, age0 + time)
                * self.sf(time, time0, age0))

    def logpdf(self, time, time0, age0):
        return (numpy.log(self.hazard(time0 + time, age0 + time))
                + self.logsf(time, time0, age0))

    def _ppf_to_solve(self, time, time0, age0, q):
        return self.cdf(time, time0, age0) - q

    def _ppf_single(self, q, time0, age0):
        assert 0 <= q <= 1
        if q == 0:
            return 0
        elif q == 1:
            return numpy.inf
        else:
            left = 0
            # ppf(left) < q.
            right = factor = 10
            while self._ppf_to_solve(right, time0, age0, q) < 0:
                left = right
                right *= factor
            # ppf(right) > q.
            return optimize.brentq(self._ppf_to_solve,
                                   left, right, args=(time0, age0, q))

    def isf(self, q, time0, age0):
        return self.ppf(1 - q, time0, age0)

    def rvs(self, time0, age0, size=None):
        U = numpy.random.random_sample(size=size)
        return self.ppf(U, time0, age0)


class _BirthParams:
    '''Dummy parameters for the birth random variable.'''
    def __init__(self, birth_peak_time_of_year,
                 birth_seasonal_coefficient_of_variation):
        self.birth_peak_time_of_year = birth_peak_time_of_year
        self.birth_seasonal_coefficient_of_variation \
            = birth_seasonal_coefficient_of_variation


def from_param_values(birth_peak_time_of_year,
                      birth_seasonal_coefficient_of_variation,
                      *args, **kwargs):
    '''Build a `gen()` instance from parameter values
    instead of a `Parameters()` object.'''
    parameters = _BirthParams(birth_peak_time_of_year,
                              birth_seasonal_coefficient_of_variation)
    return gen(parameters, *args, **kwargs)
