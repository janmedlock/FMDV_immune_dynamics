import numpy
from scipy import optimize

from . import rv
from .floquet import find_birth_scaling


def fracpart(x):
    return numpy.ma.mod(x, 1)


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


class gen(rv.RV):
    def __init__(self, parameters, _scaling=None, *args, **kwargs):
        self.peak_time_of_year = parameters.birth_peak_time_of_year
        self.seasonal_coefficient_of_variation \
            = parameters.birth_seasonal_coefficient_of_variation
        if _scaling is None:
            _scaling = find_birth_scaling(parameters, *args, **kwargs)
        self._scaling = _scaling
        self.ppf = numpy.vectorize(self._ppf_single, otypes='d')

    def __repr__(self):
        return super().__repr__(('peak_time_of_year',
                                 'seasonal_coefficient_of_variation'))

    def _getparams(self):
        if self.seasonal_coefficient_of_variation < 1 / numpy.sqrt(3):
            alpha = 1 + numpy.sqrt(3) * self.seasonal_coefficient_of_variation
            beta = (2 * numpy.sqrt(3) * self.seasonal_coefficient_of_variation
                    / (1 + numpy.sqrt(3)
                       * self.seasonal_coefficient_of_variation))
        else:
            alpha = 3 * (1 + self.seasonal_coefficient_of_variation ** 2) / 2
            beta = 3 * (1 + self.seasonal_coefficient_of_variation ** 2) / 4
        return (alpha, beta)

    def hazard(self, time, age):
        # age is the age at t = time.
        (alpha, beta) = self._getparams()
        tau = fracpart(time - self.peak_time_of_year)
        haz = alpha * (1 - beta * (1 - numpy.abs(1 - 2 * tau)))
        haz = numpy.clip(haz, 0, numpy.inf)
        # 0 if current age (age0 + time) < 4.
        haz = numpy.where(age >= 4, haz, 0)
        return self._scaling * haz

    def logsf(self, time, time0, age0):
        # age0 is the age at t = time0.
        (alpha, beta) = self._getparams()
        c = time0 + numpy.clip(4 - age0, 0, numpy.inf) - self.peak_time_of_year
        d = time0 + time - self.peak_time_of_year
        H0 = numpy.floor(d) - numpy.floor(c) - 1
        if beta < 1:
            H1 = numpy.where(
                fracpart(c) < 1 / 2,
                1 / 2 + (alpha * (1 / 2 - fracpart(c))
                         * (1 - beta + beta * (1 / 2 - fracpart(c)))),
                alpha * (1 - fracpart(c))
                * (1 - beta + beta * (1 - fracpart(c))))
            H2 = numpy.where(
                fracpart(d) < 1 / 2,
                alpha * fracpart(d) * (1 - beta * fracpart(d)),
                1 / 2 + (alpha * (fracpart(d) - 1 / 2)
                         * (1 - beta + beta * (fracpart(d) - 1 / 2))))
            H = H0 + H1 + H2
        else:
            H3 = numpy.where(
                fracpart(c) < 1 / 2 / beta,
                1 / 2 + alpha * beta * (1 / 2 / beta - fracpart(c)) ** 2,
                numpy.where(
                    fracpart(c) < 1 - 1 / 2 / beta,
                    1 / 2,
                    alpha * (1 - fracpart(c)) * (1 - beta * (1 - fracpart(c)))))
            H4 = numpy.where(
                fracpart(d) < 1 / 2 / beta,
                alpha * fracpart(d) * (1 - beta * fracpart(d)),
                numpy.where(
                    fracpart(d) < 1 - 1 / 2 / beta,
                    1 / 2,
                    1 / 2 + (alpha * beta
                             * (fracpart(d) - (1 - 1 / 2 / beta)) ** 2)))
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
