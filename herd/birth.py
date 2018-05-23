import numpy
from scipy import optimize, stats

from . import rv
from . import utility


def fracpart(x):
    return numpy.mod(x, 1)


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


# `start_time` doesn't matter since we're integrating a 1-year-periodic
# function over 1 year.
@utility.shelved('birth_seasonal_coefficient_of_variation',
                 'male_probability_at_birth')
def find_birth_scaling(parameters, _matrices=None, *args, **kwargs):
    if _matrices is None:
        _, _matrices = utility.build_ages_and_matrices(parameters,
                                                       *args, **kwargs)
    def objective(val, *matrices):
        birth_scaling, = val
        r, _ = utility.find_dominant_eigenpair(birth_scaling, *matrices)
        return r
    initial_guess = 1
    opt, _, ier, mesg = optimize.fsolve(objective, initial_guess,
                                        args=_matrices,
                                        full_output=True)
    birth_scaling, = opt
    assert ier == 1, mesg
    return birth_scaling


class gen(rv.RV, stats.rv_continuous):
    def __init__(self, parameters, _find_birth_scaling=True, *args, **kwargs):
        self.seasonal_coefficient_of_variation \
            = parameters.birth_seasonal_coefficient_of_variation
        if _find_birth_scaling:
            self.find_birth_scaling(parameters)
        else:
            self.scaling = 1
        super().__init__(self, name='birth', a=0, shapes='time0, age0',
                         *args, **kwargs)

    def _argcheck(self, time0, age0):
        return (age0 >= 0)

    def __repr__(self):
        return super().__repr__(('seasonal_coefficient_of_variation', ))

    def find_birth_scaling(self, parameters, *args, **kwargs):
        self.scaling = find_birth_scaling(parameters, *args, **kwargs)

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

    def hazard(self, time, time0, age0):
        (alpha, beta) = self._getparams()
        tau = fracpart(time + time0)
        fdown = alpha * numpy.clip(1 - 2 * beta * tau, 0, numpy.inf)
        fup = alpha * numpy.clip(1 - 2 * beta * (1 - tau), 0, numpy.inf)
        # 0 if current age (age0 + time) < 4
        # else: alpha if (time + time0 - beta / 2) mod 1 <= beta
        #       else: 0
        return self.scaling * numpy.where(age0 + time < 4, 0,
                                          numpy.where(tau <= 0.5, fdown, fup))

    def _logsf(self, time, time0, age0):
        (alpha, beta) = self._getparams()
        c = numpy.clip(4 - age0, 0, numpy.inf) + time0
        d = time + time0
        if beta < 1:
            H1 = (1 - beta / 2) * (numpy.floor(d) - numpy.floor(c) - 1)
            H2 = numpy.where(fracpart(c) < 1 / 2,
                             1 / 2 * (1 - beta / 2) +
                             (1 / 2 - fracpart(c))
                             * (1 - beta / 2 - beta * fracpart(c)),
                             (1 - fracpart(c))
                             * (1 - beta * fracpart(c)))
            H3 = numpy.where(fracpart(d) < 1 / 2,
                             fracpart(d) * (1 - beta * fracpart(d)),
                             1 / 2 * (1 - beta / 2)
                             + (fracpart(d) - 1 / 2)
                             * (1 - 3 / 2 * beta + beta * fracpart(d)))
            H = H1 + H2 + H3
        else:
            H4 = 1 / 2 / beta * (numpy.floor(d) - numpy.floor(c) - 1)
            H5 = numpy.where(fracpart(c) < 1 / 2 / beta,
                             1 / 4 / beta
                             + beta * (1 / 2 / beta - fracpart(c)) ** 2,
                             numpy.where(fracpart(c) < 1 - 1 / 2 / beta,
                                         1 / 4 / beta,
                                         (1 - fracpart(c))
                                         * (1 - beta
                                            * (1 - fracpart(c)))))
            H6 = numpy.where(fracpart(d) < 1 / 2 / beta,
                             fracpart(d) * (1 - beta * fracpart(d)),
                             numpy.where(fracpart(d) < 1 - 1 / 2 / beta,
                                         1 / 4 / beta,
                                         1 / 4 / beta
                                         + beta * (fracpart(d) - 1
                                                   + 1 / 2 / beta) ** 2))
            H = H4 + H5 + H6
        H = self.scaling * numpy.where(time < 4 - age0,
                                       0,
                                       alpha * H)
        return (- H)

    def _sf(self, time, time0, age0):
        return numpy.exp(self._logsf(time, time0, age0))

    def _cdf(self, time, time0, age0):
        return 1 - self._sf(time, time0, age0)

    def _pdf(self, time, time0, age0):
        return (self.hazard(time, time0, age0 + time)
                * self._sf(time, time0, age0))

    def _ppf(self, q, *args, **kwds):
        'Trap errors for _ppf'
        try:
            result = super()._ppf(q, *args, **kwds)
        except ValueError:
            # Assume the error is near q = 1,
            # so return the right-hand endpoint
            # of the support of the distribution
            # (which is +inf by default).
            result = self.b
        return result
