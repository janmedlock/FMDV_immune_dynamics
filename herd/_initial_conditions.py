'''Estimate the infection hazard from the Hedger 1972 data.'''

from functools import partial
import os.path
import re

from joblib import Memory
import numpy
import pandas
from scipy.integrate import quadrature
from scipy.optimize import minimize

from herd import chronic_recovery, maternal_immunity_waning, parameters


_filename = 'data/Hedger_1972_survey_data.xlsx'
# It is relative to directory as this source file.
_filename = os.path.join(os.path.dirname(__file__), _filename)


def _load_data(params):
    '''Load and format the data.'''
    if params.model == 'chronic':
        sheet_name = 'all groups'
    else:
        sheet_name = 'reclassified_no carriers'
    data = pandas.read_excel(_filename,
                             sheet_name=sheet_name,
                             skipfooter=18)
    data.drop(columns=['age', 'age.1', 'age.2'], inplace=True)
    # Each row is for an age interval.
    # Breaks are where one intervals ends and the next starts.
    data.index = pandas.IntervalIndex.from_breaks([0, 1, 2, 3, 4, 7, 11],
                                                  closed='left',
                                                  name='age (y)')
    # Convert proportions to numbers.
    N = data.pop('N')
    data = data.mul(N, axis='index').astype(int)
    # Use the multiindex (SAT, status) for the columns.
    column_re = re.compile(r'^%(.*) - SAT(.*)$')
    tuples = []
    for c in data.columns:
        m = column_re.match(c)
        status, SAT = m.groups()
        tuples.append((int(SAT), status))
    data.columns = pandas.MultiIndex.from_tuples(tuples,
                                                 names=['SAT', 'status'])
    # Add SATs together into pooled data.
    data = data.sum(axis='columns', level='status')
    # The algorithm below only needs to know whether buffalo are
    # maternal immune (M), susceptible (S), or have been infected (R).
    # Add I and C to R and drop I and C.
    ICR = data.reindex(['I', 'C', 'R'], axis='columns', fill_value=0)
    data['R'] = ICR.sum(axis='columns')
    data.drop(columns=['I', 'C'], errors='ignore', inplace=True)
    return data


def _vectorize(**kwds):
    '''Decorator to vectorize a scalar function.'''
    # Just make `numpy.vectorize()` easier to use as a decorator.
    return partial(numpy.vectorize, **kwds)


# Some of the functions below are slow, so the values are cached
# to disk with `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir, verbose=0)


class SParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `S_logprob()` etc
    so that they can be efficiently cached.'''
    def __init__(self, params):
        self.model = params.model
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        self.maternal_immunity_duration_mean = float(
            params.maternal_immunity_duration_mean)
        self.maternal_immunity_duration_shape = float(
            params.maternal_immunity_duration_shape)


def _S_logprob_integrand(b, hazard_infection, maternal_immunity_waningRV):
    return numpy.exp(maternal_immunity_waningRV.logpdf(b)
                     + hazard_infection * b)


# Make the function able to handle vector-valued `age`.
@_vectorize(otypes=[float])
# The function is slow because of the calls to `scipy.integrate.quadrature()`.
@_cache.cache
def _S_logprob_integral(age, hazard_infection, params):
    maternal_immunity_waningRV = maternal_immunity_waning.gen(params)
    val, _ = quadrature(_S_logprob_integrand, 0, age,
                        args=(hazard_infection, maternal_immunity_waningRV),
                        maxiter=10000)
    return val


def S_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of being susceptible at age `a`.
    This is
    \log \int_0^a Prob{Transitioning from M to S at age b}
                  * exp(- hazard_infection * (a - b)) db
    = \log \int_0^a Prob{Transitioning from M to S at age b}
                    * exp(hazard_infection * b) db
      - hazard_infection * a.'''
    assert hazard_infection >= 0, hazard_infection
    params_cache = SParameters(params)
    # Handle log(0).
    return numpy.ma.filled(
        numpy.ma.log(_S_logprob_integral(age, hazard_infection, params_cache))
        - hazard_infection * age,
        - numpy.inf)


def S_prob(age, hazard_infection, params):
    '''The probability of being susceptible at age `a`.'''
    return numpy.exp(S_logprob(age, hazard_infection, params))


class CParameters(SParameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `C_logprob()` etc
    so that they can be efficiently cached.'''
    def __init__(self, params):
        # This needs the parameters in `SParameters()`
        # and a few more.
        super().__init__(params)
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        self.chronic_recovery_mean = float(
            params.chronic_recovery_mean)
        self.chronic_recovery_shape = float(
            params.chronic_recovery_shape)


def _C_logprob_integrand(b, a, hazard_infection, params, chronic_recoveryRV):
    return numpy.exp(S_logprob(b, hazard_infection, params)
                     + chronic_recoveryRV.logsf(a - b))

# Make the function able to handle vector-valued `age`.
@_vectorize(otypes=[float])
# The function is slow because of the calls to `scipy.integrate.quadrature()`.
@_cache.cache
def _C_logprob_integral(age, hazard_infection, params):
    chronic_recoveryRV = chronic_recovery.gen(params)
    val, _ = quadrature(_C_logprob_integrand, 0, age,
                        args=(age, hazard_infection, params,
                              chronic_recoveryRV),
                        maxiter=10000)
    return val


def C_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of being chronically infected
    at age `a`.  This is
    \log \int_0^a Prob{Infection at age b}
                  * probabilty_chronic
                  * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * hazard_infection
                    * probabilty_chronic
                    * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * Prob{survival in chronically infected for (a - b)} db
      + \log hazard_infection
      + \log probabilty_chronic.'''
    assert hazard_infection >= 0, hazard_infection
    if params.model != 'chronic':
        # Shortcut to probability = 0.
        return - numpy.inf * numpy.ones_like(age)
    params_cache = CParameters(params)
    # Handle log(0).
    return numpy.ma.filled(
        numpy.ma.log(_C_logprob_integral(age, hazard_infection, params_cache))
        + numpy.ma.log(hazard_infection)
        + numpy.ma.log(params.probability_chronic),
        - numpy.inf)


def C_prob(age, hazard_infection, params):
    '''The probability of being chronically infected at age `a`.'''
    return numpy.exp(C_logprob(age, hazard_infection, params))


def _minus_loglikelihood(hazard_infection, params, data):
    '''This gets optimized over `hazard_infection`.'''
    # Convert the length-1 `hazard_infection` to a scalar Python float,
    # (not a size-() `numpy.array()`).
    hazard_infection = numpy.squeeze(hazard_infection)[()]
    maternal_immunity_waningRV = maternal_immunity_waning.gen(params)
    l = 0
    # Loop over age groups.
    for age_interval, data_age in data.iterrows():
        # Consider doing something better to get the
        # representative age for an age interval.
        age = age_interval.mid
        M_logprob = maternal_immunity_waningRV.logsf(age)
        # Avoid 0 * -inf.
        l += ((data_age['M'] * M_logprob)
              if (data_age['M'] > 0)
              else 0)
        S_logprob_ = S_logprob(age, hazard_infection, params)
        # Avoid 0 * -inf.
        l += ((data_age['S'] * S_logprob_)
              if (data_age['S'] > 0)
              else 0)
        # R_not_prob = 1 - R_prob.
        R_not_prob = numpy.exp(M_logprob) + numpy.exp(S_logprob_)
        # R_logprob = numpy.log(1 - R_not_prob)
        # but more accurate for R_not_prob near 0,
        # and handle log(0).
        R_logprob = (numpy.log1p(- R_not_prob)
                     if (R_not_prob < 1)
                     else (- numpy.inf))
        # Avoid 0 * -inf.
        l += ((data_age['R'] * R_logprob)
              if (data_age['R'] > 0)
              else 0)
    return -l


def minus_loglikelihood(hazard_infection, params, data):
    params_cache = SParameters(params)
    return _minus_loglikelihood(hazard_infection, params_cache, data)


# The function is very slow because of the call to `scipy.optimize.minimize()`.
@_cache.cache
def _find_hazard_infection(params):
    '''Find the MLE for the infection hazard.'''
    data = _load_data(params)
    x0 = 0.4
    res = minimize(_minus_loglikelihood, x0,
                   args=(params, data),
                   bounds=[(0, numpy.inf)])
    assert res.success, res
    # Convert the length-1 array `res.x` to a scalar Python float,
    # (not a size-() `numpy.array()`).
    return numpy.squeeze(res.x)[()]


def find_hazard_infection(params):
    '''Find the MLE for the infection hazard for each SAT.'''
    params_cache = SParameters(params)
    return _find_hazard_infection(params_cache)


def find_AIC(hazard_infection, params):
    data = _load_data(params)
    # The model has 1 parameter, hazard_infection.
    n_params = 1
    mll = minus_loglikelihood(hazard_infection, params, data)
    return 2 * mll + 2 * n_params


def plot(hazard_infection, params, CI=0.5, show=True, label=None, **kwds):
    from matplotlib import pyplot
    from scipy.stats import beta
    data = _load_data(params)
    ages = numpy.linspace(0, data.index[-1].right, 301)
    lines = pyplot.plot(ages, S_prob(ages, hazard_infection, params))
    color = lines[0].get_color()
    S = data['S']
    N = data.sum(axis=1)
    p_bar = S / N
    p_err = numpy.stack(
        (p_bar - beta.ppf(CI / 2, S + 1, N - S + 1),
         beta.ppf(1 - CI / 2, S + 1, N - S + 1) - p_bar))
    pyplot.errorbar(data.index.mid, p_bar, yerr=p_err,
                    label=label, color=color,
                    marker='_', linestyle='dotted',
                    **kwds)
    pyplot.xlabel(data.index.name)
    pyplot.ylabel('Fraction susceptible')
    if show:
        pyplot.show()
