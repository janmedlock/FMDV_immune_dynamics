#!/usr/bin/python3
'''Estimate the infection hazard from the Hedger 1972 data.'''

import os.path
import re

from joblib import Memory
import numpy
import pandas
from scipy.integrate import quadrature
from scipy.optimize import minimize

from herd import maternal_immunity_waning, parameters


_filename = 'Hedger_1972_survey_data.xlsx'
# It is in the same directory as this source file.
_filename = os.path.join(os.path.dirname(__file__), _filename)


# Some of the functions below are slow, so the values are cached
# to disk with `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir, verbose=0)


class Parameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `S_logprob()` etc
    so that they can be efficiently cached.'''
    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        self.maternal_immunity_duration_mean = float(
            params.maternal_immunity_duration_mean)
        self.maternal_immunity_duration_shape = float(
            params.maternal_immunity_duration_shape)


def _load_data():
    '''Load and format the data.'''
    data = pandas.read_excel(_filename,
                             sheet_name='reclassified_no carriers',
                             usecols=[1, 2, 3, 5, 6, 7, 9, 10, 11, 12],
                             skip_footer=18)
    # Each rows is for an age interval.
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
    # Pool the data across SATs.
    pooled = data.sum(axis=1, level='status')
    pooled.columns = pandas.MultiIndex.from_product((['Pooled'],
                                                     pooled.columns),
                                                    names=data.columns.names)
    # Estimate for each SAT and the pooled data.
    # return pandas.concat((data, pooled), axis='columns')
    # Estimate only for the pooled data.
    return pooled


def _S_logprob_integrand(b, hazard_infection, maternal_immunity_waningRV):
    return numpy.exp(maternal_immunity_waningRV.logpdf(b)
                     + hazard_infection * b)


# This is slow because of the calls to `scipy.integrate.quadrature()`.
@_cache.cache
def _S_logprob(a, hazard_infection, params):
    assert hazard_infection >= 0, hazard_infection
    maternal_immunity_waningRV = maternal_immunity_waning.gen(params)
    val, _ = quadrature(_S_logprob_integrand, 0, a,
                        args=(hazard_infection, maternal_immunity_waningRV),
                        maxiter=1000)
    # Handle log(0).
    if val == 0:
        return - numpy.inf
    else:
        return numpy.log(val) - hazard_infection * a


def S_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of being susceptible at age `a`.
    This is
    \log \int_0^a Prob{Transitioning from M to S at age b}
                  * exp(- hazard_infection * (a - b)) db
    = \log \int_0^a Prob{Transitioning from M to S at age b}
                    * exp(hazard_infection * b) db
      - hazard_infection * a.'''
    params_cache = Parameters(params)
    if numpy.isscalar(age):
        return _S_logprob(age, hazard_infection, params_cache)
    else:
        return numpy.array([_S_logprob(a, hazard_infection, params_cache)
                            for a in age])


def S_prob(age, hazard_infection, params):
    '''The probability of being susceptible at age `a`.'''
    return numpy.exp(S_logprob(age, hazard_infection, params))


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
        # Hopefully, this is just fixing roundoff errors...
        M_logprob = numpy.clip(M_logprob, -numpy.inf, 0)
        # Avoid 0 * -inf.
        l += ((data_age['M'] * M_logprob)
              if (data_age['M'] > 0)
              else 0)
        S_logprob_ = S_logprob(age, hazard_infection, params)
        # Hopefully, this is just fixing roundoff errors...
        S_logprob_ = numpy.clip(S_logprob_, -numpy.inf, 0)
        # Avoid 0 * -inf.
        l += ((data_age['S'] * S_logprob_)
              if (data_age['S'] > 0)
              else 0)
        # R_not_prob = 1 - R_prob.
        R_not_prob = numpy.exp(M_logprob) + numpy.exp(S_logprob_)
        # Hopefully, this is just fixing roundoff errors...
        R_not_prob = numpy.clip(R_not_prob, 0, 1)
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


# This is very slow because of the call to `scipy.optimize.minimize()`.
@_cache.cache
def _find_hazard_infection(params, data):
    '''Find the MLE for the infection hazard.'''
    x0 = 0.4
    res = minimize(_minus_loglikelihood, x0,
                   args=(params, data),
                   bounds=[(0, numpy.inf)])
    assert res.success, res
    # Convert the length-1 array `res.x` to a scalar Python float,
    # (not a size-() `numpy.array()`).
    return numpy.squeeze(res.x)[()]


@_cache.cache
def find_hazard_infection():
    '''Find the MLE for the infection hazard for each SAT.'''
    data = _load_data()
    # The only parameters used are for maternal-immunity waning,
    # which do not depend on SAT.
    params = parameters.Parameters()
    params_cache = Parameters(params)
    hazard_infection = pandas.Series()
    for i, data_i in data.groupby(axis=1, level='SAT'):
        # Remove 'SAT' from columns.
        data_i.columns = data_i.columns.get_level_values('status')
        hazard_infection.loc[i] = _find_hazard_infection(params_cache, data_i)
    hazard_infection.index.name = 'SAT'
    return hazard_infection


def _find_AIC(hazard_infection, params, data):
    # The model has 1 parameter, hazard_infection.
    n_params = 1
    mll = _minus_loglikelihood(hazard_infection, params, data)
    return 2 * mll + 2 * n_params


def find_AIC(hazard_infection):
    data = _load_data()
    # The only parameters used are for maternal-immunity waning,
    # which do not depend on SAT.
    params = parameters.Parameters()
    params_cache = Parameters(params)
    AIC = pandas.Series()
    for i, data_i in data.groupby(axis=1, level='SAT'):
        # Remove 'SAT' from columns.
        data_i.columns = data_i.columns.get_level_values('status')
        AIC.loc[i] = _find_AIC(hazard_infection[i], params_cache, data_i)
    AIC.index.name = 'SAT'
    return AIC


def plot(hazard_infection, CI=0.5, show=True):
    from matplotlib import pyplot
    from scipy.stats import beta
    data = _load_data()
    # The only parameters used are for maternal-immunity waning,
    # which do not depend on SAT.
    params = parameters.Parameters()
    params_cache = Parameters(params)
    ages = numpy.linspace(0, data.index[-1].right, 301)
    for i, data_i in data.groupby(axis=1, level='SAT'):
        S_prob_ = S_prob(ages, hazard_infection[i], params_cache)
        lines = pyplot.plot(ages, S_prob_, label=i)
        color = lines[0].get_color()
        N = data_i.sum(axis=1)
        ix, _ = data_i.columns.get_loc_level('S', level='status')
        S = data_i.loc[:, ix].sum(axis=1)
        p_bar = S / N
        p_err = numpy.stack(
            (p_bar - beta.ppf(CI / 2, S + 1, N - S + 1),
             beta.ppf(1 - CI / 2, S + 1, N - S + 1) - p_bar))
        pyplot.errorbar(data_i.index.mid, p_bar, yerr=p_err,
                        label=None, color=color,
                        marker='_', linestyle='dotted')
    pyplot.xlabel(data.index.name)
    pyplot.ylabel('Fraction susceptible')
    pyplot.legend(title='SAT')
    if show:
        pyplot.show()


if __name__ == '__main__':
    hazard_infection = find_hazard_infection()
    if len(hazard_infection) > 1:
        AIC = find_AIC(hazard_infection)
        print('Separate AIC = {:g}'.format(AIC.sum() - AIC['Pooled']))
        print('Pooled   AIC = {:g}'.format(AIC['Pooled']))
    plot(hazard_infection)
