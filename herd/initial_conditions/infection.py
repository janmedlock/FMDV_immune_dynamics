'''Estimate the infection hazard from the Hedger 1972 data.'''

import os.path
import re

from joblib import Memory
import numpy
import pandas
from scipy.optimize import minimize

from herd import parameters
from herd.initial_conditions import state_probabilities


_filename = '../data/Hedger_1972_survey_data.xlsx'
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


def minus_loglikelihood(hazard_infection, params, data):
    '''This gets optimized over `hazard_infection`.'''
    # Convert the length-1 `hazard_infection` to a scalar Python float,
    # (not a size-() `numpy.array()`).
    hazard_infection = numpy.squeeze(hazard_infection)[()]
    l = 0
    # Loop over age groups.
    for age_interval, data_age in data.iterrows():
        # Consider doing something better to get the
        # representative age for an age interval.
        age = age_interval.mid
        M_logprob = state_probabilities.M_logprob(age,
                                                  hazard_infection,
                                                  params)
        # Avoid 0 * -inf.
        l += ((data_age['M'] * M_logprob)
              if (data_age['M'] > 0)
              else 0)
        S_logprob = state_probabilities.S_logprob(age,
                                                  hazard_infection,
                                                  params)
        # Avoid 0 * -inf.
        l += ((data_age['S'] * S_logprob)
              if (data_age['S'] > 0)
              else 0)
        # R_not_prob = 1 - R_prob.
        R_not_prob = numpy.exp(M_logprob) + numpy.exp(S_logprob)
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


# The function is very slow because of the call to
# `scipy.optimize.minimize()`, so the values are cached to disk with
# `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir, verbose=0)
@_cache.cache
def _find_hazard(params):
    '''Find the MLE for the infection hazard.'''
    data = _load_data(params)
    x0 = 0.4
    res = minimize(minus_loglikelihood, x0,
                   args=(params, data),
                   bounds=[(0, numpy.inf)])
    assert res.success, res
    # Convert the length-1 array `res.x` to a scalar Python float,
    # (not a size-() `numpy.array()`).
    return numpy.squeeze(res.x)[()]


class CacheParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `_find_hazard()`
    so that it can be efficiently cached.'''
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


def find_hazard(params):
    '''Find the MLE for the infection hazard for each SAT.'''
    return _find_hazard(CacheParameters(params))


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
    lines = pyplot.plot(ages,
                        state_probabilities.S_prob(ages,
                                                   hazard_infection,
                                                   params))
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
