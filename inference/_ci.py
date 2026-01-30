'''Find confidence intervals for model parameters using the profile
log likelihood.'''

import functools
import itertools

import numpy
import pandas
import scipy

import _multiprocessing_
import _profile
import _utility


def _objective(theta_i, model, ll_ci, theta_not_i_0, i):
    '''Helper function for `_solve()` and `_bracket()`. This function
    is 0 when the profile log likelihood of `theta_i` is equal to
    `l_CI.`'''
    theta = _utility.join(theta_i, theta_not_i_0, i)
    ll = _profile.log_likelihood(model, theta, i)
    return ll - ll_ci


def _bracket(theta_i, model, ll_ci, theta_not_i, i, which,
             scaling_initial=1.2, scaling_factor=3):
    '''Find a bracketing interval of the root of
    `_objective(theta_i, ...)` so that
    `_objective(above, ...)` > 0 and
    `_objective(below, ...)` < 0. '''
    # `_objective(theta_i, ...)` > 0, so start with `below` = `theta_i`.
    below = theta_i
    # In each step, set `above` = `below` then move `below` until we find
    # `_objective(below, ...)` < 0.
    scaling = scaling_initial
    # For the upper CI, `below` is the right endpoint.
    if which == 'CI_lower':
        # For the lower CI, `below` is the left endpoint.
        scaling *= -1
    found = False
    while True:
        above = below  # `_objective(above, ...)` > 0.
        below += scaling
        if numpy.isinf(below):
            break
        if _objective(below, model, ll_ci, theta_not_i, i) < 0:
            found = True
            break
        if not numpy.isfinite(scaling):
            break
        scaling *= scaling_factor
    return (found, above, below)


def _solve(model, ll_ci, theta, i_which):
    '''Helper function for `ci()`.  This is used to find the value
    of `theta_i` where the profile log likelihood is equal to `ll_ci.`'''
    (i, which) = i_which
    (theta_i, theta_not_i_0) = _utility.split(theta, i)
    (found, above, below) = _bracket(theta_i, model, ll_ci, theta_not_i_0,
                                     i, which)
    if found:
        result = scipy.optimize.root_scalar(_objective,
                                            bracket=(above, below),
                                            args=(model, ll_ci,
                                                  theta_not_i_0, i))
        assert result.converged, result
        return result.root
    # `below` probably hit infinity before the log-likelihood
    # dropped below `ll_ci`.
    return below


_COLUMNS = ('CI_lower', 'CI_upper')


def estimate(model, theta, alpha=0.05, pool=None):
    '''Get the confidence interval using the profile log likelihood.'''
    # For each i, find the value of theta_i where
    # the profile log likelihod is chi2 / 2 less than
    # the maximum log likelihood.
    # chi2 is the value of a standard chi^2(1) r.v.
    # at the (1 - alpha) quantile.
    chi2 = scipy.stats.chi2.ppf(1 - alpha, 1)
    # The target log likelihood value.
    ll_ci = model.log_likelihood(theta) - chi2 / 2
    # Fix all but the last parameter of `_solve()`.
    solve = functools.partial(_solve, model, ll_ci, theta)
    # Iterate over `i`, the parameter to estimate, and
    # `which`, the lower or upper side of the CI.
    i_which_vals = itertools.product(range(len(theta)), _COLUMNS)
    with _multiprocessing_.Pool(pool) as pool_:
        ci = pool_.map(solve, i_which_vals)
    ci = numpy.reshape(ci, (len(theta), -1))
    return pandas.DataFrame(ci,
                            index=model.parameter_index,
                            columns=_COLUMNS)
