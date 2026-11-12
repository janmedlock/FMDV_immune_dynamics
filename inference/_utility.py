'''Utility functions.'''

import numpy
import scipy


def split(theta, i):
    '''Split `theta` into `theta_i` = theta[i] and
    `theta_not_i` = [theta[j]] for j != i.'''
    theta = numpy.asarray(theta)
    theta_i = theta[i]
    theta_not_i = numpy.hstack([theta[:i], theta[i+1:]])
    return (theta_i, theta_not_i)


def join(theta_i, theta_not_i, i):
    '''Combine `theta_i' and `theta_not_i` into `theta`.'''
    theta_not_i = numpy.asarray(theta_not_i)
    return numpy.hstack([theta_not_i[:i], theta_i, theta_not_i[i:]])


def log_sum_exp(a, b=None, axis=-1):
    '''log(sum(b * exp(a), axis=axis)) with careful broadcasting of a.'''
    a = numpy.stack(numpy.broadcast_arrays(*a),
                    axis=axis)
    return scipy.special.logsumexp(a, b=b, axis=axis)


def log_add_exp(x, y, axis=-1):
    '''log(exp(x) + exp(y)).'''
    return log_sum_exp((x, y), axis=axis)


def log_sub_exp(x, y, axis=-1):
    '''log(exp(x) - exp(y)).'''
    return log_sum_exp((x, y), b=(1, -1), axis=axis)


def exp_is_zero(x):
    '''Whether `exp(x) == 0`.'''
    return numpy.isneginf(x)


def choose_by_state(state, choices):
    '''Use `state` to choose among `choices`.'''
    # `state` is a categorical `pandas.Series()`.
    codes = state.cat.codes
    return numpy.choose(codes, choices)


def log_prob_is_valid(log_p):
    '''Whether `log_p` is a valid logarithm of a probability.'''
    return numpy.all(
        (log_p < 0)
        | numpy.isclose(log_p, 0)
        | numpy.isnan(log_p)
    )
