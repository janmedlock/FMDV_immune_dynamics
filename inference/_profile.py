'''Find the profile log likelihood.'''

import scipy

import _utility


def _minus_log_likelihood(theta_not_i, theta_i, i, model):
    '''Helper function for `log_likelihood()`.'''
    theta = _utility.join(theta_i, theta_not_i, i)
    return model.minus_log_likelihood(theta)


def log_likelihood(model, theta, i, jac='3-point', **kwds):
    '''Find the maximum likelihood of `theta_i` over `theta_j` for j != i:
    max_{theta_j for j !=i} l(theta_0, theta_1, ...)'''
    (theta_i, theta_not_i_0) = _utility.split(theta, i)
    result = scipy.optimize.minimize(_minus_log_likelihood,
                                     theta_not_i_0, args=(theta_i, i, model),
                                     jac=jac, **kwds)
    # Do NOT check `result.success` because a flaky solution is
    # better than raising an error here.
    # result.fun is the *minus* log likelihood.
    return -result.fun
