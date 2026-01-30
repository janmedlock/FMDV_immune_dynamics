'''Find maximum-likelihood estimates of model parameters.'''

import numpy
import pandas
import scipy

import _multiprocessing_


def _minimize_global(func, theta_0,
                     bounds_diff=numpy.log(1e6), pool=None, **kwds):
    '''Wrapper to call `scipy.optimize.shgo()` with a signature like
    that of `scipy.optimize.minimize()`.'''
    # Add and subtract `bounds_diff` to each component of `theta_0`.
    bounds = (numpy.reshape(theta_0, (-1, 1))
              + numpy.array((-bounds_diff, bounds_diff)))
    with _multiprocessing_.Pool(pool) as pool_:
        return scipy.optimize.shgo(func, bounds, workers=pool_.map, **kwds)


def estimate(model, theta_0,
             global_=False, global_kwds=None, pool=None, **kwds):
    '''Find the maximum-likelihood estimate for the model parameters.'''
    if global_:
        minimize = _minimize_global
        if global_kwds is None:
            global_kwds = {}
        # Merge `kwds` into `global_kwds['minimizer_kwargs']`.
        global_kwds.setdefault('minimizer_kwargs', {}) \
                   .update(kwds)
        global_kwds['pool'] = pool
        # `global_kwds` is the `kwds` for `_minimizer_global()`.
        kwds = global_kwds
    else:
        minimize = scipy.optimize.minimize
    result = minimize(model.minus_log_likelihood, theta_0, **kwds)
    assert result.success, result
    theta_mle = pandas.Series(result.x,
                              index=model.parameter_index,
                              name='MLE')
    # Store Hessian or covariance, if available.
    try:
        theta_mle.fisher_information = result.hess
    except AttributeError:
        pass
    try:
        theta_mle.covariance = result.covariance
    except AttributeError:
        pass
    return theta_mle
