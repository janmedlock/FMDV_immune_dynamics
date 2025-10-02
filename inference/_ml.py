'''Find maximum-likelihood estimates of model parameters.'''

import numpy
import pandas
import scipy


def _minimizer_global(func, theta_0,
                      bounds=(-20, 20), sampling_method='sobol', workers=-1,
                      **kwds):
    '''Wrapper to call `scipy.optimize.shgo()` with a signature like
    that of `scipy.optimize.minimize()` and to tweak some defaults.'''
    if numpy.ndim(bounds) < numpy.ndim(theta_0) + 1:
        bounds = (bounds, ) * len(theta_0)
    return scipy.optimize.shgo(func, bounds,
                               sampling_method=sampling_method,
                               workers=workers,
                               **kwds)


def estimate(model, theta_0, jac='3-point',
             global_=False, global_kwds=None,
             **kwds):
    '''Find the maximum-likelihood estimate for the model parameters.'''
    kwds['jac'] = jac
    if global_:
        minimizer = _minimizer_global
        if global_kwds is None:
            global_kwds = {}
        # Merge `kwds` into `global_kwds['minimizer_kwargs']`.
        global_kwds.setdefault('minimizer_kwargs', {}) \
                   .update(kwds)
        # `global_kwds` is the `kwds` for `_minimizer_global()`.
        kwds = global_kwds
    else:
        minimizer = scipy.optimize.minimize
    result = minimizer(model.minus_log_likelihood, theta_0, **kwds)
    assert result.success, result
    theta_mle = pandas.Series(result.x,
                              index=model.parameter_index,
                              name='MLE')
    # Store Hessian and covariance, if available.
    try:
        theta_mle.fisher_information = result.hess
    except AttributeError:
        pass
    try:
        theta_mle.covariance = result.covariance
    except AttributeError:
        pass
    return theta_mle
