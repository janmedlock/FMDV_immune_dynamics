'''
Calculate PRCCs, etc.
'''
import numpy
import pandas
import scipy.stats


def confidence_interval(X, level, *args, **kwargs):
    '''
    Find confidence interval using quantiles.
    '''
    q = ((1 - level) / 2, (1 + level) / 2)
    return X.quantile(q, *args, **kwargs)


def _cc(X, y, method):
    rho = pandas.Series(index=X.columns)
    for (i, x) in X.items():
        rho[i] = y.corr(x, method=method)
    return rho


def cc(X, y):
    return _cc(X, y, 'pearson')


def rcc(X, y):
    return _cc(X, y, 'spearman')


def get_residuals(Z, b):
    # Add a column of ones for intercept term.
    A = numpy.column_stack((numpy.ones(len(Z)), Z))
    result = numpy.linalg.lstsq(A, b, rcond=None)
    coefs = result[0]
    residuals = b - A @ coefs
    return residuals


def pcc(X, y):
    rho = pandas.Series(index=X.columns)
    for (i, x) in X.items():
        # All of the other columns except `i`.
        Z = X.drop(columns=i)
        x_res = get_residuals(Z, x)
        y_res = get_residuals(Z, y)
        rho[i] = x_res.corr(y_res)
    return rho


def prcc(X, y):
    return pcc(X.rank(), y.rank())


def pcc_CI(rho, N, alpha=0.05):
    p = 1
    z = numpy.arctanh(rho)
    z_crit = scipy.stats.norm.ppf((alpha / 2, 1 - alpha / 2))
    z_CI = pandas.DataFrame(z[:, None] + z_crit / numpy.sqrt(N - p - 3),
                            index=rho.index,
                            columns=('lower', 'upper'))
    rho_CI = numpy.tanh(z_CI)
    return rho_CI


def prcc_CI(rho, N, alpha=0.05):
    return pcc_CI(rho, N, alpha=alpha)
