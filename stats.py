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


def _get_d_dn(observed):
    # The number of events at this time,
    # i.e. entries in `observed` that are `True`.
    d = observed[observed].count()
    # The number of people lost due to events or lost to followup at this time.
    dn = observed.count()
    return pandas.Series((d, dn))


def get_survival(df, time, observed):
    by_time = df.groupby(time)
    # Get the number of events and total number of people lost at each time.
    d_dn = by_time[observed].apply(_get_d_dn).unstack()
    (d, dn) = (col for _, col in d_dn.items())
    # Shift `dn` so that `n` is the number of people surviving
    # up to time t (< t), *not* up to and including time t (<= t).
    # Nobody is yet lost before the first time.
    dn[:] = numpy.hstack([0, dn.iloc[:-1]])
    # Total people at start.
    N = len(df)
    n = N - dn.cumsum()
    # S(t) = \prod_{i: t_i <= t}  (1 - d_i / n_i), but use log for accuracy.
    S = (1 - d / n).apply(numpy.log).cumsum().apply(numpy.exp)
    # Add point S(0) = 1.
    S_0 = pandas.Series(1, index=pandas.Index([0], name=S.index.name))
    assert S.index.min() >= S_0.index.max()
    S = pandas.concat((S_0 ,S))
    S.name = 'survival'
    return S
