#!/usr/bin/python3

import operator

import joblib
from matplotlib import dates, gridspec, pyplot, ticker
import numpy
import pandas
from scipy import integrate
import seaborn

import herd
import herd.rv
import herd.birth
import herd.maternal_immunity_waning


class SusceptibleRecruitment(herd.rv.RV):
    '''The mixture of birth and maternal-immunity waning.'''
    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            parameters = Parameters()
        self._birth = Birth(parameters)
        self._birth = herd.birth.gen(parameters)
        self._maternal_immunity_waning = herd.maternal_immunity_waning.gen(
            parameters)

    def _pdf_integrand(self, s, t, t0, a0):
        return (self._birth.pdf(s, t0, a0)
                * self._maternal_immunity_waning.pdf(t - s))

    def pdf(self, t, t0, a0, maxiter=10000):
        '''The PDF is
        p_{susceptible}(t, t0, a0)
        = \int_{t0}^t p_{birth}(s, t0, a0) p_{waning}(t - s) ds.'''
        if numpy.ndim(t) == 0:
            pdf, _ = integrate.quadrature(self._pdf_integrand, t0, t,
                                          args=(t, t0, a0),
                                          maxiter=maxiter)
            assert pdf >= 0
            return pdf
        else:
            return numpy.array(joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self.pdf)(t_, t0, a0)
                for t_ in t))

    def _cdf_integrand(self, s, t, t0, a0):
        return (self._birth.pdf(s, t0, a0)
                * self._maternal_immunity_waning.cdf(t - s))

    def cdf(self, t, t0, a0, maxiter=10000):
        '''The CDF is
        P_{susceptible}(t, t0, a0)
        = \int_{t0}^t p_{susceptible}(s, t0, a0) ds
        = \int_{t0}^t p_{birth}(s, t0, a0) P_{waning}(t - s) ds.'''
        if numpy.ndim(t) == 0:
            cdf, _ = integrate.quadrature(self._cdf_integrand, t0, t,
                                          args=(t, t0, a0),
                                          maxiter=maxiter)
            assert 0 <= cdf <= 1
            return cdf
        else:
            return numpy.array(joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self.cdf)(t_, t0, a0)
                for t_ in t))

    def sf(self, t, t0, a0):
        return 1 - self.cdf(t, t0, a0)

    def hazard(self, t, t0, a0):
        return self.pdf(t, t0, a0) / self.sf(t, t0, a0)


def timestamp2year(ts):
    '''Convert a `pandas.Timestamp` or `pandas.DatetimeIndex`
    to a float year.'''
    try:
        t0 = ts[0]
    except TypeError:
        t0 = ts
    return (ts.dayofyear - 1) / dates.DAYS_PER_YEAR + ts.year - t0.year


def timedelta2year(ts):
    '''Convert a `pandas.Timedelta` or `pandas.TimedeltaIndex`
    to a float year.'''
    return ts.days / dates.DAYS_PER_YEAR


def MonthLocator(interval=1):
    '''Like `matplotlib.dates.MonthLocator()` but always start on January.'''
    bymonth = range(1, 12 + 1, interval)
    return dates.MonthLocator(bymonth)


def get_susceptible_recruitment():
    filename = 'susceptible_recruitment.csv'
    try:
        ser = pandas.read_csv(filename,
                              index_col=0, parse_dates=True,
                              header=None, squeeze=True)
    except FileNotFoundError:
        susceptible_recruitment = SusceptibleRecruitment()
        # Susceptible recruitment depends on time of year.
        # 2 years of dates, points every day.
        start = pandas.Timestamp(year=2001, month=7, day=1)
        end = start + pandas.DateOffset(years=2) - pandas.DateOffset(days=1)
        t_susceptible = pandas.date_range(start, end, freq='D')
        t = timestamp2year(t_susceptible)
        a0 = 4  # years, to remove age dependence from births.
        ser = pandas.Series(
            susceptible_recruitment.pdf(t - t[0], t[0], a0),
            index=t_susceptible)
        ser.to_csv(filename)
    return ser


def plot_susceptible_recruitment(ser):
    fig, ax = pyplot.subplots(constrained_layout=True)
    ax.plot(ser)
    ax.set_xlabel('Susceptible recruitment')
    ax.set_ylabel(r'Density ($\mathrm{y}^{-1}$)')
    # Major ticks every 3 months and minor ticks every month.
    n = 3
    ax.xaxis.set_major_locator(MonthLocator(interval=n))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n))
    # Label month.
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    seaborn.despine(fig, top=True, right=True)
    # fig.savefig('susceptible_recruitment.png', dpi=300)
    # fig.savefig('susceptible_recruitment.pdf')
    # fig.savefig('susceptible_recruitment.pgf')


if __name__ == '__main__':
    ser = get_susceptible_recruitment()
    plot_susceptible_recruitment(ser)
    pyplot.show()
