#!/usr/bin/python3

import joblib
from matplotlib import dates, pyplot, ticker
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
    def __init__(self, parameters, *args, **kwargs):
        self._birth = herd.birth.gen(parameters)
        self._maternal_immunity_waning = herd.maternal_immunity_waning.gen(
            parameters)

    def _pdf_integrand(self, s, t, t0, a0):
        return (self._birth.pdf(s, t0, a0)
                * self._maternal_immunity_waning.pdf(t - s))

    def _pdf_single(self, t, t0, a0, maxiter=10000):
        pdf, _ = integrate.quadrature(self._pdf_integrand, t0, t,
                                      args=(t, t0, a0),
                                      maxiter=maxiter)
        assert pdf >= 0
        return pdf

    def pdf(self, t, t0, a0):
        '''The PDF is
        p_{susceptible}(t, t0, a0)
        = \int_{t0}^t p_{birth}(s, t0, a0) p_{waning}(t - s) ds.'''
        return numpy.array(joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._pdf_single)(t_, t0, a0)
            for t_ in t))

    def _cdf_integrand(self, s, t, t0, a0):
        return (self._birth.pdf(s, t0, a0)
                * self._maternal_immunity_waning.cdf(t - s))

    def _cdf_single(self, t, t0, a0, maxiter=10000):
        cdf, _ = integrate.quadrature(self._cdf_integrand, t0, t,
                                      args=(t, t0, a0),
                                      maxiter=maxiter)
        assert 0 <= cdf <= 1
        return cdf

    def cdf(self, t, t0, a0):
        '''The CDF is
        P_{susceptible}(t, t0, a0)
        = \int_{t0}^t p_{susceptible}(s, t0, a0) ds
        = \int_{t0}^t p_{birth}(s, t0, a0) P_{waning}(t - s) ds.'''
        return numpy.array(joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._cdf_single)(t_, t0, a0)
            for t_ in t))

    def hazard(self, t, t0, a0):
        return self.pdf(t, t0, a0) / (1 - self.cdf(t, t0, a0))


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


if __name__ == '__main__':
    # Birth depends on time of year.
    # 1 year of dates, points every day.
    start = pandas.Timestamp(year=2001, month=1, day=1)
    end = start + pandas.DateOffset(years=1) - pandas.DateOffset(days=1)
    freq = 'D'
    t_birth = pandas.date_range(start, end, freq=freq)
    # Waning depends only on time since birth, not time of year.
    # 1 year of time intervals, points every day.
    t_waning = pandas.timedelta_range(0, periods=dates.DAYS_PER_YEAR, freq=freq)
    # Susceptible recruitment depends on time of year.
    # 2 years of dates, points every day.
    end = start + pandas.DateOffset(years=2) - pandas.DateOffset(days=1)
    t_susceptible = pandas.date_range(start, end, freq=freq)

    parameters = herd.Parameters()
    birth = herd.birth.gen(parameters)
    maternal_immunity_waning = herd.maternal_immunity_waning.gen(parameters)
    susceptible_recruitment = SusceptibleRecruitment(parameters)

    # For births, to remove age dependence.
    a0 = 4  # years.

    hazard = pandas.read_csv('susceptible_recruitment.csv')
    # hazard = {}
    # hazard['Birth'] = pandas.Series(
    #     birth.hazard(timestamp2year(t_birth), a0),
    #     index=t_birth)
    # # Convert the index for waning from a `pandas.TimedeltaIndex` to a
    # # `pandas.DatetimeIndex` like the other 2.
    # hazard['Maternal-immunity waning'] = pandas.Series(
    #     maternal_immunity_waning.hazard(timedelta2year(t_waning)),
    #     index=t_waning + start)
    # hazard['Susceptible recruitment'] = pandas.Series(
    #     susceptible_recruitment.hazard(timestamp2year(t_susceptible),
    #                                    timestamp2year(t_susceptible[0]), a0),
    #     index=t_susceptible)
    # hazard = pandas.DataFrame(hazard)
    # hazard.to_csv('susceptible_recruitment.csv')

    fig, ax = pyplot.subplots(1, 3)

    ax[0].plot(t_birth, hazard['Birth'].dropna(),
               color='C0', clip_on=False)
    ax[0].set_ylabel('Birth')
    # Major ticks every 6 months and minor ticks every month.
    ax[0].xaxis.set_major_locator(MonthLocator(interval=6))
    ax[0].xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
    ax[0].xaxis.set_major_formatter(dates.DateFormatter('%b'))

    ax[1].plot(timedelta2year(t_waning),
               hazard['Maternal-immunity waning'].dropna(),
               color='C1', clip_on=False)
    ax[1].set_ylabel('Maternal-immunity waning')
    # Major ticks automatic and minor ticks every month.
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1 / 12))
    # '\u2009' is thin space.
    ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}\u2009y'))

    ax[2].plot(t_susceptible, hazard['Susceptible recruitment'].dropna(),
               color='C2', clip_on=False)
    ax[2].set_ylabel('Susceptible recruitment')
    # Major ticks every 6 months and minor ticks every month.
    ax[2].xaxis.set_major_locator(MonthLocator(interval=6))
    ax[2].xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
    ax[2].xaxis.set_major_formatter(dates.DateFormatter('%b'))

    for ax_ in ax:
        ax_.yaxis.set_major_locator(ticker.NullLocator())
        ax_.set_ylim(bottom=0)
    seaborn.despine(fig, top=True, right=True)

    fig.tight_layout()
    fig.savefig('susceptible_recruitment.png', dpi=300)
    fig.savefig('susceptible_recruitment.pdf')
    fig.savefig('susceptible_recruitment.pgf')

    pyplot.show()
