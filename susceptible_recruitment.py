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


class Birth(herd.birth.gen):
    '''Birth cut off after 1 year.'''
    def hazard(self, time, time0, age0):
        return numpy.where(time < 1,
                           super().hazard(time + time0, age0 + time),
                           0)

    def logsf(self, time, time0, age0):
        return numpy.where(time < 1,
                           super().logsf(time, time0, age0),
                           super().logsf(1, time0, age0))

    def pdf(self, time, time0, age0):
        return (self.hazard(time, time0, age0)
                * self.sf(time, time0, age0))

    def logpdf(self, time, time0, age0):
        return (numpy.log(self.hazard(time, time0, age0))
                + self.logsf(time, time0, age0))


class SusceptibleRecruitment(herd.rv.RV):
    '''The mixture of birth and maternal-immunity waning.'''
    def __init__(self, parameters, *args, **kwargs):
        self._birth = Birth(parameters)
        # self._birth = herd.birth.gen(parameters)
        self._maternal_immunity_waning = herd.maternal_immunity_waning.gen(
            parameters)

    def _pdf_integrand(self, s, t, t0, a0):
        return (self._birth.pdf(s, t0, a0)
                * self._maternal_immunity_waning.pdf(t - s))

    def pdf(self, t, t0, a0, maxiter=10000):
        '''The PDF is
        p_{susceptible}(t, t0, a0)
        = \int_{t0}^t p_{birth}(s, t0, a0) p_{waning}(t - s) ds.'''
        if numpy.isscalar(t):
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
        if numpy.isscalar(t):
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


if __name__ == '__main__':
    # 'hazard', 'pdf', 'cdf', or 'sf'.
    stat = operator.attrgetter('pdf')

    # Susceptible recruitment depends on time of year.
    # 2 years of dates, points every day.
    start = pandas.Timestamp(year=2001, month=7, day=1)
    end = start + pandas.DateOffset(years=2) - pandas.DateOffset(days=1)
    # end = start + pandas.DateOffset(years=5) - pandas.DateOffset(days=1)
    freq = 'D'
    t_susceptible = pandas.date_range(start, end, freq=freq)

    parameters = herd.Parameters()
    susceptible_recruitment = SusceptibleRecruitment(parameters)

    # For births, to remove age dependence.
    a0 = 4  # years.

    try:
        ser = pandas.read_csv('susceptible_recruitment.csv',
                              index_col=0, header=None, squeeze=True)
    except FileNotFoundError:
        t = timestamp2year(t_susceptible)
        ser = pandas.Series(
            stat(susceptible_recruitment)(t - t[0], t[0], a0),
            index=t_susceptible)
        ser.to_csv('susceptible_recruitment.csv')

    fig, ax = pyplot.subplots()

    ax.plot(t_susceptible, ser)
    ax.set_xlabel('Susceptible recruitment')
    ax.set_ylabel(r'Density ($\mathrm{y}^{-1}$)')
    # Major ticks every 6 months and minor ticks every month.
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    seaborn.despine(fig, top=True, right=True)

    fig.tight_layout()
    fig.savefig('susceptible_recruitment.png', dpi=300)
    fig.savefig('susceptible_recruitment.pdf')
    fig.savefig('susceptible_recruitment.pgf')

    pyplot.show()
